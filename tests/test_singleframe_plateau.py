"""Run in openmmlab with repository/ROSA/Fisheye3DOD on PYTHONPATH."""
from types import SimpleNamespace

import pytest

from rosa.integrations.fisheye3dod.singleframe_plateau import (
    DualMetricPlateau, SingleFramePlateauHook)


def metrics(ap, nds):
    return {'fisheye_bev/mAP': ap, 'fisheye_bev/NDS': nds}


def test_either_metric_resets_wait():
    monitor = DualMetricPlateau(patience=2)
    assert not monitor.update(1, metrics(.1, .2))
    assert not monitor.update(2, metrics(.09, .19))
    assert not monitor.update(3, metrics(.11, .19))
    assert not monitor.update(4, metrics(.1, .21))
    assert not monitor.update(5, metrics(.109, .209))
    assert monitor.update(6, metrics(.11, .21))


def test_small_improvements_accumulate_against_meaningful_best():
    monitor = DualMetricPlateau(patience=4, min_delta=.001)
    monitor.update(1, metrics(.1, .2))
    monitor.update(2, metrics(.1005, .2))
    assert monitor.wait == 1
    monitor.update(3, metrics(.1011, .2))
    assert monitor.wait == 0


@pytest.mark.parametrize('values', [metrics(float('nan'), .2), metrics(.1, float('inf')), {}])
def test_invalid_metrics_fail_loudly(values):
    with pytest.raises((ValueError, KeyError)):
        DualMetricPlateau().update(1, values)


def test_resume_never_imports_future_epochs():
    previous = DualMetricPlateau(patience=2)
    previous.update(1, metrics(.1, .2))
    previous.update(2, metrics(.1, .2))
    previous.update(3, metrics(.11, .22))
    restored = DualMetricPlateau(patience=2)
    restored.restore(previous.state_dict(), completed_epoch=2)
    assert restored.wait == 1
    assert len(restored.history) == 2
    assert restored.update(3, metrics(.1, .2))


def test_resume_policy_must_not_change():
    with pytest.raises(ValueError):
        DualMetricPlateau(patience=3).restore(DualMetricPlateau().state_dict(), 1)


def test_duplicate_epochs_rejected():
    monitor = DualMetricPlateau()
    monitor.update(1, metrics(.1, .2))
    with pytest.raises(ValueError):
        monitor.update(1, metrics(.2, .3))


class Hub:
    def __init__(self):
        self.info = {}

    def update_info(self, key, value):
        self.info[key] = value

    def get_info(self, key):
        return self.info.get(key)


def fake_runner(tmp_path):
    saved = []
    runner = SimpleNamespace(work_dir=str(tmp_path), epoch=0, iter=0,
        max_iters=100, max_epochs=10, _resume=False, message_hub=Hub(),
        logger=SimpleNamespace(info=lambda *args: None),
        train_loop=SimpleNamespace(stop_training=False),
        save_checkpoint=lambda *args, **kwargs: saved.append((args, kwargs)),
        model=SimpleNamespace(head=SimpleNamespace(temporal_enabled=False,
            instance_bank=SimpleNamespace(num_temp_instances=0))))
    return runner, saved


def test_hook_stop_save_and_completed_epoch_metadata(tmp_path):
    hook = SingleFramePlateauHook(patience=2)
    runner, saved = fake_runner(tmp_path)
    hook.before_train(runner)
    for epoch in (1, 2, 3):
        runner.epoch, runner.iter = epoch, epoch * 10
        hook.after_val_epoch(runner, metrics(.1, .2))
    assert runner.train_loop.stop_training
    hook.after_train(runner)
    assert saved[0][1]['meta'] == dict(epoch=3, iter=30)
    assert (tmp_path / 'last_checkpoint').read_text().endswith('dual_metric_plateau_epoch_3.pth')
    assert (tmp_path / 'results.json').is_file()
    assert (tmp_path / 'REPORT.md').is_file()


def test_hook_resumes_validation_sidecar(tmp_path):
    first = SingleFramePlateauHook(patience=2)
    runner, _ = fake_runner(tmp_path)
    first.before_train(runner)
    for epoch in (1, 2):
        runner.epoch = epoch
        first.after_val_epoch(runner, metrics(.1, .2))
    runner._resume = True
    resumed = SingleFramePlateauHook(patience=2)
    resumed.before_train(runner)
    assert resumed.monitor.wait == 1
    runner.epoch = 3
    resumed.after_val_epoch(runner, metrics(.1, .2))
    assert runner.train_loop.stop_training


def test_temporal_model_cannot_silently_train(tmp_path):
    runner, _ = fake_runner(tmp_path)
    runner.model.head.instance_bank.num_temp_instances = 100
    with pytest.raises(RuntimeError):
        SingleFramePlateauHook().before_train(runner)


def test_real_runner_stops_and_saves_both_best_checkpoints(tmp_path, monkeypatch):
    import torch
    from torch.utils.data import DataLoader
    from mmengine.evaluator import BaseMetric, Evaluator
    from mmengine.model import BaseModel
    from mmengine.runner import Runner

    class TinyModel(BaseModel):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.head = SimpleNamespace(temporal_enabled=False,
                instance_bank=SimpleNamespace(num_temp_instances=0))

        def forward(self, inputs, data_samples=None, mode='tensor'):
            if mode == 'loss':
                return dict(loss=(self.weight * inputs).square().mean())
            return [dict(value=0)]

    class ConstantMetric(BaseMetric):
        default_prefix = 'fisheye_bev'

        def process(self, data_batch, data_samples):
            self.results.append(0)

        def compute_metrics(self, results):
            return dict(mAP=.1, NDS=.2)

    # Exercise checkpoint hooks and epoch counting on CPU, not the training GPU.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    loader = DataLoader([dict(inputs=torch.ones(1))], batch_size=1)
    runner = Runner(model=TinyModel(), work_dir=str(tmp_path),
        train_dataloader=loader,
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1),
        optim_wrapper=dict(optimizer=dict(type='SGD', lr=.01)),
        val_dataloader=loader, val_cfg=dict(type='ValLoop'),
        val_evaluator=Evaluator(ConstantMetric()),
        custom_hooks=[SingleFramePlateauHook(patience=2)],
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1,
            save_best=['fisheye_bev/mAP', 'fisheye_bev/NDS'], rule=['greater', 'greater'])),
        env_cfg=dict(dist_cfg=dict(backend='gloo')))
    runner.train()
    assert runner.epoch == 3
    assert len(list(tmp_path.glob('best_fisheye_bev_mAP_epoch_*.pth'))) == 1
    assert len(list(tmp_path.glob('best_fisheye_bev_NDS_epoch_*.pth'))) == 1
    final = torch.load(tmp_path / 'dual_metric_plateau_epoch_3.pth', map_location='cpu')
    assert final['meta']['epoch'] == 3
    assert final['meta']['iter'] == 3
    assert final['singleframe_dual_metric_plateau']['wait'] == 2
