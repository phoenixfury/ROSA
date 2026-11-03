"""Single-frame dual-metric stopping; no temporal model/data dependencies."""
import copy
import json
import math
from pathlib import Path
import time

from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS


class DualMetricPlateau:
    """Wait until neither metric has meaningfully improved for patience epochs."""

    keys = ('fisheye_bev/mAP', 'fisheye_bev/NDS')

    def __init__(self, patience=4, min_delta=1e-4):
        if patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('patience must be positive and min_delta finite/nonnegative')
        self.patience, self.min_delta = patience, min_delta
        self.history, self.best, self.wait = [], {}, 0

    def update(self, epoch, metrics):
        if self.history and epoch <= self.history[-1]['epoch']:
            raise ValueError('Validation epochs must be strictly increasing')
        values = {key: float(metrics[key]) for key in self.keys}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('Non-finite validation mAP/NDS; refusing a false plateau decision')
        improved = []
        for key, value in values.items():
            if key not in self.best or value > self.best[key]['value'] + self.min_delta:
                self.best[key] = dict(value=value, epoch=epoch)
                improved.append(key)
        self.wait = 0 if improved else self.wait + 1
        row = dict(epoch=epoch, metrics=values, improved=improved, wait=self.wait)
        self.history.append(row)
        return self.wait >= self.patience

    def state_dict(self):
        return copy.deepcopy(dict(patience=self.patience, min_delta=self.min_delta,
                                  history=self.history, best=self.best, wait=self.wait))

    def restore(self, state, completed_epoch):
        if state['patience'] != self.patience or state['min_delta'] != self.min_delta:
            raise ValueError('Resume must retain the original stopping policy')
        self.history, self.best, self.wait = [], {}, 0
        # Periodic checkpoints precede validation. A sidecar can supply the
        # matching epoch's completed validation, never a future epoch's result.
        for row in state['history']:
            if row['epoch'] <= completed_epoch:
                self.update(row['epoch'], row['metrics'])


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@HOOKS.register_module()
class SingleFramePlateauHook(Hook):
    # CheckpointHook has already retained the best mAP/NDS weights when we stop.
    priority = 'LOWEST'
    state_key = 'singleframe_dual_metric_plateau'

    def __init__(self, patience=4, min_delta=1e-4, progress_interval=50):
        self.monitor = DualMetricPlateau(patience, min_delta)
        self.progress_interval = progress_interval
        self.started = None
        self.stop_reason = None

    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if (getattr(model.head, 'temporal_enabled', False) or
                model.head.instance_bank.num_temp_instances != 0):
            raise RuntimeError('Single-frame experiment must not enable temporal slots')
        if hasattr(model, 'img_backbone'):
            if any(not parameter.requires_grad for parameter in model.img_backbone.parameters()):
                raise RuntimeError('This refresh requires a fully trainable backbone')
            runner.logger.info('Single-frame refresh: all backbone parameters trainable; '
                               'temporal slots=0; full decoder losses enabled.')
        self.started, self.start_iter = time.monotonic(), runner.iter
        if getattr(runner, '_resume', False):
            state = runner.message_hub.get_info(self.state_key)
            sidecar = Path(runner.work_dir) / 'plateau.json'
            if sidecar.exists():
                state = json.loads(sidecar.read_text())
            if state:
                self.monitor.restore(state, runner.epoch)
            if self.monitor.wait >= self.monitor.patience:
                runner.train_loop.stop_training = True
                self.stop_reason = 'dual_metric_plateau'
        self._persist(runner)

    def _persist(self, runner):
        state = self.monitor.state_dict()
        runner.message_hub.update_info(self.state_key, state)
        if is_main_process():
            write_json(Path(runner.work_dir) / 'plateau.json', state)

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint[self.state_key] = self.monitor.state_dict()

    def after_val_epoch(self, runner, metrics=None):
        stop = self.monitor.update(runner.epoch, metrics or {})
        self._persist(runner)
        runner.logger.info('Single-frame plateau: mAP=%.4f NDS=%.4f; wait=%d/%d; '
                           'either metric improving resets the wait.',
                           metrics[self.monitor.keys[0]], metrics[self.monitor.keys[1]],
                           self.monitor.wait, self.monitor.patience)
        if stop:
            self.stop_reason = 'dual_metric_plateau'
            runner.train_loop.stop_training = True
            runner.logger.info('Stopping: neither mAP nor NDS improved by >%.5f '
                               'for %d validations.', self.monitor.min_delta,
                               self.monitor.patience)
        self._progress(runner, 'stopping' if stop else 'running')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (runner.iter + 1) % self.progress_interval == 0:
            self._progress(runner, 'running', completed_iters=runner.iter + 1)

    def _progress(self, runner, status, completed_iters=None):
        if not is_main_process():
            return
        completed_iters = runner.iter if completed_iters is None else completed_iters
        elapsed = time.monotonic() - self.started
        completed_here = completed_iters - self.start_iter
        eta = ((runner.max_iters - completed_iters) * elapsed / completed_here
               if completed_here > 0 else None)
        write_json(Path(runner.work_dir) / 'progress.json', dict(
            status=status, epoch=runner.epoch, iteration=completed_iters,
            max_epochs=runner.max_epochs, elapsed_seconds=elapsed,
            eta_to_epoch_cap_seconds=eta,
            note='Actual finish may be earlier: stop when BOTH metrics plateau.',
            stopping_policy=dict(patience=self.monitor.patience,
                                 min_delta=self.monitor.min_delta),
            best_meaningful=self.monitor.best, plateau_wait=self.monitor.wait,
            latest=self.monitor.history[-1] if self.monitor.history else None))

    def after_train(self, runner):
        self._persist(runner)
        reason = self.stop_reason or 'epoch_cap'
        # after_val_epoch uses completed epochs, unlike after_train_epoch.
        # Explicit metadata avoids accidentally resuming at epoch N+1.
        filename = f'{reason}_epoch_{runner.epoch}.pth'
        runner.save_checkpoint(runner.work_dir, filename,
                               meta=dict(epoch=runner.epoch, iter=runner.iter),
                               by_epoch=True)
        if is_main_process():
            root = Path(runner.work_dir)
            (root / 'last_checkpoint').write_text(str(root / filename))
            best_paths = {key: runner.message_hub.get_info('best_ckpt_' + key)
                          for key in self.monitor.keys}
            results = dict(status='complete', stop_reason=reason,
                           completed_epochs=runner.epoch, completed_iterations=runner.iter,
                           policy=self.monitor.state_dict(), best_checkpoints=best_paths,
                           best_raw_scores={key: runner.message_hub.get_info('best_score_' + key)
                                            for key in self.monitor.keys},
                           resume_checkpoint=str(root / filename))
            write_json(root / 'results.json', results)
            lines = ['# E007: non-temporal refresh', '', f'Stopped: {reason}.',
                     f'Completed epochs: {runner.epoch}.', '',
                     'Fresh optimizer; full backbone/decoder trainable; no temporal slots.',
                     'Validation targets are pseudo-labels, not an independent manual test.', '',
                     '| Epoch | mAP | NDS | Plateau wait |', '|---|---|---|---|']
            for row in self.monitor.history:
                lines.append('| {} | {:.6f} | {:.6f} | {} |'.format(
                    row['epoch'], row['metrics'][self.monitor.keys[0]],
                    row['metrics'][self.monitor.keys[1]], row['wait']))
            lines += ['', 'Native CheckpointHook retains the best raw mAP/NDS '
                      'checkpoints separately; plateau uses meaningful improvements.']
            (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        self._progress(runner, 'complete')
