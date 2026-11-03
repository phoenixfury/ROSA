"""Frozen-weight memory selection: geometry, gating and query alignment."""
import unittest

from test_temporal_training import metadata, np, torch, TemporalInstanceBank


def make_bank(count=5, capacity=3, **kwargs):
    anchors = np.zeros((count, 11), dtype=np.float32)
    anchors[:, 0] = np.arange(count, dtype=np.float32) * 10. + 5.
    anchors[:, 3:6] = np.log([2., 4., 1.5])
    anchors[:, 7] = 1.
    options = dict(num_anchor=count, embed_dims=4, anchor=anchors,
                   num_temp_instances=capacity, min_confidence=.1,
                   max_time_interval=1.5,
                   anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'))
    options.update(kwargs)
    return TemporalInstanceBank(**options)


def proposals(memory, scores, quality_scores=None):
    """Sentinel feature channels retain the original free-query index."""
    features = torch.arange(memory.num_anchor, dtype=torch.float32)[None, :, None]
    features = features.expand(1, -1, memory.embed_dims).clone()
    logits = torch.logit(torch.tensor(scores, dtype=torch.float32))[None]
    if logits.ndim == 2:
        logits = logits[..., None]
    quality = None
    if quality_scores is not None:
        values = torch.tensor(quality_scores, dtype=torch.float32)
        quality = torch.stack((torch.logit(values), torch.zeros_like(values)), -1)[None]
    return features, memory.anchor[None].detach().clone(), logits, quality


def store(memory, values, meta=None):
    features, anchors, logits, quality = values
    memory.cache(features, anchors, logits, metadata() if meta is None else meta,
                 quality=quality)


class TemporalMemorySelectionTests(unittest.TestCase):
    def test_legacy_defaults_preserve_raw_class_topk_and_history(self):
        memory = make_bank()
        memory.get(1, metadata())
        store(memory, proposals(memory, [.8, .95, .2, .4, .01], [.99, .01, .5, .5, .5]))
        self.assertEqual(memory.cached_feature[0, :, 0].tolist(), [1., 0., 3.])
        memory.get(1, metadata('f1', 'f0', timestamp=1_789_000_000.5))
        second = proposals(memory, [.001]*5, [.01]*5)
        store(memory, second, metadata('f1', 'f0', timestamp=1_789_000_000.5))
        self.assertTrue(memory.cached_slots_valid.all())

    def test_quality_changes_ranking_without_misaligned_features_or_anchors(self):
        memory = make_bank(cache_score_mode='quality', cache_require_current=True)
        memory.get(1, metadata())
        values = proposals(memory, [.99, .8, .7, .01, .01], [.12, .9, .8, .9, .9])
        store(memory, values)
        self.assertEqual(memory.cached_feature[0, :, 0].tolist(), [1., 2., 0.])
        torch.testing.assert_close(memory.cached_anchor[0, :, 0],
                                   torch.tensor([15., 25., 5.]))
        torch.testing.assert_close(memory.confidence[0], torch.tensor([.72, .56, .1188]))

    def test_rotated_nms_is_classwise_and_preserves_query_alignment(self):
        memory = make_bank(cache_nms_iou_threshold=.5, cache_require_current=True)
        memory.get(1, metadata())
        values = proposals(memory, [[.95, .01], [.9, .01], [.01, .85],
                                   [.01, .01], [.01, .01]])
        values[1][0, :3, 0] = 10.
        # Non-axis-aligned duplicate boxes exercise sin/cos-to-yaw decoding.
        values[1][0, :3, 6:8] = torch.tensor([np.sin(.7), np.cos(.7)])
        store(memory, values)
        valid_features = memory.cached_feature[0, memory.cached_slots_valid[0], 0]
        self.assertEqual(valid_features.tolist(), [0., 2.])
        self.assertEqual(memory.cached_anchor[0, memory.cached_slots_valid[0], 0].tolist(),
                         [10., 10.])

    def test_unused_memory_slots_are_fresh_queries_not_suppressed_duplicates(self):
        memory = make_bank(cache_nms_iou_threshold=.5, cache_require_current=True)
        memory.get(1, metadata())
        values = proposals(memory, [.95, .9, .85, .01, .01])
        values[1][0, :3, 0] = 10.
        store(memory, values)
        features, anchors, _, _, _ = memory.get(
            1, metadata('f1', 'f0', timestamp=1_789_000_000.5))
        self.assertEqual(int(memory.valid_slots.sum()), 1)
        invalid = ~memory.valid_slots[0]
        torch.testing.assert_close(features[0, :3][invalid], memory.instance_feature[:3][invalid])
        torch.testing.assert_close(anchors[0, :3][invalid], memory.anchor[:3][invalid])

    def test_history_does_not_resurrect_current_low_quality(self):
        memory = make_bank(cache_score_mode='quality', cache_require_current=True)
        memory.get(1, metadata())
        store(memory, proposals(memory, [.99, .98, .97, .01, .01], [.99]*5))
        second_meta = metadata('f1', 'f0', timestamp=1_789_000_000.5)
        memory.get(1, second_meta)
        store(memory, proposals(memory, [.99]*5, [.001]*5), second_meta)
        self.assertFalse(memory.cached_slots_valid.any())

    def test_history_does_not_resurrect_current_nms_rejection(self):
        memory = make_bank(cache_require_current=True)
        memory.get(1, metadata())
        values = proposals(memory, [.99, .98, .97, .01, .01])
        values[1][0, :3, 0] = 10.
        store(memory, values)
        memory.cache_nms_iou_threshold = .5
        second_meta = metadata('f1', 'f0', timestamp=1_789_000_000.5)
        memory.get(1, second_meta)
        values = proposals(memory, [.9, .89, .88, .01, .01])
        values[1][0, :3, 0] = 10.
        store(memory, values, second_meta)
        self.assertEqual(int(memory.cached_slots_valid.sum()), 1)

    def test_bad_quality_features_and_decoded_size_never_reach_nms(self):
        memory = make_bank(cache_score_mode='quality', cache_nms_iou_threshold=.5,
                           cache_require_current=True)
        memory.get(1, metadata())
        values = proposals(memory, [.99]*5, [.99]*5)
        values[1][0, 0, 3] = 1000.  # Finite log-size, non-finite metric width.
        values[3][0, 1, 0] = float('nan')
        values[0][0, 2, 0] = float('nan')
        values[1][0, 3, 0] = float('nan')
        store(memory, values)
        selected = memory.cached_feature[0, memory.cached_slots_valid[0], 0]
        self.assertEqual(selected.tolist(), [4.])

    def test_ego_only_motion_rotates_heading_without_extrapolation_or_cache_mutation(self):
        memory = make_bank(motion_mode='ego_only')
        memory.get(1, metadata())
        values = proposals(memory, [.99, .98, .97, .01, .01])
        values[1][0, 0, :3] = torch.tensor([10., 0., 1.])
        values[1][0, 0, 8:11] = torch.tensor([2., 0., 99.])
        store(memory, values)
        saved = memory.cached_anchor.clone()
        pose = torch.eye(4)
        pose[:2, :2] = torch.tensor([[0., -1.], [1., 0.]])
        pose[0, 3] = 1.
        _, anchors, _, _, _ = memory.get(1, metadata(
            'f1', 'f0', timestamp=1_789_000_000.5, pose=pose))
        torch.testing.assert_close(anchors[0, 0, :3], torch.tensor([0., -9., 1.]))
        torch.testing.assert_close(anchors[0, 0, 6:8], torch.tensor([-1., 0.]))
        torch.testing.assert_close(anchors[0, 0, 8:11], torch.tensor([0., -2., 0.]))
        torch.testing.assert_close(memory.cached_anchor, saved)


if __name__ == '__main__':
    unittest.main()
