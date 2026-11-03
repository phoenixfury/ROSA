"""Causal, deliberately weak XY-motion targets from pseudo-labelled tracks.

Only NumPy is required. Thresholds are uncertainty heuristics, not physical
ground truth. Unknown targets stay NaN; global-static classification never
uses displacement in the ego/LiDAR frame. No Z velocity is estimated.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionTargetConfig:
    window: int = 5
    min_observations: int = 3
    max_gap_seconds: float = 1.5
    max_speed_mps: float = 45.
    min_span_seconds: float = .8
    huber_delta_m: float = .35
    max_residual_m: float = .75
    max_endpoint_residual_m: float = 1.25
    static_speed_mps: float = .3
    static_max_spread_m: float = .3
    iterations: int = 8

    def __post_init__(self):
        if not 3 <= self.min_observations <= self.window or self.iterations < 1:
            raise ValueError('Motion fitting requires window >= min_observations >= 3')
        positive = (self.max_gap_seconds, self.max_speed_mps,
                    self.min_span_seconds, self.huber_delta_m,
                    self.max_residual_m, self.max_endpoint_residual_m)
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError('Motion fit limits must be positive and finite')
        if self.static_speed_mps < 0 or self.static_max_spread_m < 0:
            raise ValueError('Static-motion limits must be nonnegative')


def fit_world_xy(timestamp_ns, positions, config=None):
    """Fit the supplied past/current observations, never future observations.

    Returns a diagnostic dictionary including a world XY velocity, fitted
    current XY, and validity/status. The caller must enforce track identity
    and chronological causality. Rejected fits contain NaN velocity.
    """
    config = config or MotionTargetConfig()
    times = np.asarray(timestamp_ns, dtype=np.int64)[-config.window:]
    points = np.asarray(positions, dtype=np.float64)[-config.window:, :2]
    result = dict(status='insufficient_history', support=len(times),
                  span_seconds=0., residual_m=None, endpoint_residual_m=None,
                  speed_mps=None, is_static=False,
                  velocity_world=np.full(2, np.nan),
                  fitted_world_xy=np.full(2, np.nan))
    if len(times) != len(points) or not np.isfinite(points).all():
        result['status'] = 'invalid_observation'
        return result
    if len(times) < config.min_observations:
        return result
    gaps = np.diff(times).astype(np.float64)*1e-9
    if (gaps <= 0).any() or (gaps > config.max_gap_seconds).any():
        result['status'] = 'time_discontinuity'
        return result
    seconds = (times - times[-1]).astype(np.float64)*1e-9
    result['span_seconds'] = float(-seconds[0])
    if result['span_seconds'] < config.min_span_seconds:
        result['status'] = 'short_span'
        return result
    # Center coordinates before least squares to avoid ill-conditioning at
    # large map offsets. Vector-Huber weights preserve the XY direction.
    origin = points[-1]
    centered = points-origin
    design = np.column_stack([np.ones(len(times)), seconds])
    weights = np.ones(len(times))
    coefficients = None
    for _ in range(config.iterations):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(design*root[:, None], centered*root[:, None], rcond=None)[0]
        residual = np.linalg.norm(design@coefficients-centered, axis=1)
        weights = np.minimum(1., config.huber_delta_m/np.maximum(residual, 1e-12))
    residual = np.linalg.norm(design@coefficients-centered, axis=1)
    # Weighted RMS audits the supported fit; endpoint gate separately prevents
    # an ignored current-frame outlier becoming a confident target.
    rms = float(np.sqrt(np.average(residual**2, weights=weights)))
    endpoint = float(residual[-1])
    velocity = coefficients[1]
    speed = float(np.linalg.norm(velocity))
    fitted = origin+coefficients[0]
    result.update(residual_m=rms, endpoint_residual_m=endpoint,
                  speed_mps=speed, fitted_world_xy=fitted)
    if speed > config.max_speed_mps:
        result['status'] = 'excessive_speed'
    elif endpoint > config.max_endpoint_residual_m:
        result['status'] = 'endpoint_outlier'
    elif rms > config.max_residual_m:
        result['status'] = 'uncertain_fit'
    else:
        # Sustained support and low *global* speed plus a compact trajectory
        # are all required. A stopped ego does not imply a static object.
        spread = float(np.max(np.linalg.norm(points-np.median(points, axis=0), axis=1)))
        static = speed <= config.static_speed_mps and spread <= config.static_max_spread_m
        result.update(status='static' if static else 'moving',
                      is_static=static,
                      velocity_world=np.zeros(2) if static else velocity)
    return result


def robust_track_velocities(samples, config=None, return_diagnostics=False):
    """Return token -> N x 2 current-LiDAR velocities from causal world fits.

    Input schema matches the chronological ROSA PKLs. History is isolated by
    (scene_name, gt_id); a gap clears history. Diagnostics optionally return a
    second token -> list[dict] mapping aligned with the untouched annotations.
    """
    config = config or MotionTargetConfig()
    history = defaultdict(lambda: deque(maxlen=config.window))
    velocities, diagnostics = {}, {}
    for sample in sorted(samples, key=lambda row: int(row['metainfo']['acquisition_timestamp_ns'])):
        meta = sample['metainfo']
        scene = str(meta['scene_name'])
        timestamp = int(meta['acquisition_timestamp_ns'])
        pose = np.asarray(meta['T_world_CLIDAR'], dtype=np.float64)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])):
            raise ValueError('Motion targets require valid T_world_CLIDAR')
        boxes = np.asarray(sample['annos']['gt_boxes'], dtype=np.float64).reshape(-1, 7)
        ids = np.asarray(sample['annos']['gt_ids'], dtype=np.int64)
        if len(ids) != len(boxes) or len(set(ids.tolist())) != len(ids):
            raise ValueError('Motion targets require aligned unique per-frame track IDs')
        values = np.full((len(boxes), 2), np.nan, dtype=np.float32)
        records = []
        for index, (box, track_id) in enumerate(zip(boxes, ids)):
            world = pose[:3, :3]@box[:3]+pose[:3, 3]
            past = history[(scene, int(track_id))]
            dt = (timestamp-past[-1][0])*1e-9 if past else None
            if dt is not None and (dt <= 0 or dt > config.max_gap_seconds):
                past.clear()
                dt = None
            if not np.isfinite(world).all():
                record = dict(status='invalid_observation', support=len(past),
                              velocity_world=np.full(2, np.nan), fitted_world_xy=np.full(2, np.nan))
            elif dt is not None and np.linalg.norm(world[:2]-past[-1][1][:2])/dt > config.max_speed_mps:
                # Do not contaminate subsequent fits with a rejected gross
                # ID/position jump; later plausible observations may recover.
                record = dict(status='jump_speed', support=len(past),
                              velocity_world=np.full(2, np.nan), fitted_world_xy=np.full(2, np.nan))
            else:
                past.append((timestamp, world))
                record = fit_world_xy([row[0] for row in past], [row[1] for row in past], config)
            if np.isfinite(record['velocity_world']).all():
                world_velocity = np.r_[record['velocity_world'], 0.]
                values[index] = (pose[:3, :3].T@world_velocity)[:2]
            records.append(record)
        token = str(sample['token'])
        if token in velocities:
            raise ValueError('Duplicate frame tokens are not allowed')
        velocities[token], diagnostics[token] = values, records
    return (velocities, diagnostics) if return_diagnostics else velocities
