"""Clearly labeled synthetic fixtures for a simulation-only fusion prototype."""
from __future__ import annotations

from runtime.fusion import digest


def common(config, stream, stamp, available=None):
    return {'schema_version': 1, 'origin': 'simulation', 'node_id': config['node_id'],
            'session_id': config['session_id'], 'clock_id': config['clock_id'],
            'stream_id': stream, 'record_id': f'{stream}-{stamp}', 'sample_time_ms': stamp,
            'available_time_ms': stamp if available is None else available,
            'valid': True, 'invalid_reason': None, 'health': 'OK'}


def vision(config, stamp, fire=0.0, smoke=0.0, available=None):
    record = common(config, 'vision', stamp, available)
    detections = [{'project_class_id': c, 'confidence': score, 'xyxy_original_pixels': [10, 10, 40, 40],
                   'roi_box_area_fraction': 0.5} for c, score in ((0, smoke), (1, fire)) if score > 0]
    record.update(kind='vision', frame_id=stamp, model_sha256=digest('synthetic model identity, not trained weights'),
                  run_sha256=digest('synthetic fixture'), vision_config_sha256=digest('synthetic visual config'),
                  roi_status='simulation_only', supported_classes=[0, 1], image_width=100, image_height=100,
                  confidence_floor=0.0, detections=detections)
    return record


def sensor(config, stamp, temperature=25.0, available=None):
    record = common(config, 'ambient_temperature', stamp, available)
    record.update(kind='sensor', sensor_id='simulated-ambient-01', sensor_type='ambient_temperature',
                  unit='degC', calibration_status='simulation_only', value=temperature)
    return record


def scenarios(config):
    normal = [record for t in range(0, 3500, 500) for record in (vision(config, t), sensor(config, t))]
    strong = [vision(config, t, fire=.95) for t in (0, 500, 1000)]
    combo = [record for t in (0, 500, 1000) for record in (vision(config, t, smoke=.7), sensor(config, t, 65))]
    recovery = strong + [record for t in range(1500, 5000, 500) for record in (vision(config, t), sensor(config, t))]
    future = [vision(config, 1000, fire=.95, available=3000)]
    return {
        'normal_then_stale': (normal, list(range(0, 6500, 500))),
        'strong_single_source': (strong, [0, 500, 1000, 3000, 5000]),
        'isolated_fire_frame': ([vision(config, 0, fire=.95)], [0, 500, 1000, 2000]),
        'smoke_heat_combination': (combo, [0, 500, 1000]),
        'fresh_recovery': (recovery, list(range(0, 5000, 500))),
        'duplicate_cannot_accumulate': ([vision(config, 0, fire=.95)] * 4, [0, 500, 1000, 2000]),
        'late_future_record': (future, [0, 1000, 2000, 3000]),
        'late_older_sample': ([vision(config, 1000), vision(config, 0, fire=.95, available=1500)], [1000, 1500, 2000]),
    }
