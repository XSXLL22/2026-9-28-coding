import json
import tempfile
import unittest
from pathlib import Path

from runtime.common import file_hash, write_json
from tools.adapt_p2_fusion import adapt


class AdapterTests(unittest.TestCase):
    def fixture(self, root):
        config = json.loads(Path('configs/fusion_simulation.json').read_text(encoding='utf-8'))
        image = root / 'source.bin'
        image.write_bytes(b'fixture source hash, no decoder needed')
        original = root / 'original_config.json'
        vision = {'node_id': config['node_id'], 'roi': {'status': 'example_not_site_calibrated'}}
        original.write_text(json.dumps(vision), encoding='utf-8')
        write_json(root / 'config_snapshot.json', vision)
        run = {'status': 'complete', 'frame_count': 1, 'model_sha256': 'a'*64,
               'config_sha256': file_hash(original), 'class_mapping': {'0': 0, '1': 1},
               'input_hashes': {str(image): file_hash(image)}, 'args': {'conf': .25, 'config': str(original)}}
        write_json(root / 'run.json', run)
        record = {'frame_id': 0, 'model_sha256': 'a'*64, 'source': str(image), 'source_timestamp_ms': None,
                  'timestamp_basis': 'unavailable', 'image_width': 100, 'image_height': 100, 'detections': []}
        (root / 'detections.jsonl').write_text(json.dumps(record)+'\n', encoding='utf-8')
        manifest = root / 'timeline.json'
        write_json(manifest, {'schema_version': 1, 'origin': 'simulation',
                              'frames': [{'frame_id': 0, 'sample_time_ms': 1234, 'available_time_ms': 1334}]})
        return config, manifest

    def test_null_source_time_requires_and_preserves_explicit_simulated_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = self.fixture(root)
            records = adapt(root, manifest, config)
            self.assertEqual(records[0]['sample_time_ms'], 1234)
            self.assertEqual(records[0]['available_time_ms'], 1334)
            self.assertEqual(records[0]['origin'], 'simulation')
            self.assertIsNone(records[0]['provenance']['source_timestamp_ms'])
            self.assertEqual(records[0]['confidence_floor'], .25)

    def test_missing_timeline_frame_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = self.fixture(root)
            write_json(manifest, {'schema_version': 1, 'origin': 'simulation', 'frames': []})
            with self.assertRaisesRegex(ValueError, 'cover each frame'):
                adapt(root, manifest, config)

    def test_changed_source_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = self.fixture(root)
            (root / 'source.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Source image/video changed'):
                adapt(root, manifest, config)
