import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from training.baseline import main, parse_args, augmentation_observer


class TrainingOptionsTests(unittest.TestCase):
    def test_invalid_close_mosaic_rejected_and_boundary_accepted(self):
        base = ['train', '--data', 'missing.yaml', '--weights', 'missing.pt', '--output', 'unused', '--epochs', '50']
        for value in ('-1', '51', '1.5'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + ['--close-mosaic', value])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['evaluate'] + base[1:] + ['--close-mosaic', '0'])
        self.assertEqual(parse_args(base + ['--close-mosaic', '50']).close_mosaic, 50)

    def test_observer_records_changed_graph_once_per_epoch_without_running_transform(self):
        class Mosaic:
            def __init__(self, p): self.p = p
            def __call__(self, *_): raise AssertionError('Observer must not run augmentation')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'epochs.jsonl'
            transform = Mosaic(1.0)
            dataset = SimpleNamespace(transforms=SimpleNamespace(transforms=[transform]))
            trainer = SimpleNamespace(epoch=39, args=SimpleNamespace(close_mosaic=10), train_loader=SimpleNamespace(dataset=dataset))
            callback = augmentation_observer(path)
            callback(trainer)
            callback(trainer)
            dataset.transforms = SimpleNamespace(transforms=[Mosaic(0.0)])
            trainer.epoch = 40
            callback(trainer)
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['epoch'] for r in rows], [40, 41])
            self.assertEqual([r['transforms'][0]['p'] for r in rows], [1.0, 0.0])

    def test_invalid_rate_or_evaluation_override_rejected_before_running(self):
        base = ['train', '--data', 'missing.yaml', '--weights', 'missing.pt', '--output', 'unused']
        for value in ('nan', 'inf', '-0.1', '1.01'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): parse_args(base + ['--warmup-bias-lr', value])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['evaluate'] + base[1:] + ['--warmup-bias-lr', '0'])

    def test_override_reaches_real_training_call_and_report_default_stays_omitted(self):
        # Exercise the entry point, mock only expensive training/evaluation and data access.
        for value, close, size in ((None, None, 128), ('0', None, 128), ('0.1', None, 128), ('0.1', '10', 128), ('0.1', '0', 320)):
            with self.subTest(value=value, close=close), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                weights, best, output = root/'initial.pt', root/'best.pt', root/'output'
                weights.write_bytes(b'fixture-initial')
                best.write_bytes(b'fixture-best')
                model = MagicMock()
                model.names = {0: 'smoke', 1: 'fire'}
                model.trainer.best = best
                model.val.return_value = SimpleNamespace(box=SimpleNamespace(ap_class_index=[]), results_dict={}, speed={})
                argv = ['train', '--data', str(root/'data.yaml'), '--weights', str(weights),
                        '--output', str(output), '--epochs', '50', '--eval-sizes', str(size)]
                if size != 128: argv += ['--imgsz', str(size)]
                if value is not None: argv += ['--warmup-bias-lr', value]
                if close is not None: argv += ['--close-mosaic', close]
                with patch('training.baseline.configure_ultralytics'), patch('training.baseline.environment', return_value={}), \
                     patch('training.baseline.audit_dataset', return_value={'names': model.names}), \
                     patch('ultralytics.YOLO', return_value=model), contextlib.redirect_stdout(io.StringIO()):
                    main(argv)
                options = model.train.call_args.kwargs
                if value is None: self.assertNotIn('warmup_bias_lr', options)
                else: self.assertEqual(options['warmup_bias_lr'], float(value))
                report = json.loads((output/'report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['training_options'], options)
                self.assertEqual(options['epochs'], 50)
                self.assertEqual(options['imgsz'], size)
                self.assertEqual(model.val.call_args.kwargs['imgsz'], size)
                self.assertEqual(options['optimizer'], 'AdamW')
                self.assertEqual(options['lr0'], .001)
                self.assertEqual(options['close_mosaic'], int(close) if close is not None else 0)
                self.assertEqual(report['status'], 'complete')
