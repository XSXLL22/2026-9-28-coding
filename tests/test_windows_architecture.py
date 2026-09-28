import os
import platform
import unittest
from unittest.mock import patch

from runtime.common import restore_windows_architecture


@unittest.skipUnless(os.name == 'nt', 'Windows metadata recovery')
class WindowsArchitectureTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(platform, 'invalidate_caches'):
            platform.invalidate_caches()
        else:
            platform._uname_cache = None

    def test_missing_metadata_is_recovered_from_real_os_and_cpu_check_runs(self):
        with patch.dict(os.environ):
            os.environ.pop('PROCESSOR_ARCHITECTURE', None)
            self.tearDown()
            restore_windows_architecture()
            self.assertIn(platform.machine(), ('AMD64', 'ARM64', 'x86'))
            self.assertNotIn('POLARS_SKIP_CPU_CHECK', os.environ)
            # Real import/read exercises the dependency that failed at checkpoint save.
            import polars as pl
            self.assertEqual(pl.read_csv(b'epoch,loss\n1,0.5\n').to_dict(as_series=False),
                             {'epoch': [1], 'loss': [.5]})

    def test_existing_metadata_is_preserved(self):
        with patch.dict(os.environ, {'PROCESSOR_ARCHITECTURE': 'existing-value'}):
            restore_windows_architecture()
            self.assertEqual(os.environ['PROCESSOR_ARCHITECTURE'], 'existing-value')
