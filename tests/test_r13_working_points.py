import unittest

from tools.run_r13_working_points import SCHEMES, parse_args


class WorkingPointSweepTests(unittest.TestCase):
    def test_scheme_grid_matches_predeclared_plan(self):
        self.assertEqual([(n, c, i) for n, c, i in SCHEMES],
                         [('S0_conf0.25_nms0.45', 0.25, 0.45),
                          ('S1_conf0.35_nms0.45', 0.35, 0.45),
                          ('S2_conf0.45_nms0.45', 0.45, 0.45),
                          ('S3_conf0.25_nms0.30', 0.25, 0.30),
                          ('S4_conf0.35_nms0.30', 0.35, 0.30)])

    def test_arguments(self):
        args = parse_args(['--weights', 'w.pt', '--imgsz', '320', '--output', 'out'])
        self.assertEqual(args.imgsz, 320)
        self.assertEqual(args.device, 'cpu')
        with self.assertRaises(SystemExit):
            parse_args(['--weights', 'w.pt', '--imgsz', '256', '--output', 'out'])


if __name__ == '__main__':
    unittest.main()
