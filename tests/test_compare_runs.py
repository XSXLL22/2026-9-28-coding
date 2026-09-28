import copy
import tempfile
import unittest
from pathlib import Path
from tools.compare_vision_runs import compare


class CompareRunsTests(unittest.TestCase):
    def test_cross_size_requires_opt_in_and_common_bins_and_nms(self):
        with tempfile.TemporaryDirectory() as directory:
            old = self.make_review(directory, 'FN')
            new = copy.deepcopy(old)
            new['input_size'] = 320
            with self.assertRaisesRegex(ValueError, 'input_size'):
                compare(old, new)
            with self.assertRaisesRegex(ValueError, 'size_reference'):
                compare(old, new, allow_cross_size=True)
            new['size_reference'] = 128
            with self.assertRaisesRegex(ValueError, 'nms_iou'):
                compare(old, new, allow_cross_size=True)
            old['nms_iou'] = new['nms_iou'] = .45
            result = compare(old, new, allow_cross_size=True)
            self.assertEqual(result['size_reference'], 128)
            self.assertEqual(result['new_input_size'], 320)
            new['nms_iou'] = .7
            with self.assertRaisesRegex(ValueError, 'nms_iou'):
                compare(old, new, allow_cross_size=True)

    def test_cross_size_still_rejects_changed_geometry_or_bucket(self):
        with tempfile.TemporaryDirectory() as directory:
            old = self.make_review(directory, 'FN')
            old.update(nms_iou=.45, size_reference=128)
            old['records'][0]['ground_truth'][0]['size_bucket'] = 'lt8'
            new = copy.deepcopy(old)
            new['input_size'] = 320
            new['records'][0]['ground_truth'][0]['size_bucket'] = '8to16'
            with self.assertRaisesRegex(ValueError, 'size bucket'):
                compare(old, new, allow_cross_size=True)
            new['records'][0]['ground_truth'][0] = dict(old['records'][0]['ground_truth'][0], xyxy=[2,3,4,5])
            with self.assertRaisesRegex(ValueError, 'Ground truth mismatch'):
                compare(old, new, allow_cross_size=True)

    def make_review(self, directory, status):
        image, label = Path(directory)/'sample.jpg', Path(directory)/'sample.txt'
        image.write_bytes(b'input provenance test'); label.write_text('label')
        return {'status':'complete', 'input_size':128,'confidence':0.25,'match_iou':0.5,
                'model_sha256':status,'totals':{},'size_counts':{},'records':[
                    {'source':str(image),'label':str(label),'fp':0,'fn':int(status=='FN'),'predictions':[],
                     'ground_truth':[{'class':'fire','index_in_class':0,'xyxy':[1,2,3,4],'status':status}]}]}

    def test_paired_target_gain_is_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            result=compare(self.make_review(directory,'FN'),self.make_review(directory,'TP'))
        self.assertEqual(result['recovered_targets'],1)
        self.assertEqual(result['lost_targets'],0)

    def test_threshold_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            old=self.make_review(directory,'FN'); new=copy.deepcopy(old); new['confidence']=0.1
            with self.assertRaisesRegex(ValueError,'setting mismatch'):
                compare(old,new)

    def test_changed_target_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            old=self.make_review(directory,'FN'); new=copy.deepcopy(old)
            new['records'][0]['ground_truth'][0]['xyxy']=[2,3,4,5]
            with self.assertRaisesRegex(ValueError,'Ground truth mismatch'):
                compare(old,new)
