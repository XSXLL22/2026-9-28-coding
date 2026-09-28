import unittest
import tempfile
from pathlib import Path
from tools.prepare_expanded_dataset import quarantine, similarity_pairs, image_signature


class ExpandedDatasetTests(unittest.TestCase):
    def test_pixel_identity_ignores_png_metadata(self):
        from PIL import Image, PngImagePlugin
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/'a.png', Path(directory)/'b.png'
            image = Image.new('RGB',(20,15),'red')
            image.save(first)
            meta = PngImagePlugin.PngInfo(); meta.add_text('source','same pixels')
            image.save(second,pnginfo=meta)
            a,b=image_signature(first),image_signature(second)
            self.assertNotEqual(first.read_bytes(),second.read_bytes())
            self.assertEqual(a['decoded_rgb_sha256'],b['decoded_rgb_sha256'])
            self.assertEqual(a['phash63'],b['phash63'])

    def test_exif_rotation_requires_explicit_label_review(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rotated.jpg'
            image=Image.new('RGB',(20,15),'red')
            exif=Image.Exif(); exif[274]=6
            image.save(path,exif=exif)
            with self.assertRaisesRegex(ValueError,'EXIF orientation'):
                image_signature(path)

    def test_transitive_component_cannot_bridge_train_and_validation(self):
        records = [{'split': s, 'image': str(i)} for i,s in enumerate(('train','train','val'))]
        excluded, groups = quarantine(records, [{'a':0,'b':1}, {'a':1,'b':2}])
        self.assertEqual(set(excluded), {0,1})
        self.assertEqual(groups[0]['kept'], [2])

    def test_test_is_retained_and_never_transferred_to_training(self):
        records = [{'split': s, 'image': str(i)} for i,s in enumerate(('val','test','test','train'))]
        excluded, groups = quarantine(records, [{'a':0,'b':1}, {'a':1,'b':2}, {'a':2,'b':3}])
        self.assertEqual(set(excluded), {0,3})
        self.assertEqual(groups[0]['kept'], [1,2])

    def test_hash_boundary_and_pixel_duplicate(self):
        def record(i, h, pixels):
            return {'split':'train', 'phash63':h, 'image_sha256':str(i), 'decoded_rgb_sha256':pixels}
        records = [record(0,'0','a'),record(1,'3f','b'),record(2,'7f','c'),record(3,'ffff','a')]
        pairs = similarity_pairs(records,6)
        self.assertTrue(any(p['a']==0 and p['b']==1 for p in pairs))
        self.assertFalse(any(p['a']==0 and p['b']==2 for p in pairs))
        self.assertTrue(any(p['a']==0 and p['b']==3 and p['reason']=='exact_pixels_or_file' for p in pairs))
