import tempfile
import unittest
from pathlib import Path
from pipeline.rf_backend import coco_ap


class RFAPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)
        self.record=dict(file_name='one.jpg',width=100,height=100,
                         ground_truth=[{'class':0,'box':[0,0,10,10]}],predictions=[])

    def test_perfect_and_unsupported_class(self):
        self.record['predictions']=[{'class':0,'box':[0,0,10,10],'confidence':.9}]
        result=coco_ap([self.record],['outsole','absent'],self.out)
        self.assertAlmostEqual(result[0]['ap50'],1)
        self.assertAlmostEqual(result[0]['ap50_95'],1)
        self.assertIsNone(result[1]['ap50'])

    def test_empty_predictions(self):
        result=coco_ap([self.record],['outsole'],self.out)
        self.assertEqual(result[0]['ap50'],0)
        self.assertEqual(result[0]['ap50_95'],0)


if __name__=='__main__': unittest.main()
