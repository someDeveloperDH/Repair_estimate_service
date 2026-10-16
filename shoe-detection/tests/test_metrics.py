"""Run with: python -m unittest discover -s tests -v"""
import unittest
from pipeline.metrics import evaluate, iou, match


def gt(c=0, box=None):
    return {'class': c, 'box': box or [0, 0, 10, 10]}


def pred(c=0, score=.9, box=None):
    return dict(gt(c, box), confidence=score)


class DetectionMetricsTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0,0,10,10], [0,0,10,10]), 1)
        self.assertEqual(iou([0,0,10,10], [20,20,30,30]), 0)
        self.assertAlmostEqual(iou([0,0,10,10], [5,0,15,10]), 1/3)

    def test_duplicate_is_false_positive(self):
        pairs, fps, fns = match([gt()], [pred(), pred(score=.8)], .5)
        self.assertEqual((len(pairs), len(fps), len(fns)), (1,1,0))

    def test_wrong_class_is_fp_and_fn(self):
        pairs, fps, fns = match([gt()], [pred(c=1)], .5)
        self.assertEqual((len(pairs), len(fps), len(fns)), (0,1,1))

    def test_threshold_and_localization(self):
        pairs, fps, fns = match([gt()], [pred(score=.1), pred(box=[20,20,30,30])], .5)
        self.assertEqual((len(pairs), len(fps), len(fns)), (0,1,1))

    def test_empty_class_na_and_empty_predictions(self):
        r = dict(file_name='a.jpg', width=100, height=100, ground_truth=[gt()], predictions=[])
        m = evaluate([r], ['shoe','absent'], .5)
        self.assertEqual(m['classes'][0]['recall'], 0)
        self.assertIsNone(m['classes'][1]['recall'])
        self.assertIsNone(m['classes'][1]['f1'])
        self.assertEqual(m['supported_class_ids'], [0])

    def test_micro_and_size_support(self):
        r = dict(file_name='a.jpg', width=100, height=100,
                 ground_truth=[gt(), gt(box=[50,50,55,55])], predictions=[pred()])
        m = evaluate([r], ['shoe'], .5)
        self.assertEqual(m['micro']['precision'], 1)
        self.assertEqual(m['micro']['recall'], .5)
        self.assertEqual(m['size_recall']['under_1pct']['support'], 1)
        self.assertEqual(m['size_recall']['1_to_10pct']['support'], 1)


if __name__ == '__main__':
    unittest.main()
