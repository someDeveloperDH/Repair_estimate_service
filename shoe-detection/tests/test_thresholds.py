import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from pipeline.run import fixed_conf, select_threshold


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.weights = self.out/'best.pt'
        self.weights.write_bytes(b'fixture-fingerprint-not-a-real-model')
        self.args = SimpleNamespace(task='data2', weights=self.weights, imgsz=512,
                                    nms_iou=.7, max_det=300, conf=None)

    def records(self, missing_second=False):
        g = [{'class':0, 'box':[0,0,10,10]}, {'class':1, 'box':[20,20,30,30]}]
        predictions = [dict(g[0], confidence=.8)]
        if not missing_second:
            predictions.append(dict(g[1], confidence=.6))
        return [dict(file_name='sample.jpg', width=100, height=100,
                     ground_truth=g, predictions=predictions)]

    def test_data2_both_recalls_constrain_threshold(self):
        config = select_threshold(self.args, self.records(), ['outsole','heel lining','not_sneakers'], self.out)
        self.assertTrue(config['constraints_met'])
        self.assertLessEqual(config['confidence'], .6)
        self.assertEqual(fixed_conf(self.args, config), config['confidence'])

    def test_infeasible_is_explicit(self):
        config = select_threshold(self.args, self.records(True), ['outsole','heel lining','not_sneakers'], self.out)
        self.assertFalse(config['constraints_met'])

    def test_reject_different_weights(self):
        config = select_threshold(self.args, self.records(), ['outsole','heel lining','not_sneakers'], self.out)
        self.weights.write_bytes(b'changed-checkpoint')
        with self.assertRaises(ValueError):
            fixed_conf(self.args, config)

    def test_reject_different_prediction_settings(self):
        config = select_threshold(self.args, self.records(), ['outsole','heel lining','not_sneakers'], self.out)
        self.args.imgsz = 640
        with self.assertRaises(ValueError):
            fixed_conf(self.args, config)

    def test_reject_different_matching_rule(self):
        config=select_threshold(self.args,self.records(),['outsole','heel lining','not_sneakers'],self.out)
        self.args.match_rule='any_overlap'
        with self.assertRaises(ValueError):fixed_conf(self.args,config)


if __name__ == '__main__':
    unittest.main()
