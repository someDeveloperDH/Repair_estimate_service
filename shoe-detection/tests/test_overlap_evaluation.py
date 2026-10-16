import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.metrics import match,evaluate,damage_presence,any_overlap_ap
from pipeline.rf_backend import evaluate_task


class OverlapEvaluationTests(unittest.TestCase):
    def test_tiny_positive_overlap_counts_but_edge_touch_does_not(self):
        gt=[{'class':0,'box':[0,0,10,10]}]
        tiny=[{'class':0,'box':[9,9,20,20],'confidence':.9}]
        touching=[{'class':0,'box':[10,0,20,10],'confidence':.9}]
        self.assertEqual(len(match(gt,tiny,.25,rule='any_overlap')[0]),1)
        self.assertEqual(len(match(gt,tiny,.25,rule='iou50')[0]),0)
        self.assertEqual(len(match(gt,touching,.25,rule='any_overlap')[0]),0)

    def test_wrong_class_and_duplicate_still_count_as_errors(self):
        gt=[{'class':0,'box':[0,0,10,10]}]
        predictions=[dict(gt[0],confidence=.9),dict(gt[0],confidence=.8),
                     {'class':1,'box':[0,0,10,10],'confidence':.7}]
        pairs,fp,fn=match(gt,predictions,.25,rule='any_overlap')
        self.assertEqual((len(pairs),len(fp),len(fn)),(1,2,0))

    def test_one_large_prediction_cannot_claim_two_ground_truths(self):
        gt=[{'class':0,'box':[0,0,10,10]},{'class':0,'box':[20,20,30,30]}]
        predictions=[{'class':0,'box':[0,0,30,30],'confidence':.9}]
        pairs,fp,fn=match(gt,predictions,.25,rule='any_overlap')
        self.assertEqual((len(pairs),len(fp),len(fn)),(1,0,1))

    def test_both_damage_types_are_scored_independently(self):
        gt=[{'class':0,'box':[0,0,10,10]},{'class':1,'box':[20,20,30,30]}]
        records=[dict(file_name='both.jpg',width=100,height=100,ground_truth=gt,
                      predictions=[dict(g,confidence=.8) for g in gt])]
        m=evaluate(records,['outsole','heel lining'],.25,rule='any_overlap')
        self.assertEqual([c['tp'] for c in m['classes']],[1,1])

    def test_all_four_presence_states_and_exception_class(self):
        outsole={'class':0,'confidence':.9,'box':[0,0,10,10]}
        lining={'class':1,'confidence':.9,'box':[20,20,30,30]}
        for predictions,state in [([outsole,lining],'both'),([outsole],'outsole_only'),
                                  ([lining],'heel_lining_only'),([],'no_damage_detected'),
                                  ([dict(outsole,confidence=.1)],'no_damage_detected'),
                                  ([dict(outsole,**{'class':2})],'no_damage_detected')]:
            self.assertEqual(damage_presence(predictions,.25)['state'],state)

    def test_custom_ap_uses_ranked_matches_and_absent_class_na(self):
        records=[dict(file_name='a',ground_truth=[{'class':0,'box':[0,0,10,10]}],
                      predictions=[{'class':0,'confidence':.95,'box':[30,30,40,40]},
                                   {'class':0,'confidence':.9,'box':[9,9,20,20]},
                                   {'class':0,'confidence':.8,'box':[0,0,10,10]}])]
        result=any_overlap_ap(records,['outsole','heel lining'])
        self.assertAlmostEqual(result[0]['ap_any_overlap'],.5)
        self.assertIsNone(result[1]['ap_any_overlap'])

    def test_backend_exports_overlap_score_reference_and_presence(self):
        records=[dict(file_name='a.jpg',width=100,height=100,
                      ground_truth=[{'class':0,'box':[0,0,10,10]},{'class':1,'box':[20,20,30,30]}],
                      predictions=[{'class':0,'box':[9,9,20,20],'confidence':.9},
                                   {'class':1,'box':[29,29,40,40],'confidence':.8}])]
        args=SimpleNamespace(task='data2',split='val',select_threshold=False,threshold_config=None,
                             conf=.25,match_rule='any_overlap',max_det=300,normal_source=None,benchmark=False)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch('pipeline.rf_backend.setup'),patch('pipeline.rf_backend.dataset',return_value=(out,['outsole','heel lining','not_sneakers'],{})),patch('pipeline.rf_backend.images_in',return_value=[]),patch('pipeline.rf_backend.model_for'),patch('pipeline.rf_backend.predict',return_value=records):
                evaluate_task(args,out)
            saved=json.loads((out/'metrics.json').read_text())
            self.assertEqual([c['tp'] for c in saved['fixed_threshold']['classes']],[1,1,0])
            self.assertEqual([c['tp'] for c in saved['reference_iou50']['classes']],[0,0,0])
            self.assertEqual(saved['primary_overlap_ap']['map_any_overlap'],1)
            self.assertEqual(json.loads((out/'records.json').read_text())['records'][0]['damage_presence']['state'],'both')


if __name__=='__main__':unittest.main()
