import unittest
from pipeline.inference_trace import absolute_boxes, find_query, merge_with_lineage, strongest_spatial_change, target_plan
from run_improvement_experiments import merge_predictions


class InferenceTraceTests(unittest.TestCase):
    def test_query_link_uses_box_and_score(self):
        boxes=[[10,10,30,30],[10,10,30,30],[50,50,60,60]]
        self.assertEqual(find_query(boxes,[.1,.8,.9],[10,10,30,30],.8),1)

    def test_query_link_refuses_inexact_explanation(self):
        with self.assertRaises(ValueError):
            find_query([[10,10,30,30]],[.8],[50,50,60,60],.8)

    def test_box_conversion_clamps_source_coordinates(self):
        self.assertEqual(absolute_boxes([[.5,.5,2,2]],512,320),[[0,0,512,320]])

    def test_nms_lineage_matches_real_experiment_merge(self):
        rows=[dict(id='full/q1/c0',**{'class':0},confidence=.9,box=[10,10,30,30]),
              dict(id='tile1/q2/c0',**{'class':0},confidence=.8,box=[11,11,31,31]),
              dict(id='tile2/q2/c1',**{'class':1},confidence=.7,box=[10,10,30,30])]
        kept,decisions=merge_with_lineage(rows)
        self.assertEqual(kept,merge_predictions(rows))
        self.assertEqual(decisions[1]['action'],'suppressed')
        self.assertEqual(decisions[1]['by'],'full/q1/c0')
        self.assertEqual(decisions[2]['action'],'kept')

    def test_max_det_is_recorded_as_exclusion(self):
        rows=[dict(id=str(i),**{'class':0},confidence=.9-i*.1,box=[i*30,0,i*30+10,10]) for i in range(3)]
        kept,decisions=merge_with_lineage(rows,max_det=2)
        self.assertEqual(len(kept),2)
        self.assertEqual(decisions[-1]['action'],'max_det_excluded')

    def test_explanation_does_not_confuse_query_switch_with_detection_loss(self):
        probes=[dict(id='upper_context',score_drop=.76,spatial_score_drop=-.01),
                dict(id='toe_blur',score_drop=.75,spatial_score_drop=.74)]
        self.assertEqual(strongest_spatial_change(probes)['id'],'toe_blur')

    def test_multiple_classes_and_multiple_same_class_boxes_get_separate_traces(self):
        rows=[dict(**{'class':0},confidence=.9,box=[0,0,10,10]),
              dict(**{'class':1},confidence=.8,box=[50,50,60,60]),
              dict(**{'class':0},confidence=.7,box=[100,100,110,110]),
              dict(**{'class':1},confidence=.1,box=[150,150,160,160])]
        plan=target_plan(rows,.25)
        self.assertEqual([(p['class_id'],p['class_rank']) for p in plan],[(0,0),(1,0),(0,1)])
        self.assertEqual(len(target_plan(rows,.95)),0)


if __name__=='__main__':unittest.main()
