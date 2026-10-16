import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from run_improvement_experiments import windows, crop_boxes, merge_predictions, prepare, score


class ImprovementTransformsTests(unittest.TestCase):
    def test_windows_cover_frame_and_preserve_coordinates(self):
        views=windows(512,512)
        self.assertEqual(views,[(0,0,320,320),(192,0,512,320),(0,192,320,512),(192,192,512,512)])
        boxes=[{'class':0,'box':[220,230,250,260]}]
        self.assertEqual(crop_boxes(boxes,views[3]),[{'class':0,'box':[28,38,58,68]}])

    def test_partial_boxes_clipped_and_low_visibility_removed(self):
        boxes=[{'class':0,'box':[300,100,330,130]}, {'class':1,'box':[315,150,350,180]}]
        self.assertEqual(crop_boxes(boxes,(0,0,320,320)),[{'class':0,'box':[300,100,320,130]}])

    def test_merging_removes_duplicate_views_but_keeps_distinct_classes(self):
        predictions=[{'class':0,'box':[10,10,30,30],'confidence':.9},
                     {'class':0,'box':[11,11,31,31],'confidence':.8},
                     {'class':1,'box':[10,10,30,30],'confidence':.7}]
        self.assertEqual(merge_predictions(predictions),[predictions[0],predictions[2]])

    def test_preparation_repeats_only_train_and_preserves_source(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'datasets/data2'
            for split in ('train','val'):
                (source/'images'/split).mkdir(parents=True)
                (source/'labels'/split).mkdir(parents=True)
                Image.new('RGB',(512,512)).save(source/'images'/split/f'{split}.jpg')
                (source/'labels'/split/f'{split}.txt').write_text('0 0.5 0.5 0.05 0.05\n')
            (source/'data.yaml').write_text('names: [outsole, heel lining, not_sneakers]\n')
            with patch('run_improvement_experiments.ROOT',root):
                prepare(root/'prepared',balanced=True)
            saved=json.loads((root/'prepared/preparation.json').read_text())
            self.assertEqual(saved['train_images'],15)
            self.assertEqual(saved['val_images'],1)
            self.assertFalse((root/'prepared/images/test').exists())
            self.assertEqual((source/'labels/train/train.txt').read_text(),'0 0.5 0.5 0.05 0.05\n')

    def test_scoring_saves_threshold_candidate_and_flags_unmet_target(self):
        records=[dict(file_name='sample.jpg',width=100,height=100,
                      ground_truth=[{'class':0,'box':[0,0,10,10]}, {'class':0,'box':[50,50,60,60]}],
                      predictions=[{'class':0,'box':[0,0,10,10],'confidence':.8}])]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'score'
            score(records,['outsole'],out)
            candidate=json.loads((out/'class_threshold_candidates.json').read_text())[0]
            self.assertEqual(candidate['name'],'outsole')
            self.assertFalse(candidate['target_met'])
            self.assertEqual(candidate['recall'],.5)


if __name__=='__main__': unittest.main()
