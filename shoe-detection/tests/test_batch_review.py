import tempfile
import unittest
from pathlib import Path

from pipeline.batch_review import render_report,draw_images,write_index


class BatchReviewTests(unittest.TestCase):
    def test_report_escapes_names_and_links_to_actual_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/'model/images/photo';folder.mkdir(parents=True)
            (root/'index.html').write_text('index')
            render_report(folder,'photo <test>',['## 안감 예측','검출 2개 <사실>'])
            text=(folder/'report.html').read_text(encoding='utf-8')
            self.assertIn('photo &lt;test&gt;',text)
            self.assertIn('검출 2개 &lt;사실&gt;',text)
            self.assertTrue((folder/'../../../index.html').resolve().exists())

    def test_no_detection_and_multiple_ground_truths_save_separately(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            gt=[{'class':0,'box':[5,5,20,20]},{'class':1,'box':[40,40,60,60]}]
            draw_images(Image.new('RGB',(100,100)),[],gt,['outsole','heel lining'],folder)
            for name in ['predictions.png','ground_truth.png','overlay.png','comparison.jpg']:
                self.assertTrue((folder/name).is_file())
            with Image.open(folder/'comparison.jpg') as im:self.assertEqual(im.size,(300,170))

    def test_index_links_to_each_image_without_class_collapsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            status=dict(split='val',status='running',completed_images=1,planned_images=2,confidence=.25)
            reports={'model':[dict(file_name='both.jpg',report='images/both/report.html',predictions=3,tp=2,fp=1,fn=0,
                                  presence={'state':'both'})]}
            write_index(root,status,reports)
            text=(root/'index.html').read_text(encoding='utf-8')
            self.assertIn('model/images/both/report.html',text)
            self.assertIn('밑창·안감 모두 검출',text)


if __name__=='__main__':unittest.main()
