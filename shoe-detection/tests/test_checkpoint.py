import tempfile
import unittest
from pathlib import Path
from pipeline.checkpoint import BestOnlyCheckpointMixin


class FakeFrameworkTrainer:
    def save_model(self):
        data = str(self.epoch).encode()
        self.last.write_bytes(data)
        if self.fitness == self.best_fitness:
            self.best.write_bytes(data)
        if self.save_period > 0:
            (self.best.parent / f'epoch{self.epoch}.pt').write_bytes(data)
        return True


class FakeBestTrainer(BestOnlyCheckpointMixin, FakeFrameworkTrainer):
    pass


class BestCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trainer = FakeBestTrainer()
        self.trainer.best = self.root/'best.pt'
        self.trainer.last = self.root/'last.pt'
        self.trainer.save_period = 1  # periodic saving must still be suppressed

    def epoch(self, epoch, score):
        t = self.trainer
        t.epoch, t.fitness = epoch, score
        t.best_fitness = max(getattr(t, 'best_fitness', float('-inf')), score)
        return t.save_model()

    def test_only_best_and_strict_improvement(self):
        for epoch, score in enumerate([.4, .3, .7, .7, .6]):
            self.epoch(epoch, score)
            self.assertEqual([p.name for p in self.root.iterdir()], ['best.pt'])
        self.assertEqual(self.trainer.best.read_bytes(), b'2')
        self.assertEqual(self.trainer._saved_best_epoch, 3)
        self.assertEqual(self.trainer._saved_best_score, .7)
        self.assertEqual(self.trainer.last, self.root/'last.pt')
        self.assertEqual(self.trainer.save_period, 1)

    def test_invalid_score_does_not_overwrite_best(self):
        self.epoch(0, .5)
        for score in [float('nan'), float('inf'), float('-inf')]:
            self.assertFalse(self.epoch(1, score))
        self.assertEqual(self.trainer.best.read_bytes(), b'0')

    def test_first_zero_score_is_saved(self):
        self.assertTrue(self.epoch(0, 0))
        self.assertTrue(self.trainer.best.is_file())


if __name__ == '__main__':
    unittest.main()
