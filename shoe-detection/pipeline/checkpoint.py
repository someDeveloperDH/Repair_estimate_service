"""Keep only the strictly best validation checkpoint during training."""
import math


class BestOnlyCheckpointMixin:
    """Use the parent serializer, redirecting its last checkpoint to best.pt.

    No last.pt or periodic epoch checkpoint is created. This mixin must precede
    the framework trainer in the inheritance list.
    """

    def save_model(self):
        score = float(self.fitness) if self.fitness is not None else float('nan')
        previous = getattr(self, '_saved_best_score', -math.inf)
        if not math.isfinite(score) or score <= previous:
            return False
        original_last, original_period = self.last, self.save_period
        try:
            self.last = self.best
            self.save_period = -1
            result = super().save_model()
        finally:
            self.last, self.save_period = original_last, original_period
        if result is False or not self.best.is_file():
            raise RuntimeError('Best checkpoint was not saved.')
        self._saved_best_score = score
        self._saved_best_epoch = int(self.epoch) + 1
        return True
