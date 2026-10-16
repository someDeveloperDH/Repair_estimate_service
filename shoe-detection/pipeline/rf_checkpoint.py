"""RF-DETR callback retaining exactly one best checkpoint, no last/archive."""
import math
from rfdetr.training.callbacks.best_model import BestModelCallback


class SingleBestCallback(BestModelCallback):
    def __init__(self,output_dir):
        super().__init__(output_dir=output_dir,monitor_regular='val/mAP_50_95',
                         monitor_ema=None,run_test=False,evaluates_base_model=True)
        self.filename='best'
        self.selected_epoch=None
        self.selected_score=None

    def on_validation_end(self,trainer,pl_module):
        score=trainer.callback_metrics.get(self.monitor)
        if score is None or not math.isfinite(float(score)): return
        if self.selected_score is not None and float(score)<=self.selected_score: return
        super().on_validation_end(trainer,pl_module)

    def _save_checkpoint(self,trainer,filepath):
        super()._save_checkpoint(trainer,filepath)
        self.selected_epoch=int(trainer.current_epoch)+1
        self.selected_score=float(trainer.callback_metrics[self.monitor])

    def on_fit_end(self,trainer,pl_module):
        # Native implementation duplicates best_regular into best_total; suppress it.
        pass
