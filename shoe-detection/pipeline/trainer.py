"""Current Ultralytics backend with a single-checkpoint storage policy."""
from ultralytics.models.yolo.detect import DetectionTrainer
from .checkpoint import BestOnlyCheckpointMixin


class BestOnlyDetectionTrainer(BestOnlyCheckpointMixin, DetectionTrainer):
    pass
