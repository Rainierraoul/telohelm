"""Compatibility: the tanker's prediction tool now lives in its task pack (packs/tanker_prediction.py).

Kept so the run, the learner's limits and saved readers that import it from here keep working.
"""
from ..taskpacks.tanker.prediction import GAP_NM; from ..taskpacks.tanker.prediction import MAX_EVALUATIONS; from ..taskpacks.tanker.prediction import STOPPED_TURN_RATE; from ..taskpacks.tanker.prediction import predict  # noqa: F401
