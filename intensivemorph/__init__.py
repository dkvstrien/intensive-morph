"""IntensiveMorph — high-density multi-target sentence immersion for Anki."""
from .config import IntensiveMorphConfig
from .inference import BayesianInference
from .models import IntensiveMorphDB, Stage, LemmaState
from .morphemizer import Morphemizer
from .scheduler import IntensiveMorphScheduler
from .sentence_pool import SentencePool
from .target_list import TargetList
from .importer import TextImporter
__all__ = [
    "IntensiveMorphConfig", "IntensiveMorphDB", "Stage", "LemmaState",
    "Morphemizer", "BayesianInference", "IntensiveMorphScheduler",
    "SentencePool", "TargetList", "TextImporter",
]
__version__ = "0.2.0"
