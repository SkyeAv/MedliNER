"""MEDliNER: reviewed medical NER data and GLiNER fine-tuning."""

from .schema import ALLOWED_LABELS, ALLOWED_TASKS, Annotation, Example, SourceMetadata

__version__ = "0.1.0"

__all__ = ["ALLOWED_LABELS", "ALLOWED_TASKS", "Annotation", "Example", "SourceMetadata", "__version__"]
