"""Canonical MedliNER contracts.

Label Studio is an input UI. These Pydantic models are the stable, reviewable contract used
by normalization, splitting, evaluation, and packaging.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "medliner.example.v1"
ALLOWED_LABELS = ("disease", "phenotype")
ALLOWED_TASKS = ("indication", "contraindication")
REVIEWED_STATUSES = ("reviewed", "adjudicated")
Provenance = Literal["human", "adjudicated", "model_suggestion"]
# Derived so the runtime tuple and the annotated type cannot drift apart.
PROVENANCE_VALUES = get_args(Provenance)
# Label Studio stamps each submitted region with where it came from. Pre-labeled projects need
# this to stay auditable: "prediction" means a human submitted a model span without touching it,
# which is a weaker signal than a span they drew or corrected themselves.
SpanOrigin = Literal["manual", "prediction", "prediction-changed"]
SPAN_ORIGINS = get_args(SpanOrigin)


class EntityLabel(StrEnum):
    DISEASE = "disease"
    PHENOTYPE = "phenotype"


class TaskKind(StrEnum):
    INDICATION = "indication"
    CONTRAINDICATION = "contraindication"


class AnnotationStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    ADJUDICATED = "adjudicated"
    REJECTED = "rejected"


class SourceMetadata(BaseModel):
    """Traceability for a task without coupling MedliNER to DAKP runtime objects."""

    model_config = ConfigDict(extra="allow")

    family: str = "unknown"
    document_id: str | None = None
    record_id: str | None = None
    section: str | None = None
    source_uri: str | None = None
    source_hash: str | None = None


class Annotation(BaseModel):
    """Half-open character span, matching Label Studio's exported offsets."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    label: str
    text: str
    annotator: str | None = None
    status: AnnotationStatus = AnnotationStatus.REVIEWED
    provenance: Provenance = "human"
    # None for exports that predate pre-labeling, or from a Label Studio version that omits it.
    origin: SpanOrigin | None = None

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in ALLOWED_LABELS:
            raise ValueError(f"unsupported label {value!r}; expected one of {ALLOWED_LABELS}")
        return value

    @model_validator(mode="after")
    def valid_span(self) -> Annotation:
        if self.end <= self.start:
            raise ValueError("annotation end must be greater than start")
        if self.provenance == "model_suggestion" and self.status in {
            AnnotationStatus.REVIEWED,
            AnnotationStatus.ADJUDICATED,
        }:
            raise ValueError("model suggestions cannot be marked reviewed without human provenance")
        return self


class Example(BaseModel):
    """One imported and normalized annotation task."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    id: str
    text: str
    task: str
    source: SourceMetadata = Field(default_factory=SourceMetadata)
    annotations: list[Annotation] = Field(default_factory=list)
    annotation_status: AnnotationStatus = AnnotationStatus.REVIEWED
    annotation_set_id: str | None = None
    export_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task")
    @classmethod
    def valid_task(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in ALLOWED_TASKS:
            raise ValueError(f"unsupported task {value!r}; expected one of {ALLOWED_TASKS}")
        return value

    @model_validator(mode="after")
    def validate_annotations(self) -> Example:
        seen: set[tuple[int, int]] = set()
        ordered = sorted(self.annotations, key=lambda item: (item.start, item.end, item.label))
        previous: Annotation | None = None
        for annotation in ordered:
            if annotation.end > len(self.text):
                raise ValueError(f"annotation {annotation.id or annotation.text!r} exceeds text length")
            if self.text[annotation.start : annotation.end] != annotation.text:
                raise ValueError(
                    f"annotation text mismatch at [{annotation.start}:{annotation.end}]: "
                    f"expected {self.text[annotation.start : annotation.end]!r}, got {annotation.text!r}"
                )
            key = (annotation.start, annotation.end)
            if key in seen:
                raise ValueError(f"duplicate annotation span {key}")
            if previous is not None and annotation.start < previous.end:
                raise ValueError(f"overlapping annotations are not allowed: {previous.text!r} / {annotation.text!r}")
            seen.add(key)
            previous = annotation
        if self.annotation_status in {AnnotationStatus.REVIEWED, AnnotationStatus.ADJUDICATED}:
            if any(annotation.provenance == "model_suggestion" for annotation in self.annotations):
                raise ValueError("model suggestions must be accepted or replaced by a human before review")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class DatasetManifest(BaseModel):
    schema_version: str = "medliner.dataset.v1"
    dataset_id: str
    input_export_hash: str
    example_count: int = Field(ge=0)
    label_counts: dict[str, int] = Field(default_factory=dict)
    task_counts: dict[str, int] = Field(default_factory=dict)
    # How much of this dataset is an unmodified model suggestion a human merely submitted. The
    # central risk of pre-labeling, and invisible without counting it here.
    origin_counts: dict[str, int] = Field(default_factory=dict)
    split_hash: str | None = None
    annotation_policy_version: str = "medliner.policy.v1"


class SplitManifest(BaseModel):
    schema_version: str = "medliner.splits.v1"
    seed: int
    ratios: dict[str, float]
    group_count: int
    example_count: int
    example_ids: dict[str, list[str]]
    held_out_ids: list[str] = Field(default_factory=list)
    split_hash: str


__all__ = [
    "ALLOWED_LABELS",
    "ALLOWED_TASKS",
    "Annotation",
    "AnnotationStatus",
    "DatasetManifest",
    "EntityLabel",
    "Example",
    "PROVENANCE_VALUES",
    "Provenance",
    "SCHEMA_VERSION",
    "SPAN_ORIGINS",
    "SourceMetadata",
    "SpanOrigin",
    "SplitManifest",
    "TaskKind",
]
