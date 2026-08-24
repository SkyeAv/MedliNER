"""Rubric-gated annotator onboarding for the local Label Studio workflow.

The onboarding project receives answer-free tasks. Gold spans live only in a workdir sidecar, and
attempt/report files are immutable JSON records so retries and audits remain reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from blake3 import blake3
from pydantic import BaseModel, Field, field_validator, model_validator

from .evaluation import load_gold_benchmark
from .label_studio import LabelStudioExportError, normalize_task, read_tasks

CONFIG_SCHEMA_VERSION = "medliner.onboarding.config.v1"
TEST_BANK_SCHEMA_VERSION = "medliner.onboarding.test-bank.v1"
ATTEMPT_SCHEMA_VERSION = "medliner.onboarding.attempt.v1"
REPORT_SCHEMA_VERSION = "medliner.onboarding.report.v1"
PROMOTION_SCHEMA_VERSION = "medliner.onboarding.promotion.v1"
GENERATOR_VERSION = "medliner.onboarding.v1"
DEFAULT_PROJECT_TITLE = "Onboarding"
DEFAULT_CONFIG_PATH = Path("configs/onboarding.json")


class OnboardingError(ValueError):
    """Raised when onboarding state, benchmark data, or an export is unsafe to use."""


class UnknownAnnotatorError(OnboardingError):
    """Raised when an export contains a submitting user without a started attempt."""


class OnboardingConfig(BaseModel):
    schema_version: str = CONFIG_SCHEMA_VERSION
    project_title: str = DEFAULT_PROJECT_TITLE
    test_bank_size: int = Field(default=10, ge=1)
    questions_per_attempt: int = Field(default=4, ge=1)
    pass_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    selection_seed: int = 2026
    case_ids: list[str] | None = None

    @model_validator(mode="after")
    def validate_sizes(self) -> OnboardingConfig:
        if self.questions_per_attempt > self.test_bank_size:
            raise ValueError("questions_per_attempt cannot exceed test_bank_size")
        if self.case_ids is not None and len(self.case_ids) != self.test_bank_size:
            raise ValueError("case_ids must contain exactly test_bank_size ids")
        return self


class GoldSpan(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    label: str
    text: str


class TestBankCase(BaseModel):
    id: str
    task_id: str
    text: str
    task: str
    source_family: str = "unknown"
    source_document_id: str | None = None
    gold: list[GoldSpan] = Field(default_factory=list)


class TestBankManifest(BaseModel):
    schema_version: str = TEST_BANK_SCHEMA_VERSION
    config_schema_version: str = CONFIG_SCHEMA_VERSION
    config_hash: str
    benchmark_hash: str
    test_bank_hash: str
    generated_at: str
    cases: list[TestBankCase]

    @model_validator(mode="after")
    def validate_bank(self) -> TestBankManifest:
        if len(self.cases) < 1:
            raise ValueError("test bank must not be empty")
        ids = [case.id for case in self.cases]
        task_ids = [case.task_id for case in self.cases]
        if len(set(ids)) != len(ids) or len(set(task_ids)) != len(task_ids):
            raise ValueError("test bank case and task ids must be unique")
        return self


class OnboardingAttempt(BaseModel):
    schema_version: str = ATTEMPT_SCHEMA_VERSION
    attempt_id: str
    username: str
    attempt_number: int = Field(ge=1)
    config_hash: str
    test_bank_hash: str
    selected_task_ids: list[str]
    started_at: str
    status: Literal["started", "incomplete", "failed", "passed"] = "started"

    @field_validator("username")
    @classmethod
    def valid_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("username must be non-blank")
        return value


class TaskScore(BaseModel):
    task_id: str
    correct: bool
    submitted: bool
    reason: str | None = None


class OnboardingReport(BaseModel):
    schema_version: str = REPORT_SCHEMA_VERSION
    report_id: str
    attempt_id: str
    username: str
    config_hash: str
    test_bank_hash: str
    export_hash: str
    evaluated_at: str
    status: Literal["incomplete", "failed", "passed"]
    correct_tasks: int
    total_tasks: int
    score: float | None
    task_scores: list[TaskScore]
    unknown_annotators: list[str] = Field(default_factory=list)


class PromotionRecord(BaseModel):
    schema_version: str = PROMOTION_SCHEMA_VERSION
    username: str
    attempt_id: str
    report_id: str
    config_hash: str
    test_bank_hash: str
    promoted_at: str


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_json(value: Any) -> str:
    return _hash_bytes(_canonical(value).encode("utf-8"))


def hash_file(path: str | Path) -> str:
    return _hash_bytes(Path(path).read_bytes())


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> OnboardingConfig:
    try:
        config = OnboardingConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise OnboardingError(f"cannot read onboarding config {path}: {exc}") from exc
    if config.schema_version != CONFIG_SCHEMA_VERSION:
        raise OnboardingError(f"unsupported onboarding config schema {config.schema_version!r}")
    return config


def config_hash(config: OnboardingConfig) -> str:
    return _hash_json(config.model_dump(mode="json", exclude={"schema_version"}))


def _task_id(case_id: str) -> str:
    return f"onboarding-{blake3(f'onboarding\\n{case_id}'.encode()).hexdigest()[:16]}"


def _case_from_example(example: Any) -> TestBankCase:
    return TestBankCase(
        id=example.id,
        task_id=_task_id(example.id),
        text=example.text,
        task=example.task,
        source_family=example.source.family,
        source_document_id=example.source.document_id or example.source.record_id,
        gold=[
            GoldSpan(start=item.start, end=item.end, label=item.label, text=item.text) for item in example.annotations
        ],
    )


def build_test_bank(
    benchmark_path: str | Path,
    config: OnboardingConfig,
    *,
    generated_at: str | None = None,
) -> TestBankManifest:
    """Build the private, versioned bank from the adjudicated benchmark."""
    try:
        examples = load_gold_benchmark(benchmark_path)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise OnboardingError(f"cannot load onboarding benchmark {benchmark_path}: {exc}") from exc
    by_id = {example.id: example for example in examples}
    if config.case_ids is None:
        selected = sorted(examples, key=lambda item: item.id)[: config.test_bank_size]
    else:
        missing = [case_id for case_id in config.case_ids if case_id not in by_id]
        if missing:
            raise OnboardingError(f"onboarding test bank ids are missing from benchmark: {missing[:5]}")
        selected = [by_id[case_id] for case_id in config.case_ids]
    if len(selected) != config.test_bank_size:
        raise OnboardingError(
            f"benchmark has only {len(selected)} usable onboarding cases; need {config.test_bank_size}"
        )
    cases = [_case_from_example(example) for example in selected]
    config_hash_value = config_hash(config)
    benchmark_hash = hash_file(benchmark_path)
    case_payload = [case.model_dump(mode="json") for case in cases]
    bank_hash = _hash_json({"config_hash": config_hash_value, "benchmark_hash": benchmark_hash, "cases": case_payload})
    return TestBankManifest(
        config_hash=config_hash_value,
        benchmark_hash=benchmark_hash,
        test_bank_hash=bank_hash,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        cases=cases,
    )


def write_test_bank(manifest: TestBankManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read_test_bank(path: str | Path) -> TestBankManifest:
    try:
        return TestBankManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise OnboardingError(f"cannot read onboarding test bank {path}: {exc}") from exc


def build_onboarding_tasks(manifest: TestBankManifest, task_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Return Label Studio tasks without the private gold spans."""
    wanted = set(task_ids) if task_ids is not None else None
    tasks: list[dict[str, Any]] = []
    for case in manifest.cases:
        if wanted is not None and case.task_id not in wanted:
            continue
        data: dict[str, Any] = {
            "text": case.text,
            "task": case.task,
            "source_family": "onboarding",
            "source_document_id": case.id,
            "generator_version": GENERATOR_VERSION,
            "onboarding": True,
            "test_bank_hash": manifest.test_bank_hash,
        }
        tasks.append({"id": case.task_id, "data": data})
    if wanted is not None and {task["id"] for task in tasks} != wanted:
        missing = sorted(wanted - {task["id"] for task in tasks})
        raise OnboardingError(f"onboarding task ids are not in the test bank: {missing[:5]}")
    return tasks


def state_dir(workdir: str | Path) -> Path:
    return Path(workdir) / "onboarding"


def versioned_bank_path(workdir: str | Path, manifest: TestBankManifest) -> Path:
    return state_dir(workdir) / "banks" / f"test-bank-{manifest.test_bank_hash}.json"


def write_current_bank_pointer(workdir: str | Path, manifest: TestBankManifest) -> Path:
    path = state_dir(workdir) / "current-test-bank.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _canonical({"schema_version": TEST_BANK_SCHEMA_VERSION, "test_bank_hash": manifest.test_bank_hash}) + "\n",
        encoding="utf-8",
    )
    return path


def _attempt_dir(workdir: str | Path) -> Path:
    return state_dir(workdir) / "attempts"


def _report_dir(workdir: str | Path) -> Path:
    return state_dir(workdir) / "reports"


def _safe_username(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@-]+", "_", value.strip()) or "unknown"


def read_attempts(workdir: str | Path, *, username: str | None = None) -> list[OnboardingAttempt]:
    directory = _attempt_dir(workdir)
    if not directory.exists():
        return []
    attempts: list[OnboardingAttempt] = []
    for path in sorted(directory.glob("*.json")):
        try:
            attempt = OnboardingAttempt.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise OnboardingError(f"invalid onboarding attempt {path}: {exc}") from exc
        if username is None or attempt.username == username:
            attempts.append(attempt)
    return sorted(attempts, key=lambda item: (item.username, item.attempt_number, item.attempt_id))


def _selection(
    manifest: TestBankManifest, config: OnboardingConfig, username: str, number: int, previous: list[set[str]]
) -> list[str]:
    previous_sets = {frozenset(item) for item in previous}
    selected: list[str] = []
    # There are finitely many combinations; keep trying deterministic salts until a fresh one is
    # found, then allow a repeat only after the bank's combinations have been exhausted.
    for salt in range(len(manifest.cases) + 1):
        ranked = sorted(
            manifest.cases,
            key=lambda case: blake3(
                f"{config.selection_seed}\n{username}\n{number}\n{salt}\n{case.task_id}".encode()
            ).hexdigest(),
        )
        selected = [case.task_id for case in ranked[: config.questions_per_attempt]]
        if frozenset(selected) not in previous_sets:
            return selected
    return selected


def start_attempt(
    workdir: str | Path,
    manifest: TestBankManifest,
    config: OnboardingConfig,
    username: str,
) -> OnboardingAttempt:
    username = username.strip()
    if not username:
        raise OnboardingError("annotator username must be non-blank")
    existing = [
        item
        for item in read_attempts(workdir, username=username)
        if item.config_hash == manifest.config_hash and item.test_bank_hash == manifest.test_bank_hash
    ]
    number = max((item.attempt_number for item in existing), default=0) + 1
    selected = _selection(manifest, config, username, number, [set(item.selected_task_ids) for item in existing])
    attempt_id = _hash_json(
        {
            "username": username,
            "attempt_number": number,
            "config_hash": manifest.config_hash,
            "test_bank_hash": manifest.test_bank_hash,
            "selected_task_ids": selected,
        }
    )[:24]
    attempt = OnboardingAttempt(
        attempt_id=attempt_id,
        username=username,
        attempt_number=number,
        config_hash=manifest.config_hash,
        test_bank_hash=manifest.test_bank_hash,
        selected_task_ids=selected,
        started_at=datetime.now(UTC).isoformat(),
    )
    path = _attempt_dir(workdir) / f"{_safe_username(username)}-{attempt_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return OnboardingAttempt.model_validate_json(path.read_text(encoding="utf-8"))
    path.write_text(attempt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return attempt


def _export_hash(path: str | Path) -> str:
    return hash_file(path)


def _annotation_user(annotation_set: dict[str, Any]) -> str | None:
    value = annotation_set.get("created_username") or annotation_set.get("completed_by")
    if isinstance(value, dict):
        value = value.get("username") or value.get("email") or value.get("id")
    return str(value).strip() if value is not None and str(value).strip() else None


def _annotation_key(annotation_set: dict[str, Any]) -> tuple[str, str]:
    return (
        str(annotation_set.get("updated_at") or annotation_set.get("created_at") or ""),
        str(annotation_set.get("id") or ""),
    )


def _submissions_by_user(tasks: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    submissions: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        task_id = str(task.get("id"))
        raw_sets = task.get("annotations", [])
        if not isinstance(raw_sets, list):
            raise OnboardingError(f"task {task_id!r} has malformed annotations")
        for annotation_set in raw_sets:
            if not isinstance(annotation_set, dict) or annotation_set.get("was_cancelled", False):
                continue
            username = _annotation_user(annotation_set)
            if username is None:
                raise OnboardingError(f"task {task_id!r} has a completed annotation without an annotator")
            key = (username, task_id)
            if key not in submissions or _annotation_key(annotation_set) >= _annotation_key(submissions[key]):
                submissions[key] = {**task, "annotations": [annotation_set]}
    return submissions


def _gold_set(case: TestBankCase) -> set[tuple[int, int, str]]:
    return {(span.start, span.end, span.label) for span in case.gold}


def evaluate_attempt(
    export_path: str | Path,
    workdir: str | Path,
    manifest: TestBankManifest,
    config: OnboardingConfig,
    attempt: OnboardingAttempt,
) -> OnboardingReport:
    """Score one attempt; repeated evaluation of the same export is idempotent."""
    if attempt.config_hash != manifest.config_hash or attempt.test_bank_hash != manifest.test_bank_hash:
        raise OnboardingError("attempt belongs to a different onboarding test-bank/config version; start a new attempt")
    try:
        tasks = read_tasks(export_path)
    except (OSError, UnicodeError, LabelStudioExportError) as exc:
        raise OnboardingError(f"cannot read onboarding export {export_path}: {exc}") from exc
    submissions = _submissions_by_user(tasks)
    known_attempt_users = {item.username for item in read_attempts(workdir)}
    unknown = sorted(username for username, _task_id in submissions if username not in known_attempt_users)
    if unknown:
        raise UnknownAnnotatorError(f"export contains annotators without onboarding attempts: {unknown}")
    by_task_id = {case.task_id: case for case in manifest.cases}
    scores: list[TaskScore] = []
    for task_id in attempt.selected_task_ids:
        case = by_task_id.get(task_id)
        if case is None:
            raise OnboardingError(f"attempt references unknown test-bank task {task_id!r}")
        submission = submissions.get((attempt.username, task_id))
        if submission is None:
            scores.append(TaskScore(task_id=task_id, correct=False, submitted=False, reason="incomplete"))
            continue
        try:
            example = normalize_task(submission)
            predicted = {(item.start, item.end, item.label) for item in example.annotations}
        except (LabelStudioExportError, ValueError) as exc:
            scores.append(TaskScore(task_id=task_id, correct=False, submitted=True, reason=str(exc)))
            continue
        correct = predicted == _gold_set(case)
        scores.append(
            TaskScore(
                task_id=task_id, correct=correct, submitted=True, reason=None if correct else "span/type mismatch"
            )
        )
    complete = all(item.submitted for item in scores)
    correct_count = sum(item.correct for item in scores)
    score = correct_count / len(scores) if complete and scores else None
    status: Literal["incomplete", "failed", "passed"]
    if not complete:
        status = "incomplete"
    elif score is not None and score >= config.pass_threshold:
        status = "passed"
    else:
        status = "failed"
    report_id = _hash_json({"attempt_id": attempt.attempt_id, "export_hash": _export_hash(export_path)})[:24]
    return OnboardingReport(
        report_id=report_id,
        attempt_id=attempt.attempt_id,
        username=attempt.username,
        config_hash=manifest.config_hash,
        test_bank_hash=manifest.test_bank_hash,
        export_hash=_export_hash(export_path),
        evaluated_at=datetime.now(UTC).isoformat(),
        status=status,
        correct_tasks=correct_count,
        total_tasks=len(scores),
        score=score,
        task_scores=scores,
        unknown_annotators=unknown,
    )


def write_report(report: OnboardingReport, workdir: str | Path) -> Path:
    path = _report_dir(workdir) / f"{report.attempt_id}-{report.export_hash}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read_reports(workdir: str | Path) -> list[OnboardingReport]:
    directory = _report_dir(workdir)
    if not directory.exists():
        return []
    reports: list[OnboardingReport] = []
    for path in sorted(directory.glob("*.json")):
        try:
            reports.append(OnboardingReport.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise OnboardingError(f"invalid onboarding report {path}: {exc}") from exc
    return reports


def promote(workdir: str | Path, report: OnboardingReport, manifest: TestBankManifest) -> PromotionRecord:
    if report.status != "passed":
        raise OnboardingError("only a passing onboarding report can promote an annotator")
    if report.test_bank_hash != manifest.test_bank_hash or report.config_hash != manifest.config_hash:
        raise OnboardingError("report belongs to a different onboarding test-bank/config version")
    record = PromotionRecord(
        username=report.username,
        attempt_id=report.attempt_id,
        report_id=report.report_id,
        config_hash=report.config_hash,
        test_bank_hash=report.test_bank_hash,
        promoted_at=datetime.now(UTC).isoformat(),
    )
    path = state_dir(workdir) / "promotions" / f"{_safe_username(record.username)}-{record.test_bank_hash}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    else:
        record = PromotionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    return record


def promoted_users(workdir: str | Path, manifest: TestBankManifest) -> set[str]:
    directory = state_dir(workdir) / "promotions"
    if not directory.exists():
        return set()
    users: set[str] = set()
    for path in directory.glob("*.json"):
        try:
            record = PromotionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise OnboardingError(f"invalid onboarding promotion {path}: {exc}") from exc
        if record.test_bank_hash == manifest.test_bank_hash and record.config_hash == manifest.config_hash:
            users.add(record.username)
    return users


def filter_production_export(
    export_path: str | Path,
    workdir: str | Path,
    manifest: TestBankManifest,
    allowed_users: set[str],
) -> tuple[Path, Path, int]:
    """Keep only production annotations from promoted users and retain excluded audit data.

    Tasks with no live annotation are preserved. A task with only non-promoted live annotations
    is excluded from the training input, while the original task and the excluded annotation
    sets are written to an operator-readable audit artifact.
    """
    tasks = read_tasks(export_path)
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for task in tasks:
        annotations = task.get("annotations", [])
        if not isinstance(annotations, list):
            raise OnboardingError(f"production task {task.get('id')!r} has malformed annotations")
        live = [item for item in annotations if isinstance(item, dict) and not item.get("was_cancelled", False)]
        if not live:
            excluded.append({"task": task, "reason": "not_annotated", "excluded_annotators": []})
            continue
        non_promoted = [item for item in live if _annotation_user(item) not in allowed_users]
        promoted = [item for item in live if _annotation_user(item) in allowed_users]
        if non_promoted and not promoted:
            excluded.append(
                {
                    "task": task,
                    "reason": "annotator_not_promoted",
                    "excluded_annotators": sorted(
                        {user for user in (_annotation_user(item) for item in non_promoted) if user}
                    ),
                }
            )
            continue
        if promoted and len(promoted) != len(live):
            copied = dict(task)
            copied["annotations"] = promoted
            eligible.append(copied)
        else:
            eligible.append(task)
    source_hash = _export_hash(export_path)
    audit_path = state_dir(workdir) / "audit" / f"excluded-production-{source_hash}.json"
    eligible_path = state_dir(workdir) / "eligible" / f"production-{source_hash}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    eligible_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "medliner.onboarding.excluded-production.v1",
                "source_export_hash": source_hash,
                "test_bank_hash": manifest.test_bank_hash,
                "allowed_users": sorted(allowed_users),
                "excluded_count": len(excluded),
                "excluded": excluded,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    eligible_path.write_text(json.dumps(eligible, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return eligible_path, audit_path, len(excluded)


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_PROJECT_TITLE",
    "GENERATOR_VERSION",
    "GoldSpan",
    "OnboardingAttempt",
    "OnboardingConfig",
    "OnboardingError",
    "OnboardingReport",
    "PromotionRecord",
    "REPORT_SCHEMA_VERSION",
    "TEST_BANK_SCHEMA_VERSION",
    "TestBankCase",
    "TestBankManifest",
    "UnknownAnnotatorError",
    "build_onboarding_tasks",
    "build_test_bank",
    "config_hash",
    "evaluate_attempt",
    "filter_production_export",
    "load_config",
    "promote",
    "promoted_users",
    "read_attempts",
    "read_reports",
    "read_test_bank",
    "start_attempt",
    "state_dir",
    "versioned_bank_path",
    "write_current_bank_pointer",
    "write_report",
    "write_test_bank",
]
