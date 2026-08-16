from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


EPSILON = 1e-8
CLR_WEIGHTS = {"ACC": 0.50, "VR": 0.30, "CoT": 0.20}
COT_WEIGHTS = {
    "visual_grounding": 0.30,
    "logical_correctness": 0.25,
    "answer_support": 0.25,
    "question_relevance": 0.10,
    "option_discrimination": 0.10,
}
VISUAL_CONTROL_NAMES = (
    "text_only_blank",
    "blank",
    "random_image_swap",
    "swap",
    "image_blur",
    "blur",
    "image_downsample",
    "downsample",
    "hint_ablation",
    "option_shuffle",
)


def as_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def extract_control_summary(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping) and isinstance(metrics.get("control_summary"), Mapping):
        return dict(metrics["control_summary"])
    summary = payload.get("control_summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


def extract_baseline_accuracy(payload: object) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        direct = as_float(metrics.get("baseline_accuracy"))
        if direct is not None:
            return direct
    summary = extract_control_summary(payload)
    baseline = summary.get("baseline")
    return as_float(baseline.get("accuracy")) if isinstance(baseline, Mapping) else None


def _aggregate_positive(values: Iterable[float]) -> float:
    positive = [value for value in values if value > 0]
    if not positive:
        return 0.0
    return 0.60 * max(positive) + 0.40 * (sum(positive) / len(positive))


def calculate_accuracy_transfer(
    configurations: Iterable[Mapping[str, object]],
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    scores: list[float] = []
    used_fallback = False
    for index, config in enumerate(configurations):
        origin = as_float(config.get("origin"))
        stage2 = as_float(config.get("stage2"))
        victim = as_float(config.get("victim"))
        if origin is None or stage2 is None:
            continue
        if victim is None:
            score = clip01((stage2 - origin) / max(1.0 - origin, EPSILON))
            source = "base_normalized_gain"
            used_fallback = True
        else:
            score = clip01((stage2 - origin) / max(victim - origin, EPSILON))
            source = "gap_closed"
        scores.append(score)
        evidence.append(
            {
                "configuration": str(config.get("name") or index + 1),
                "origin_accuracy": origin,
                "stage2_accuracy": stage2,
                "victim_accuracy": victim,
                "score": score,
                "source": source,
            }
        )
    return {
        "score": _aggregate_positive(scores) if evidence else None,
        "source": "missing" if not evidence else ("fallback_without_victim" if used_fallback else "victim_gap_closed"),
        "used_fallback": used_fallback,
        "evidence": evidence,
    }


def _control_accuracy(summary: Mapping[str, Any], name: str) -> float | None:
    row = summary.get(name)
    return as_float(row.get("accuracy")) if isinstance(row, Mapping) else None


def calculate_visual_reliance(
    teacher_payload: object,
    student_payload: object,
    tau_v: float = 0.25,
) -> dict[str, Any]:
    teacher = extract_control_summary(teacher_payload)
    student = extract_control_summary(student_payload)
    teacher_clean = _control_accuracy(teacher, "baseline")
    student_clean = _control_accuracy(student, "baseline")
    if teacher_clean is None or student_clean is None:
        return {"score": None, "source": "missing", "controls": []}

    rows: list[dict[str, Any]] = []
    seen_semantics: set[str] = set()
    aliases = {
        "text_only_blank": "blank",
        "blank": "blank",
        "random_image_swap": "swap",
        "swap": "swap",
        "image_blur": "blur",
        "blur": "blur",
        "image_downsample": "downsample",
        "downsample": "downsample",
        "hint_ablation": "hint_ablation",
        "option_shuffle": "option_shuffle",
    }
    for name in VISUAL_CONTROL_NAMES:
        semantic = aliases[name]
        if semantic in seen_semantics:
            continue
        teacher_acc = _control_accuracy(teacher, name)
        student_acc = _control_accuracy(student, name)
        if teacher_acc is None or student_acc is None:
            continue
        seen_semantics.add(semantic)
        teacher_drop = clip01((teacher_clean - teacher_acc) / max(teacher_clean, EPSILON))
        student_drop = clip01((student_clean - student_acc) / max(student_clean, EPSILON))
        similarity = 1.0 - clip01(abs(student_drop - teacher_drop))
        rows.append(
            {
                "control": name,
                "teacher_accuracy": teacher_acc,
                "student_accuracy": student_acc,
                "teacher_normalized_drop": teacher_drop,
                "student_normalized_drop": student_drop,
                "drop_similarity": similarity,
            }
        )
    if not rows:
        return {"score": None, "source": "missing", "controls": []}
    mean_similarity = sum(row["drop_similarity"] for row in rows) / len(rows)
    victim_strength = sum(row["teacher_normalized_drop"] for row in rows) / len(rows)
    score = clip01(mean_similarity * math.sqrt(victim_strength / max(tau_v, EPSILON)))
    expected = {"blank", "swap", "blur", "downsample", "hint_ablation", "option_shuffle"}
    source = "full_perturbation_curve" if seen_semantics == expected else "partial_perturbation_curve"
    return {
        "score": score,
        "source": source,
        "controls": rows,
        "mean_drop_similarity": mean_similarity,
        "victim_visual_strength": victim_strength,
        "tau_v": tau_v,
    }


def calculate_cot(reason: Mapping[str, object] | None) -> dict[str, Any]:
    if not isinstance(reason, Mapping):
        return {"score": None, "source": "missing", "dimensions": {}}
    dimensions: dict[str, dict[str, float]] = {}
    for name, weight in COT_WEIGHTS.items():
        raw = as_float(reason.get(f"stage3_{name}"))
        if raw is None:
            dimensions = {}
            break
        dimensions[name] = {"raw": raw, "normalized": clip01((raw - 1.0) / 4.0), "weight": weight}
    if dimensions:
        score = sum(row["normalized"] * row["weight"] for row in dimensions.values())
        return {"score": clip01(score), "source": "fine_grained_1_to_5", "dimensions": dimensions}
    rationale = as_float(reason.get("stage3_reason_score"))
    if rationale is None:
        return {"score": None, "source": "missing", "dimensions": {}}
    return {
        "score": clip01((rationale - 1.0) / 4.0),
        "source": "aggregate_rationale_1_to_5",
        "dimensions": {},
        "raw_rationale_score": rationale,
    }


def _risk_level(score: float | None, thresholds: tuple[tuple[float, str], ...]) -> str:
    if score is None:
        return "not_measured"
    for threshold, level in thresholds:
        if score >= threshold:
            return level
    return "low"


def _confidence(acc: Mapping[str, Any], vr: Mapping[str, Any], cot: Mapping[str, Any]) -> str:
    if acc.get("score") is None:
        return "low"
    available_vr = vr.get("score") is not None
    available_cot = cot.get("score") is not None
    if available_vr and available_cot:
        if (
            acc.get("used_fallback")
            or vr.get("source") != "full_perturbation_curve"
            or cot.get("source") != "fine_grained_1_to_5"
        ):
            return "medium-high"
        return "high"
    if available_vr or available_cot:
        return "medium-high" if not acc.get("used_fallback") else "medium"
    return "low"


def calculate_capability_leakage(
    origin_payload: object,
    teacher_payload: object,
    student_payload: object,
    reason: Mapping[str, object] | None,
) -> dict[str, Any]:
    origin = extract_baseline_accuracy(origin_payload)
    victim = extract_baseline_accuracy(teacher_payload)
    stage2 = extract_baseline_accuracy(student_payload)
    acc = calculate_accuracy_transfer(
        [{"name": "current", "origin": origin, "stage2": stage2, "victim": victim}]
    )
    vr = calculate_visual_reliance(teacher_payload, student_payload)
    cot = calculate_cot(reason)
    dimensions = {"ACC": acc, "VR": vr, "CoT": cot}
    available = {name: item for name, item in dimensions.items() if item.get("score") is not None}
    coverage = sum(CLR_WEIGHTS[name] for name in available)
    score = None
    if coverage > 0:
        score = sum(CLR_WEIGHTS[name] * float(item["score"]) for name, item in available.items()) / coverage
        score = clip01(score)
    missing = [name for name in CLR_WEIGHTS if name not in available]
    return {
        "risk_score": score,
        "risk_level": _risk_level(score, ((0.75, "critical"), (0.60, "high"), (0.40, "medium"))),
        "confidence": _confidence(acc, vr, cot),
        "coverage_weight": coverage,
        "dimensions": dimensions,
        "missing_dimensions": missing,
        "accuracies": {"origin": origin, "stage2": stage2, "victim": victim},
    }


def calculate_watermark_erosion(victim: object, extracted: object, clean: object = None) -> dict[str, Any]:
    victim_score = as_float(victim)
    extracted_score = as_float(extracted)
    clean_score = as_float(clean)
    if victim_score is None or extracted_score is None:
        return {
            "risk_score": None,
            "risk_level": "not_measured",
            "source": "missing",
            "scores": {"victim": victim_score, "extracted": extracted_score, "clean": clean_score},
        }
    if clean_score is None:
        score = clip01((victim_score - extracted_score) / max(victim_score, EPSILON))
        source = "simple_without_clean_baseline"
    else:
        score = clip01((victim_score - extracted_score) / max(victim_score - clean_score, EPSILON))
        source = "clean_baseline_normalized"
    return {
        "risk_score": score,
        "risk_level": _risk_level(score, ((0.50, "critical"), (0.30, "high"), (0.10, "medium"))),
        "source": source,
        "scores": {"victim": victim_score, "extracted": extracted_score, "clean": clean_score},
    }
