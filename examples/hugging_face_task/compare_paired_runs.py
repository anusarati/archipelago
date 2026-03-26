#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


@dataclass(frozen=True)
class RunInfo:
    path: Path
    label: str
    tasks: dict[str, Path]
    summary_scores: dict[str, float]


@dataclass
class MetricResult:
    name: str
    pairs: list[tuple[str, float, float]]
    missing_a: list[str]
    missing_b: list[str]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _discover_tasks(run_dir: Path) -> dict[str, Path]:
    tasks: dict[str, Path] = {}
    if not run_dir.exists():
        return tasks
    for entry in run_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("task_"):
            tasks[entry.name] = entry
    return tasks


def _load_summary_scores(run_dir: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for summary_path in run_dir.glob("world_*/summary.json"):
        summary_data = _load_json(summary_path)
        if not summary_data:
            continue
        entries = summary_data.get("summary", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            task_id = entry.get("task_id")
            score = entry.get("final_score")
            if isinstance(task_id, str) and isinstance(score, (int, float)):
                scores[task_id] = float(score)
    return scores


def _detect_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str) and part.strip():
                return True
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return True
    return True


def _assistant_message_count(task_dir: Path, mode: str) -> int | None:
    trajectory_path = task_dir / "trajectory.json"
    data = _load_json(trajectory_path)
    if not data:
        return None
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return None
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        if mode == "content" and not _detect_content(message):
            continue
        count += 1
    return count


def _final_score(task_id: str, task_dir: Path, summary_scores: dict[str, float]) -> float | None:
    grades_path = task_dir / "grades.json"
    data = _load_json(grades_path)
    if data:
        scoring = data.get("scoring_results", {})
        if isinstance(scoring, dict):
            score = scoring.get("final_score")
            if isinstance(score, (int, float)):
                return float(score)
    score = summary_scores.get(task_id)
    return float(score) if isinstance(score, (int, float)) else None


def _collect_metrics(
    run_a: RunInfo,
    run_b: RunInfo,
    assistant_mode: str,
) -> tuple[MetricResult, MetricResult]:
    common_task_ids = sorted(set(run_a.tasks) & set(run_b.tasks))
    assistant_pairs: list[tuple[str, float, float]] = []
    assistant_missing_a: list[str] = []
    assistant_missing_b: list[str] = []
    score_pairs: list[tuple[str, float, float]] = []
    score_missing_a: list[str] = []
    score_missing_b: list[str] = []

    for task_id in common_task_ids:
        task_a = run_a.tasks[task_id]
        task_b = run_b.tasks[task_id]

        assistant_a = _assistant_message_count(task_a, assistant_mode)
        assistant_b = _assistant_message_count(task_b, assistant_mode)
        if assistant_a is None:
            assistant_missing_a.append(task_id)
        if assistant_b is None:
            assistant_missing_b.append(task_id)
        if assistant_a is not None and assistant_b is not None:
            assistant_pairs.append((task_id, float(assistant_a), float(assistant_b)))

        score_a = _final_score(task_id, task_a, run_a.summary_scores)
        score_b = _final_score(task_id, task_b, run_b.summary_scores)
        if score_a is None:
            score_missing_a.append(task_id)
        if score_b is None:
            score_missing_b.append(task_id)
        if score_a is not None and score_b is not None:
            score_pairs.append((task_id, float(score_a), float(score_b)))

    return (
        MetricResult(
            name="assistant_messages",
            pairs=assistant_pairs,
            missing_a=assistant_missing_a,
            missing_b=assistant_missing_b,
        ),
        MetricResult(
            name="final_score",
            pairs=score_pairs,
            missing_a=score_missing_a,
            missing_b=score_missing_b,
        ),
    )


def _format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _summary_stats(values: np.ndarray) -> dict[str, Any]:
    skewness = None
    if stats is not None and values.size > 2:
        try:
            skewness = float(stats.skew(values))
        except Exception:
            pass
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "skewness": skewness,
    }


def _paired_test(a_vals: np.ndarray, b_vals: np.ndarray, diff: np.ndarray, alpha: float) -> dict[str, Any]:
    if stats is None:
        return {"test": "unavailable", "reason": "scipy_not_installed"}
    if diff.size < 3:
        return {"test": "unavailable", "reason": "insufficient_pairs"}
    if np.allclose(diff, diff[0]):
        return {"test": "unavailable", "reason": "no_variation"}

    normality_p = None
    use_ttest = False
    if 3 <= diff.size <= 5000:
        try:
            normality_p = float(stats.shapiro(diff).pvalue)
            use_ttest = normality_p >= alpha
        except Exception:
            use_ttest = False

    ret: dict[str, Any] = {}
    if use_ttest:
        result = stats.ttest_rel(diff, np.zeros_like(diff))
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
        sd = float(np.std(diff, ddof=1))
        cohen_d = float(np.mean(diff) / sd) if sd else None
        ret = {
            "test": "paired_t",
            "statistic": t_stat,
            "p_value": p_value,
            "effect_size": cohen_d,
            "effect_size_label": "cohen_d",
            "normality_p": normality_p,
        }
    else:
        try:
            result = stats.wilcoxon(diff, zero_method="pratt", alternative="two-sided")
            w_stat = float(result.statistic)
            p_value = float(result.pvalue)
            if p_value > 0:
                z_score = float(stats.norm.isf(p_value / 2))
                r = z_score / math.sqrt(diff.size)
                if float(np.median(diff)) < 0:
                    r = -r
            else:
                r = None
            ret = {
                "test": "wilcoxon_signed_rank",
                "statistic": w_stat,
                "p_value": p_value,
                "effect_size": r,
                "effect_size_label": "r",
                "normality_p": normality_p,
            }
        except Exception as exc:
            ret = {"test": "unavailable", "reason": f"wilcoxon_error:{exc}", "normality_p": normality_p}

    pos = int(np.sum(diff > 0))
    neg = int(np.sum(diff < 0))
    if pos + neg > 0:
        try:
            if hasattr(stats, 'binomtest'):
                ret["sign_p"] = float(stats.binomtest(min(pos, neg), pos + neg, 0.5).pvalue)
            else:
                ret["sign_p"] = float(stats.binom_test(min(pos, neg), pos + neg, 0.5))
        except Exception:
            pass

    try:
        ks_res = stats.ks_2samp(a_vals, b_vals)
        ret["ks_stat"] = float(ks_res.statistic)
        ret["ks_p"] = float(ks_res.pvalue)
    except Exception:
        pass

    try:
        # anderson_ksamp might raise error on many ties or exact matches, so it uses try-except
        ad_res = stats.anderson_ksamp([a_vals, b_vals])
        ret["ad_stat"] = float(ad_res.statistic)
        ret["ad_p"] = float(getattr(ad_res, 'pvalue', getattr(ad_res, 'significance_level', None)))
    except Exception:
        pass

    return ret


def _render_metric(
    metric: MetricResult, label_a: str, label_b: str, alpha: float
) -> tuple[str, dict[str, Any]]:
    pairs = metric.pairs
    if not pairs:
        return (
            f"{metric.name}: no paired samples available",
            {
                "metric": metric.name,
                "pairs": 0,
                "missing_a": metric.missing_a,
                "missing_b": metric.missing_b,
            },
        )
    a_vals = np.array([p[1] for p in pairs], dtype=float)
    b_vals = np.array([p[2] for p in pairs], dtype=float)
    diff = b_vals - a_vals
    stats_a = _summary_stats(a_vals)
    stats_b = _summary_stats(b_vals)
    stats_diff = _summary_stats(diff)
    test_result = _paired_test(a_vals, b_vals, diff, alpha)

    lines = [
        f"{metric.name}: paired samples = {stats_diff['n']}",
        f"{label_a}: mean={_format_float(stats_a['mean'])} "
        f"median={_format_float(stats_a['median'])} "
        f"std={_format_float(stats_a['std'])} "
        f"min={_format_float(stats_a['min'])} "
        f"max={_format_float(stats_a['max'])} "
        f"skewness={_format_float(stats_a.get('skewness'))}",
        f"{label_b}: mean={_format_float(stats_b['mean'])} "
        f"median={_format_float(stats_b['median'])} "
        f"std={_format_float(stats_b['std'])} "
        f"min={_format_float(stats_b['min'])} "
        f"max={_format_float(stats_b['max'])} "
        f"skewness={_format_float(stats_b.get('skewness'))}",
        f"diff (B-A): mean={_format_float(stats_diff['mean'])} "
        f"median={_format_float(stats_diff['median'])} "
        f"std={_format_float(stats_diff['std'])} "
        f"min={_format_float(stats_diff['min'])} "
        f"max={_format_float(stats_diff['max'])} "
        f"skewness={_format_float(stats_diff.get('skewness'))}",
    ]

    if test_result.get("test") == "paired_t":
        lines.append(
            "test: paired t-test (two-sided), "
            f"t={_format_float(test_result.get('statistic'))}, "
            f"p={_format_float(test_result.get('p_value'))}, "
            f"d={_format_float(test_result.get('effect_size'))}"
        )
    elif test_result.get("test") == "wilcoxon_signed_rank":
        lines.append(
            "test: Wilcoxon signed-rank (two-sided), "
            f"W={_format_float(test_result.get('statistic'))}, "
            f"p={_format_float(test_result.get('p_value'))}, "
            f"r={_format_float(test_result.get('effect_size'))}"
        )
    else:
        lines.append(
            f"test: unavailable ({test_result.get('reason','unknown')})"
        )

    if test_result.get("sign_p") is not None:
        lines.append(f"test: Sign Test, p={_format_float(test_result.get('sign_p'))}")

    if test_result.get("ks_p") is not None:
        lines.append(
            "test: Two-sample KS, "
            f"statistic={_format_float(test_result.get('ks_stat'))}, "
            f"p={_format_float(test_result.get('ks_p'))}"
        )

    if test_result.get("ad_p") is not None:
        lines.append(
            "test: Two-sample Anderson-Darling, "
            f"statistic={_format_float(test_result.get('ad_stat'))}, "
            f"p={_format_float(test_result.get('ad_p'))}"
        )

    if test_result.get("normality_p") is not None:
        lines.append(
            f"normality (Shapiro) p={_format_float(test_result.get('normality_p'))}"
        )

    summary = {
        "metric": metric.name,
        "pairs": stats_diff["n"],
        "summary_a": stats_a,
        "summary_b": stats_b,
        "summary_diff": stats_diff,
        "test": test_result,
        "missing_a": metric.missing_a,
        "missing_b": metric.missing_b,
    }
    return "\n".join(lines), summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, "")
                if value is None:
                    value = ""
                values.append(str(value))
            f.write(",".join(values) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two paired run directories for assistant message counts "
            "and final scores."
        )
    )
    parser.add_argument("--run-a", required=True, help="First run directory.")
    parser.add_argument("--run-b", required=True, help="Second run directory.")
    parser.add_argument("--label-a", help="Label for run A (default: folder name).")
    parser.add_argument("--label-b", help="Label for run B (default: folder name).")
    parser.add_argument(
        "--assistant-mode",
        choices=["all", "content"],
        default="all",
        help="How to count assistant messages (default: all).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Alpha for normality checks."
    )
    parser.add_argument("--json", dest="json_path", help="Write JSON summary file.")
    parser.add_argument("--csv", dest="csv_path", help="Write CSV of per-task pairs.")
    args = parser.parse_args()

    run_a_path = Path(args.run_a)
    run_b_path = Path(args.run_b)
    if not run_a_path.exists() or not run_b_path.exists():
        print("Run directory not found.", file=sys.stderr)
        return 2

    run_a = RunInfo(
        path=run_a_path,
        label=args.label_a or run_a_path.name,
        tasks=_discover_tasks(run_a_path),
        summary_scores=_load_summary_scores(run_a_path),
    )
    run_b = RunInfo(
        path=run_b_path,
        label=args.label_b or run_b_path.name,
        tasks=_discover_tasks(run_b_path),
        summary_scores=_load_summary_scores(run_b_path),
    )

    assistant_metric, score_metric = _collect_metrics(
        run_a, run_b, args.assistant_mode
    )

    common_tasks = sorted(set(run_a.tasks) & set(run_b.tasks))
    print(f"Run A: {run_a.label} ({run_a.path})")
    print(f"Run B: {run_b.label} ({run_b.path})")
    print(f"Tasks in A: {len(run_a.tasks)}")
    print(f"Tasks in B: {len(run_b.tasks)}")
    print(f"Paired tasks: {len(common_tasks)}")
    if args.assistant_mode == "content":
        print("Assistant message count: content-only messages")
    else:
        print("Assistant message count: all assistant messages (including tool calls)")
    print("")

    assistant_text, assistant_summary = _render_metric(
        assistant_metric, run_a.label, run_b.label, args.alpha
    )
    score_text, score_summary = _render_metric(
        score_metric, run_a.label, run_b.label, args.alpha
    )
    print(assistant_text)
    print("")
    print(score_text)

    if args.json_path:
        summary_payload = {
            "run_a": {
                "path": str(run_a.path),
                "label": run_a.label,
                "task_count": len(run_a.tasks),
            },
            "run_b": {
                "path": str(run_b.path),
                "label": run_b.label,
                "task_count": len(run_b.tasks),
            },
            "paired_task_count": len(common_tasks),
            "assistant_messages": assistant_summary,
            "final_score": score_summary,
        }
        Path(args.json_path).write_text(json.dumps(summary_payload, indent=2))

    if args.csv_path:
        rows: list[dict[str, Any]] = []
        for task_id in common_tasks:
            task_a = run_a.tasks[task_id]
            task_b = run_b.tasks[task_id]
            assistant_a = _assistant_message_count(task_a, args.assistant_mode)
            assistant_b = _assistant_message_count(task_b, args.assistant_mode)
            score_a = _final_score(task_id, task_a, run_a.summary_scores)
            score_b = _final_score(task_id, task_b, run_b.summary_scores)
            rows.append(
                {
                    "task_id": task_id,
                    "assistant_messages_a": assistant_a,
                    "assistant_messages_b": assistant_b,
                    "assistant_messages_diff": None
                    if assistant_a is None or assistant_b is None
                    else assistant_b - assistant_a,
                    "final_score_a": score_a,
                    "final_score_b": score_b,
                    "final_score_diff": None
                    if score_a is None or score_b is None
                    else score_b - score_a,
                }
            )
        _write_csv(Path(args.csv_path), rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
