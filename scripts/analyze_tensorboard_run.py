from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "output"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a training run from TensorBoard event files.")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Run directory containing logs/, config.json, and checkpoints/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="Root directory used with --latest when run_dir is omitted.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Analyze the most recently updated run under --output-root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full analysis as JSON instead of a text report.",
    )
    parser.add_argument(
        "--step-window",
        type=int,
        default=25,
        help="Window size for first/last train-step loss averages.",
    )
    return parser


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()

    output_root = Path(args.output_root).resolve()
    candidates = [path for path in output_root.iterdir() if path.is_dir() and (path / "logs").exists()]
    if not candidates:
        raise FileNotFoundError(f"No run directories with logs/ found under {output_root}")
    if args.latest or args.run_dir is None:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError("No run_dir provided")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_event_files(log_dir: Path) -> list[Path]:
    files = sorted(log_dir.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    return files


def _series_stats(events: list[Any]) -> dict[str, Any]:
    values = [event.value for event in events]
    steps = [event.step for event in events]
    return {
        "count": len(events),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
    }


def _summarize_event_file(path: Path) -> dict[str, Any]:
    accumulator = EventAccumulator(str(path))
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    scalars = {}
    for tag in scalar_tags:
        events = accumulator.Scalars(tag)
        if events:
            scalars[tag] = _series_stats(events)
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
        "scalar_tags": scalar_tags,
        "scalars": scalars,
    }


def _event_scalar_series(path: Path, tag: str) -> list[Any]:
    accumulator = EventAccumulator(str(path))
    accumulator.Reload()
    return accumulator.Scalars(tag)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return fmean(values)


def _step_loss_summary(latest_event: Path, window: int) -> dict[str, Any] | None:
    events = _event_scalar_series(latest_event, "train/loss_step")
    if not events:
        return None
    values = [event.value for event in events]
    steps = [event.step for event in events]
    size = max(1, min(window, len(values)))
    percentiles = []
    for pct in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        idx = max(0, min(len(values) - 1, int(len(values) * pct) - 1))
        percentiles.append({"pct": pct, "step": steps[idx], "loss": values[idx]})
    return {
        "num_points": len(values),
        "window": size,
        "first_window_mean": _safe_mean(values[:size]),
        "last_window_mean": _safe_mean(values[-size:]),
        "best_window_mean": _safe_mean(sorted(values)[:size]),
        "last_logged_step": steps[-1],
        "percentiles": percentiles,
    }


def _epoch_table(latest_event: Path) -> list[dict[str, Any]]:
    val_loss = _event_scalar_series(latest_event, "val/loss")
    val_acc = _event_scalar_series(latest_event, "val/token_accuracy")
    train_loss = _event_scalar_series(latest_event, "train/loss_epoch")
    epoch_sec = _event_scalar_series(latest_event, "timing/epoch_seconds")
    rows = []
    count = min(len(val_loss), len(val_acc), len(train_loss), len(epoch_sec))
    for idx in range(count):
        rows.append(
            {
                "epoch": idx + 1,
                "train_loss": train_loss[idx].value,
                "val_loss": val_loss[idx].value,
                "val_token_accuracy": val_acc[idx].value,
                "epoch_seconds": epoch_sec[idx].value,
            }
        )
    return rows


def _checkpoint_summary(run_dir: Path) -> dict[str, Any]:
    checkpoints_dir = run_dir / "checkpoints"
    result: dict[str, Any] = {"available": False, "checkpoints": {}}
    if not checkpoints_dir.exists():
        return result
    result["available"] = True
    for name in ("best", "last"):
        ckpt_dir = checkpoints_dir / name
        if ckpt_dir.exists():
            result["checkpoints"][name] = {
                "path": str(ckpt_dir),
                "training_state": _load_json(ckpt_dir / "training_state.json"),
                "config_exists": (ckpt_dir / "config.json").exists(),
                "model_exists": (ckpt_dir / "model.pt").exists(),
                "optimizer_exists": (ckpt_dir / "optimizer.pt").exists(),
                "scheduler_exists": (ckpt_dir / "scheduler.pt").exists(),
                "scaler_exists": (ckpt_dir / "scaler.pt").exists(),
            }
    best_state = result["checkpoints"].get("best", {}).get("training_state")
    last_state = result["checkpoints"].get("last", {}).get("training_state")
    if best_state and last_state:
        best_loss = best_state.get("best_val_loss")
        last_loss = last_state.get("best_val_loss")
        if best_loss is not None and last_loss is not None and best_loss < last_loss:
            result["warning"] = (
                "last/training_state.json lags behind best/training_state.json. "
                "This usually means last was saved before best_val_loss was updated."
            )
    return result


def _schedule_summary(
    config: dict[str, Any] | None,
    epoch_rows: list[dict[str, Any]],
    step_loss: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not config or not epoch_rows:
        return None
    if "max_epochs" not in config or "batch_size" not in config:
        return None
    total_planned_steps = None
    warmup_completion_ratio = None
    if step_loss and step_loss.get("last_logged_step") is not None:
        last_logged_step = step_loss["last_logged_step"]
        if config.get("warmup_ratio") is not None and config.get("max_epochs"):
            approx_steps_per_epoch = last_logged_step / len(epoch_rows)
            total_planned_steps = round(approx_steps_per_epoch * config["max_epochs"])
            warmup_steps = round(total_planned_steps * config["warmup_ratio"])
            warmup_completion_ratio = last_logged_step / max(1, warmup_steps)
    return {
        "max_epochs": config.get("max_epochs"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "warmup_steps": config.get("warmup_steps"),
        "warmup_ratio": config.get("warmup_ratio"),
        "freeze_encoder": config.get("freeze_encoder"),
        "density_loss_weight": config.get("density_loss_weight"),
        "approx_total_planned_steps": total_planned_steps,
        "warmup_completion_ratio": warmup_completion_ratio,
    }


def analyze_run(run_dir: Path, step_window: int) -> dict[str, Any]:
    log_dir = run_dir / "logs"
    event_files = _load_event_files(log_dir)
    event_summaries = [_summarize_event_file(path) for path in event_files]
    non_empty_events = [item for item in event_summaries if item["scalars"]]
    latest_non_empty = Path(run_dir / "logs" / non_empty_events[-1]["name"]) if non_empty_events else event_files[-1]
    config = _load_json(run_dir / "config.json")
    training_summary = _load_json(run_dir / "training_summary.json")
    epoch_rows = _epoch_table(latest_non_empty)
    step_loss = _step_loss_summary(latest_non_empty, step_window)
    checkpoints = _checkpoint_summary(run_dir)

    findings = []
    if epoch_rows:
        first = epoch_rows[0]
        last = epoch_rows[-1]
        if last["val_loss"] < first["val_loss"]:
            findings.append(
                f"Validation loss improved from {first['val_loss']:.4f} to {last['val_loss']:.4f} "
                f"over {len(epoch_rows)} epochs."
            )
        if last["val_token_accuracy"] > first["val_token_accuracy"]:
            findings.append(
                f"Validation token accuracy improved from {first['val_token_accuracy']:.4f} "
                f"to {last['val_token_accuracy']:.4f}."
            )
        gap = last["train_loss"] - last["val_loss"]
        findings.append(f"Final train/val loss gap is {gap:.4f}.")
    if training_summary is None:
        findings.append("training_summary.json is missing; the run may have ended before post-training cleanup.")
    if "warning" in checkpoints:
        findings.append(checkpoints["warning"])

    return {
        "run_dir": str(run_dir),
        "config": config,
        "training_summary": training_summary,
        "event_files": event_summaries,
        "latest_event_file": str(latest_non_empty),
        "epoch_rows": epoch_rows,
        "step_loss": step_loss,
        "checkpoints": checkpoints,
        "schedule": _schedule_summary(config, epoch_rows, step_loss),
        "findings": findings,
    }


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_text_report(analysis: dict[str, Any]) -> str:
    lines = []
    lines.append(f"Run: {analysis['run_dir']}")
    lines.append(f"Latest event file: {analysis['latest_event_file']}")
    lines.append("")

    epoch_rows = analysis["epoch_rows"]
    if epoch_rows:
        first = epoch_rows[0]
        last = epoch_rows[-1]
        lines.append("Learning dynamics")
        lines.append(
            f"- val/loss: {_format_float(first['val_loss'])} -> {_format_float(last['val_loss'])} "
            f"over {len(epoch_rows)} epochs"
        )
        lines.append(
            f"- val/token_accuracy: {_format_float(first['val_token_accuracy'])} -> "
            f"{_format_float(last['val_token_accuracy'])}"
        )
        lines.append(
            f"- train/loss_epoch: {_format_float(first['train_loss'])} -> {_format_float(last['train_loss'])}"
        )
        lines.append(
            f"- epoch_seconds: mean {_format_float(_safe_mean([row['epoch_seconds'] for row in epoch_rows]))}"
        )
        lines.append("")

    step_loss = analysis["step_loss"]
    if step_loss:
        lines.append("Step loss")
        lines.append(
            f"- points={step_loss['num_points']} last_logged_step={step_loss['last_logged_step']} "
            f"first_window_mean={_format_float(step_loss['first_window_mean'])} "
            f"last_window_mean={_format_float(step_loss['last_window_mean'])}"
        )
        lines.append(
            f"- best_window_mean={_format_float(step_loss['best_window_mean'])} window={step_loss['window']}"
        )
        lines.append("")

    schedule = analysis["schedule"]
    if schedule:
        lines.append("Config")
        lines.append(
            f"- max_epochs={schedule['max_epochs']} batch_size={schedule['batch_size']} "
            f"lr={schedule['learning_rate']} warmup_steps={schedule['warmup_steps']} "
            f"warmup_ratio={schedule['warmup_ratio']}"
        )
        if schedule.get("approx_total_planned_steps") is not None:
            lines.append(
                f"- approx_total_planned_steps={schedule['approx_total_planned_steps']} "
                f"warmup_completion_ratio={_format_float(schedule['warmup_completion_ratio'])}"
            )
        lines.append("")

    if analysis["findings"]:
        lines.append("Findings")
        for finding in analysis["findings"]:
            lines.append(f"- {finding}")
        lines.append("")

    checkpoints = analysis["checkpoints"]
    if checkpoints.get("available"):
        lines.append("Checkpoints")
        for name, item in checkpoints["checkpoints"].items():
            state = item.get("training_state") or {}
            lines.append(
                f"- {name}: epoch={state.get('epoch')} global_step={state.get('global_step')} "
                f"best_val_loss={state.get('best_val_loss')}"
            )

    return "\n".join(lines).strip()


def main() -> None:
    args = build_parser().parse_args()
    run_dir = _resolve_run_dir(args)
    analysis = analyze_run(run_dir, args.step_window)
    if args.json:
        print(json.dumps(analysis, indent=2))
        return
    print(render_text_report(analysis))


if __name__ == "__main__":
    main()
