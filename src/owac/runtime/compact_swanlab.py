"""Keep compact-policy training and completed rollouts in the same SwanLab run."""

import argparse
import json
import re
from pathlib import Path


def init_run(cfg, logdir, *, resume="allow"):
    import swanlab

    swanlab.login()
    swanlab.merge_settings(swanlab.Settings(probe=swanlab.Settings.Probe(monitor=False)))
    swanlab.init(
        project="OWAC",
        workspace=cfg["swanlab"].get("workspace"),
        id=cfg["swanlab"].get("id"),
        resume=resume,
        experiment_name=cfg["swanlab"]["name"],
        group=cfg["swanlab"]["group"],
        job_type=cfg["swanlab"].get("job_type", "raw-proprio-language-flow"),
        config=cfg,
        logdir=str(logdir),
    )
    for prefix, label in [
        ("objective", "01 训练目标"),
        ("validation", "02 动作验证"),
        ("optimizer", "03 优化器"),
        ("runtime", "04 运行资源"),
        ("success", "05 闭环评估"),
    ]:
        swanlab.define_metric(f"{prefix}/*", section_name=label)
    return swanlab


def result_step(result):
    if "step" in result:
        return int(result["step"])
    # Older summaries did not store the step, but used this checkpoint contract.
    match = re.fullmatch(r"step-(\d+)\.pt", Path(result["checkpoint"]).name)
    if match is None:
        raise ValueError("Evaluation summary needs the checkpoint's training step")
    return int(match[1])


def result_metrics(cfg, result, output):
    if result["native_exit"] != 0 or result["episodes"] <= 0:
        raise ValueError("Only completed native evaluations can enter the validation curve")
    kind = result.get("evaluation_kind")
    if kind is None:
        # Compatibility with already completed evaluations and the running controller.
        if output.name == "independent-final":
            kind = "independent-final"
        elif result["episodes"] == cfg["research"]["selection_episodes"]:
            kind = "development"
        else:
            kind = "screening"
    prefix = "success" if kind == "development" else f"success/{kind}"
    metrics = {
        f"{prefix}/{key}": value["successes"] / value["episodes"]
        for key, value in result["groups"].items()
    }
    metrics[f"{prefix}/overall"] = result["successes"] / result["episodes"]
    metrics[f"{prefix}/episodes"] = result["episodes"]
    return result_step(result), metrics


def log_results(sdk, cfg, evaluations, *, force=False):
    """Log in checkpoint order; return receipts to persist after SDK upload finishes."""
    pending = []
    for output, result in sorted(evaluations, key=lambda item: result_step(item[1])):
        step, metrics = result_metrics(cfg, result, output)
        receipt = {"run_id": cfg["swanlab"]["id"], "step": step, "metrics": metrics}
        path = output / "swanlab-training-link.json"
        if not force and path.exists() and json.loads(path.read_text()) == receipt:
            continue
        sdk.log(metrics, step=step)
        pending.append((path, receipt))
    return pending


def completed_results(root):
    paths = list(root.glob("development-*/summary.json"))
    final = root / "independent-final/summary.json"
    if final.exists():
        paths.append(final)
    return [(path.parent, json.loads(path.read_text())) for path in paths]


def save_receipts(pending):
    for path, receipt in pending:
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))


def publish_results(cfg, evaluations, logdir, *, force=False):
    sdk = init_run(cfg, logdir, resume="must")
    pending = log_results(sdk, cfg, evaluations, force=force)
    sdk.finish()
    save_receipts(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Experiment root")
    parser.add_argument(
        "--force", action="store_true", help="Replay even with local upload receipts"
    )
    args = parser.parse_args()
    publish_results(
        json.loads(args.config.read_text()),
        completed_results(args.output),
        args.output / "swanlab-closed-loop-backfill",
        force=args.force,
    )


if __name__ == "__main__":
    main()
