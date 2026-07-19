#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from legumegenomefm.training import TrainConfig, run_training


def load_config(path: Path) -> TrainConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    project_root = (path.parent / payload.pop("project_root")).resolve()
    payload["project_root"] = project_root
    payload["dataset_manifest"] = (project_root / payload["dataset_manifest"]).resolve()
    payload["output_dir"] = (project_root / payload["output_dir"]).resolve()
    return TrainConfig(**payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the frozen LegumeGenomeFM model")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    result = run_training(
        config,
        device_override=args.device,
        resume=args.resume,
        initialize_checkpoint=args.initialize_from,
    )
    print(json.dumps({"checkpoint": str(result.checkpoint_dir), "loss": result.loss, "step": result.step, "tokens_seen": result.tokens_seen}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
