"""Command-line entry point for local pipeline execution."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from apexrisk.config import FeatureConfig, SimulationConfig
from apexrisk.pipeline import run_synthetic_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apexrisk",
        description="Synthetic-first telematics risk-scoring demonstrator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pipeline = subparsers.add_parser(
        "pipeline", help="Generate drivers, vehicles, trips, and behavioural features"
    )
    pipeline.add_argument("--config", default="configs/simulation.yaml")
    pipeline.add_argument("--features-config", default="configs/features.yaml")
    pipeline.add_argument("--output", default="data/generated")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "pipeline":
        simulation_config = SimulationConfig.from_yaml(args.config)
        feature_config = FeatureConfig.from_yaml(args.features_config)
        manifest = run_synthetic_pipeline(
            simulation_config,
            feature_config,
            args.output,
        )
        print(json.dumps(manifest, indent=2))
