"""Command-line entry point for the synthetic pipeline and telemetry demo."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import yaml

from apexrisk.config import ClaimsConfig, EventDetectionConfig, FeatureConfig, SimulationConfig
from apexrisk.pipeline import run_synthetic_pipeline
from apexrisk.telemetry.demo import run_telemetry_demo


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
    pipeline.add_argument("--claims-config", default="configs/claims.yaml")
    pipeline.add_argument("--output", default="data/generated")
    demo = subparsers.add_parser(
        "telemetry-demo", help="Run telemetry scenarios and save comparison results"
    )
    demo.add_argument(
        "--config", default="configs/telemetry.yaml", help="Path to the telemetry configuration"
    )
    demo.add_argument(
        "--output",
        default="data/generated/telemetry_demo",
        help="Directory for telemetry files and summary.json (named files are replaced)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pipeline":
        simulation_config = SimulationConfig.from_yaml(args.config)
        feature_config = FeatureConfig.from_yaml(args.features_config)
        claims_config = ClaimsConfig.from_yaml(args.claims_config)
        manifest = run_synthetic_pipeline(
            simulation_config,
            feature_config,
            args.output,
            claims_config=claims_config,
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "telemetry-demo":
        try:
            config = EventDetectionConfig.from_yaml(args.config)
        except (OSError, ValueError, TypeError, OverflowError, yaml.YAMLError) as exc:
            parser.exit(2, f"apexrisk: error: cannot load telemetry config {args.config}: {exc}\n")
        try:
            summary = run_telemetry_demo(config, args.output)
        except (OSError, ValueError) as exc:
            parser.exit(2, f"apexrisk: error: telemetry demo could not complete: {exc}\n")

        for scenario in summary["scenarios"]:
            status = "PASS" if scenario["passed"] else "FAIL"
            print(f"{status}: {scenario['name']}")
            for field, values in scenario["mismatches"].items():
                print(f"  {field}: expected {values['expected']}, got {values['actual']}")
        passed = sum(scenario["passed"] for scenario in summary["scenarios"])
        print(f"{passed}/{len(summary['scenarios'])} scenarios passed.")
        print(f"Results saved to: {Path(args.output) / 'summary.json'}")
        if not summary["all_passed"]:
            raise SystemExit(1)
