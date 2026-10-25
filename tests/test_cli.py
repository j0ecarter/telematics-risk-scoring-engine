"""Exercise telemetry-demo through the real Python command-line entry point."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from apexrisk.cli import build_parser
from apexrisk.telemetry.schema import validate_telemetry_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_CONFIG = PROJECT_ROOT / "configs" / "telemetry.yaml"


def run_cli(*args: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "apexrisk", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("args", [("--help",), ("telemetry-demo", "--help")])
def test_cli_help_exposes_demo_command_and_options(args: tuple[str, ...]) -> None:
    result = run_cli(*args)
    assert result.returncode == 0
    assert "telemetry-demo" in result.stdout
    if len(args) == 2:
        assert "--config" in result.stdout
        assert "--output" in result.stdout
    else:
        assert "pipeline" in result.stdout


def test_demo_cli_defaults_custom_paths_and_repeatability(tmp_path: Path) -> None:
    # A working directory outside the checkout catches accidental hardcoded paths.
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "telemetry.yaml").write_text(TELEMETRY_CONFIG.read_text())
    default_output = tmp_path / "data" / "generated" / "telemetry_demo"
    first = run_cli("telemetry-demo", cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    assert "6/6 scenarios passed." in first.stdout
    assert "Results saved to: data/generated/telemetry_demo/summary.json" in first.stdout
    summary_bytes = (default_output / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    assert summary["all_passed"] is True
    assert len(summary["scenarios"]) == 6
    original_frames = {}
    for scenario in summary["scenarios"]:
        assert f"PASS: {scenario['name']}" in first.stdout
        assert scenario["passed"] is True
        assert scenario["expected"] == scenario["actual"]
        frame = pd.read_parquet(default_output / scenario["telemetry_file"])
        validate_telemetry_frame(frame)
        assert len(frame) == 180
        original_frames[scenario["telemetry_file"]] = frame

    for output in (default_output, tmp_path / "custom output"):
        result = run_cli(
            "telemetry-demo", "--config", str(TELEMETRY_CONFIG), "--output", str(output),
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert "6/6 scenarios passed." in result.stdout
        assert (output / "summary.json").read_bytes() == summary_bytes
        for filename, frame in original_frames.items():
            pd.testing.assert_frame_equal(pd.read_parquet(output / filename), frame)


def test_demo_cli_mismatch_saves_evidence_and_exits_one(tmp_path: Path) -> None:
    payload = yaml.safe_load(TELEMETRY_CONFIG.read_text())
    payload["telemetry"]["speeding_tolerance_kmh"] = 10.0
    config = tmp_path / "changed.yaml"
    config.write_text(yaml.safe_dump(payload))
    output = tmp_path / "results"
    result = run_cli("telemetry-demo", "--config", str(config), "--output", str(output))
    assert result.returncode == 1
    assert "FAIL: speeding" in result.stdout
    assert "speeding_seconds: expected 29, got 0" in result.stdout
    assert "5/6 scenarios passed." in result.stdout
    assert "Traceback" not in result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["all_passed"] is False
    assert len(list(output.glob("*.parquet"))) == 6


@pytest.mark.parametrize(
    "contents",
    [None, "telemetry: [", "- invalid-root", "telemetry: null",
     "telemetry:\n  smoothing_window_seconds: invalid\n"],
    ids=["missing", "malformed-yaml", "invalid-root", "invalid-section", "invalid-value"],
)
def test_demo_cli_config_error_is_useful_and_exits_two(
    tmp_path: Path, contents: str | None
) -> None:
    config = tmp_path / "invalid.yaml"
    if contents is not None:
        config.write_text(contents)
    output = tmp_path / "results"
    result = run_cli("telemetry-demo", "--config", str(config), "--output", str(output))
    assert result.returncode == 2
    assert f"cannot load telemetry config {config}" in result.stderr
    assert "Traceback" not in result.stderr
    assert "PASS:" not in result.stdout
    assert "Results saved" not in result.stdout
    assert not output.exists()


def test_demo_cli_output_error_does_not_report_success(tmp_path: Path) -> None:
    output = tmp_path / "existing-file"
    output.write_text("Keep this content.")
    result = run_cli("telemetry-demo", "--output", str(output))
    assert result.returncode == 2
    assert "telemetry demo could not complete" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Results saved" not in result.stdout
    assert output.read_text() == "Keep this content."


def test_existing_pipeline_arguments_keep_their_defaults() -> None:
    args = build_parser().parse_args(["pipeline"])
    assert args.config == "configs/simulation.yaml"
    assert args.features_config == "configs/features.yaml"
    assert args.claims_config == "configs/claims.yaml"
    assert args.output == "data/generated"
