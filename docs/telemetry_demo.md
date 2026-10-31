# Deterministic telemetry demo

`apexrisk.telemetry.demo.build_demo_scenarios()` now creates six deterministic,
three-minute traces: steady driving, braking, acceleration, cornering, speeding
and GPS dropout. Each scenario includes its name, description, speed limit and a
validated telemetry table. The generator does not write files or run detection.
Six integration checks run these traces through the detector using
`configs/telemetry.yaml` and compare every result field with independently specified
expectations: one harsh event for each manoeuvre, 29 speeding seconds for the
speeding trace, and zero events or speeding for steady driving and GPS dropout.
All other fields must be zero. Changes to the configuration require explicit review
of these expectations; detector unit tests continue to use independent settings.

Run the complete demo from the repository root after installing the project and
activating the virtual environment:

```bash
python -m apexrisk telemetry-demo \
  --config configs/telemetry.yaml \
  --output data/generated/telemetry_demo
```

These paths are the defaults, so `python -m apexrisk telemetry-demo` does the same
thing. The installed `apexrisk telemetry-demo` entry point also works. Paths are
relative to your current directory unless absolute paths are supplied.
Use `python -m apexrisk telemetry-demo --help` to see the options.

With the supplied configuration, the output is:

```text
PASS: steady
PASS: braking
PASS: acceleration
PASS: cornering
PASS: speeding
PASS: gps_dropout
6/6 scenarios passed.
Results saved to: data/generated/telemetry_demo/summary.json
```

The exit code is `0` when all comparisons pass, `1` when a comparison fails, and
`2` for command usage, configuration or file errors. Failed comparisons still save
their evidence and print each mismatched field. For example, using a separate
configuration with `speeding_tolerance_kmh: 10.0` reports:

```text
FAIL: speeding
  speeding_seconds: expected 29, got 0
```

This is a disagreement with the fixed baseline expectations, not necessarily a
detector defect. Invalid or missing configuration is reported with its path and
an error message rather than a traceback.

The same workflow is available directly from Python:

```python
from apexrisk.config import EventDetectionConfig
from apexrisk.telemetry.demo import run_telemetry_demo

config = EventDetectionConfig.from_yaml("configs/telemetry.yaml")
summary = run_telemetry_demo(config, "data/generated/telemetry_demo")
print(summary["all_passed"])
```

This writes six observed-telemetry Parquet files (`steady.parquet`,
`braking.parquet`, `acceleration.parquet`, `cornering.parquet`, `speeding.parquet`
and `gps_dropout.parquet`) and `summary.json`. The summary records detector
settings, expected and actual results, field-level mismatches, GPS valid time and
distance, GPS coverage, and individual and overall pass/fail status. Missing GPS
readings remain `NaN` in Parquet; no hidden truth channels are saved.

Repeated runs replace these seven files, leaving unrelated files alone. JSON uses
relative filenames and no timestamp, so identical settings produce identical
summaries and telemetry values across runs and output directories. The generated
files are ignored by Git. Changed detector settings are compared with the same
fixed expectations; mismatches are saved with `all_passed: false` rather than
silently changing the expected answers.

The saved-output and CLI tests cover Parquet round trips, repeatability across
runs and directories, deliberate comparison failure, help and default options,
invalid/missing configuration, and output-file errors. Run these checks with:

```bash
python -m pytest tests/test_telemetry_demo.py tests/test_cli.py
```

## Baseline results and limits

Each scenario contains 180 samples at 1 Hz. The supplied detector settings and
fixed expected results are checked together; all unlisted event counts are zero.

| Scenario | Expected detector result |
|---|---|
| Steady driving | No events; no speeding seconds |
| Braking | One harsh braking event |
| Acceleration | One harsh acceleration event |
| Cornering | One harsh cornering event |
| Speeding | 29 speeding seconds; no manoeuvre events |
| GPS dropout | No events across the missing readings; no speeding seconds |

The demo uses fixed profiles without random measurement noise. The GPS-dropout
scenario has five invalid samples, retained as rows with missing speed and heading.
Noise robustness is exercised separately by detector tests. The unused
`telemetry_simulation` settings describe a possible future population generator;
they do not control these demo scenarios.

These checks show agreement with controlled examples. They do not measure accident
prediction, real-world detector precision or recall, or insurance pricing quality.
Steady driving with no detected events does not establish that a journey is safe.
At 1 Hz, short events can be missed and smoothing can reduce peaks. GPS gaps remove
observations; the detector must not infer an event through them.

For the current finish line and deferred work, see [Project scope](scope.md).
