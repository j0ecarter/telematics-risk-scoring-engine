# Telemetry assumptions, version 1

Recorded 26 September 2026. Scope: synthetic passenger cars sampled at 1 Hz.
The settings are in `configs/telemetry.yaml`. They are a documented starting
point for implementation, not a validated detector or evidence of real-world
insurance risk. Schema validation, preprocessing, event counting and deterministic
demo checks are implemented. Population trace generation and pipeline integration
are deferred extensions; see [Project scope](scope.md). The fixed demo uses no
random measurement noise and does not load the GPS-error settings below.

## Evidence and its limits

1. [Geotab Rules overview](https://support.geotab.com/help/mygeotab/groups-and-rules/rules/rules-overview),
   GPS-based safety rules, Passenger row: harsh acceleration and braking are
   0.300g; cornering is 0.420g. We use these as reference magnitudes, not a claim
   to reproduce Geotab's sampling, filtering or event counts. The simulator
   generates hatchbacks, saloons, SUVs and estates; using one passenger class
   for all is a v1 simplification, not a vehicle-specific calibration.
2. [GPS.gov GPS Accuracy](https://www.gps.gov/gps-accuracy) explains that receiver
   speed accuracy depends on local conditions and hardware. Satellite-service
   accuracy is not a receiver's observed error distribution. This motivates
   noise/dropout scenarios, but supplies no universal numerical rate for them.
3. [u-blox MAX-M10S data sheet](https://content.u-blox.com/sites/default/files/MAX-M10S_DataSheet_UBX-20035208.pdf),
   page 4, lists velocity accuracy 0.05 m/s and heading accuracy 0.3 degrees,
   qualified as 50% at 30 m/s during dynamic operation. These are not Gaussian
   standard deviations or low-speed guarantees. Our simulation values below
   are deliberately separate project assumptions, not conversions of that spec.

## Detector choices

| Setting | Value | Reason and trade-off |
|---|---:|---|
| Centred median window | 3 samples / seconds at 1 Hz | Removes isolated spikes; a wider window risks losing more short events. Covered by detector tests. |
| Trim per valid-segment end | 2 samples | One neighbour for the median plus one for the central difference requires a five-sample raw neighbourhood. |
| Minimum event duration | 2 consecutive qualifying samples | Rejects isolated crossings; also misses some real brief events. Sample count is a duration approximation. |
| Minimum separation | 3 below-threshold samples | Merge gaps of 0-2 samples; a gap of 3 is separate. Prevents brief dips splitting a manoeuvre, but can merge distinct actions. |
| Cornering speed gate | 20 km/h | Suppresses low-speed manoeuvres and unreliable heading interpretation. Exactly 20 is eligible. Some genuine low-speed events are excluded. |
| Speeding tolerance | 5 km/h | Noise buffer; count only values strictly above limit + tolerance. Not a legal allowance. Smaller excesses are intentionally omitted. |

The four previously selected settings (3, 2, 2, 3) are retained. The speed gate
and tolerance are engineering choices, not published empirical estimates.
Never calculate a derivative or merge an event across an invalid GPS gap.
For the current centred speed method, a valid segment shorter than five samples
has no usable interior acceleration. Two interior samples require six raw
samples, though those samples must also qualify to count as an event. The
preprocessing, event qualification and merging are implemented and tested. Edge trimming
for acceleration does not apply to speeding exposure, whose denominator is valid
GPS seconds.

### Severity bands

Convert g to m/s² using 9.81. Store positive magnitudes, including for braking.
The extra decimal places record the conversion, not calibration precision.

| Event | Moderate | Harsh | Severe |
|---|---:|---:|---:|
| Braking | 1.962 (0.20g) | 2.943 (0.30g) | 4.905 (0.50g) |
| Acceleration | 1.962 (0.20g) | 2.943 (0.30g) | 4.905 (0.50g) |
| Cornering | 2.943 (0.30g) | 4.120 (0.42g, rounded) | 5.886 (0.60g) |

Only the harsh reference levels come from Geotab. Moderate and severe levels
are project choices that bracket those levels with separated bands. Equal
braking/acceleration bands simplify v1; they do not imply equal accident risk.
A run qualifies at or above the moderate magnitude for the configured duration.
After qualifying runs are merged within a valid segment, classify each event
once by its peak magnitude: moderate below harsh, harsh below severe, severe
at or above severe. A short peak can therefore raise an otherwise qualifying
event's band. Preserve `hard_*_count` as harsh + severe, per the existing contract.
These bands are implemented by the detector and covered by its tests; the values
remain engineering assumptions rather than calibrated accident-risk categories.

## GPS-error scenario for an optional future population generator

These are separate `telemetry_simulation` design settings. The existing
`EventDetectionConfig` loads only `telemetry`; the population generator and its
settings loader have not been implemented. These settings affect neither the
current pipeline nor the deterministic demo, which has a fixed dropout interval.

- Speed noise: independent zero-mean Gaussian draws, standard deviation
  **1.8 km/h** (0.5 m/s), before clipping observed speed at zero.
- Heading noise: independent zero-mean Gaussian draws, standard deviation
  **1 degree**, before wrapping headings into [0, 360).
- Dropouts: at each sample outside an existing dropout, probability **0.002**
  of beginning one at that sample. Duration is an inclusive discrete-uniform
  draw from **1 to 5 seconds**. Clip at the trip end; do not start overlapping
  dropouts. This yields approximately **0.6% invalid samples** on long traces:
  with mean duration 3, the fraction is p*3 / (1-p + p*3).
- Use the simulation's seeded random generator so repeated runs are reproducible.

The noise is large enough to exercise preprocessing while the dropout scenario
keeps most of a trip observable. Neither is fitted to a real device or fleet.
Independent Gaussian errors omit correlated drift, rare extreme spikes and
road-dependent reception; dedicated robustness traces must cover those cases.
Clipping speed creates a positive bias near rest. Heading noise at rest is not
modelled realistically by this simple scenario. A speed gate helps detection,
but does not make the underlying heading estimate accurate.

This note supplies the initial numerical choices referenced by section 7 of
`telemetry_contract.md`. The schema fixes the surrounding behaviour:
invalid observations use `NaN` while their rows remain present, the internal
true trace reconciles total distance, observed valid time/distance supply the
detector denominators, and one uninterrupted synthetic trip is capped at four
hours. Those decisions are enforced by the schema and journey tests.

## Validation and later sensitivity checks

The detector configuration loads, each numerical choice has a source or explicit
rationale, and 1 Hz limitations are documented. Passing configuration validation
proves admissible settings, not detection quality. The deterministic demo checks
known traces against fixed expectations; it is not the broader sensitivity study
outlined below.

For an optional sensitivity study, use known traces and vary one setting at a time:

- Median window 1 / 3 / 5, with derivative trim recomputed as window//2 + 1.
- Minimum duration 1 / 2 / 3 samples, separation 1 / 3 / 5, speed gate 10 / 20 / 30 km/h.
- Speeding tolerance 3 / 5 / 7 km/h and all severity thresholds scaled by 0.8 / 1 / 1.2.
- Speed-noise standard deviation 0 / 1.8 / 3.6 km/h and heading 0 / 1 / 3 degrees.
- Dropout start probability 0 / 0.002 / 0.01; also test explicit longer outage blocks.

Measure false detections on steady traces, missed/merged injected events and
sensitivity of speeding estimates. Do not tune synthetic outcomes to claim real
predictive accuracy. No event-rate-per-distance calibration target is assigned
until a compatible benchmark is available.

At 1 Hz, short events can fall between samples. Median smoothing and central
differencing can reduce peaks, and two qualifying samples do not precisely
measure physical event duration. These event rates cannot be equated to rates
from a higher-frequency accelerometer or a different detection algorithm.
