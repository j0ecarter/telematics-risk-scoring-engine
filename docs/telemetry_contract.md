# Telemetry data contract

Status: **frozen for v1**. Changes to this document require a corresponding change to
`src/apexrisk/telemetry/schema.py` and to the tests that enforce it.

This document fixes the interface between three things that are built separately:

1. the **trace generator**, which produces a second-by-second driving signal per trip;
2. the **event detector**, which reads that signal and counts harsh driving events; and
3. the **trip summary rows** in `trips.parquet`, whose schema must not change.

Freezing the interface first means the generator and the detector can be written and
tested independently, and it means the existing feature and claims code keeps working
without modification.

---

## 1. Sampling rate

**Decision: 1 Hz (one sample per second).**

A typical insurance telematics dongle fixes GPS at about 1 Hz while sampling its
accelerometer at 10-50 Hz, and most commercial vendors detect harsh events from the
high-rate accelerometer channel rather than from GPS. This project detects events from
1 Hz GPS-derived speed, which is a real approach used by several usage-based insurance
programmes, but it carries a limitation that must be stated rather than hidden:

> A harsh braking event lasting roughly 1.5 seconds is represented by only one or two
> samples. Event *duration* is therefore poorly resolved, short events may be missed
> entirely, and detected event rates are sensitive to the smoothing window. Any figure
> this project produces for events per 100 km is specific to a 1 Hz detector and is not
> comparable with a vendor figure derived from accelerometer data.

This limitation belongs in the README alongside the existing methodological boundaries.

Raising the rate later is possible but multiplies data volume linearly, and the volume
argument in section 5 is already the binding constraint. 1 Hz stays.

---

## 2. Published telemetry columns

| Column | Type | Units | Description |
|---|---|---|---|
| `trip_id` | string | - | Foreign key to `trips.trip_id`. |
| `second_index` | uint32 | s | 0-based offset from trip start. Contiguous, no gaps. |
| `speed_kmh` | float32 | km/h | **Observed** ground speed, including measurement noise. Finite and non-negative when GPS is valid; `NaN` otherwise. |
| `heading_degrees` | float32 | degrees | **Observed** compass heading in `[0, 360)` when GPS is valid; `NaN` otherwise. |
| `is_gps_valid` | bool | - | `False` marks a dropped or degraded fix and requires `NaN` speed and heading. |

Speeds are expressed in km/h to match the existing trip summary columns
(`average_speed_kmh`, `maximum_speed_kmh`). Conversion to m/s happens inside the
detector and nowhere else, so that a unit error cannot cross a module boundary.

### Invalid GPS representation

Every second remains present and `second_index` stays contiguous. An invalid sample stores
`NaN` for both observed channels rather than zero or a carried-forward value. This makes it
impossible to mistake a placeholder for a genuine stationary or straight-driving sample.
The detector splits on `is_gps_valid` before smoothing or differencing, so no calculated
signal or merged event may cross an invalid interval.

### Journey duration and `second_index`

A generated trip represents one uninterrupted driving leg and is limited to 3 minutes
through 4 hours. If the initially calculated duration falls outside that range, duration is
bounded and average speed is recomputed from distance divided by duration so the summary
remains internally coherent. The existing `uint32` index is retained as a stable, generous
representation that leaves room for future contract revisions and higher sampling rates.

### Deliberately excluded columns

**Timestamp.** Derivable as `trips.start_timestamp_utc + second_index` seconds. Storing
it would cost 8 bytes per row for zero additional information.

**Longitudinal and lateral acceleration.** These are *derived quantities produced by the
detector*, not raw sensor channels. Publishing them would conflate the raw signal with
the detector's intermediate output, and would make event detection a trivial threshold
comparison - removing the entire point of the exercise.

**Latitude and longitude.** Not required by any current or planned feature, and costly
in bytes. Revisit only if road-type inference, map matching or geofencing is added.

---

## 3. Observed signal versus simulation truth

**Decision: only the observed (noisy) signal is ever persisted.**

The generator produces two channels internally:

- a **true** kinematic trace - what the vehicle actually did; and
- an **observed** trace - the true trace plus measurement noise and dropouts.

The true trace feeds claim generation and exact trip-summary reconciliation, then is
discarded. Only the observed trace is written to disk or passed to the detector.

This mirrors the policy already applied to latent driver traits, and it is what makes
the published features an honest *noisy measurement* of behaviour rather than a
relabelling of the simulator's hidden ground truth. Record it in the manifest as
`raw_truth_channels_published: false`, next to the existing
`private_latent_traits_published` flag.

---

## 4. Detector output contract

The detector consumes a trace and returns counts. It must produce the three columns
`trips.parquet` already publishes, so that `behavioural.py` and `claims.py` need no
changes:

- `hard_brake_count`
- `hard_acceleration_count`
- `harsh_cornering_count`

### Severity bands

**Decision: detect three exclusive severity bands per event type.** Commercial telematics
providers band events rather than treating them as binary, it costs roughly three extra
comparisons in the detector, and it yields richer features for modelling later.

Bands are **mutually exclusive** - an event is counted once, in its highest band only.
Overlapping (cumulative) bands would produce collinear features and make later
coefficients hard to interpret.

New columns, added alongside the existing ones:

```
brake_events_moderate    brake_events_harsh    brake_events_severe
accel_events_moderate    accel_events_harsh    accel_events_severe
corner_events_moderate   corner_events_harsh   corner_events_severe
```

The existing columns are then defined as:

```
hard_brake_count = brake_events_harsh + brake_events_severe
```

and likewise for acceleration and cornering. That is, the existing column keeps its
current meaning - "events at or above the harsh threshold" - and the moderate band is
purely additive new information.

**Adding columns is safe.** `behavioural.REQUIRED_TRIP_COLUMNS` is checked as a subset,
so extra columns pass validation untouched. Renaming or removing a column is not safe.

### Speeding is now an estimate, not ground truth

This is the subtle consequence of moving to detection, and it needs to be understood
before the generator is written.

`speeding_duration_seconds` is currently exact - the simulator decides it directly. Once
it is derived by counting seconds where observed speed exceeds the limit, it becomes a
**biased estimator**. For a driver travelling at exactly the limit, symmetric noise puts
roughly half the samples above it, so the estimate tends toward 50% of the trip rather
than 0%. The bias is largest precisely where most drivers actually sit.

Real telematics programmes handle this with a tolerance band, only counting speed above
`limit + tolerance`. The detector must do the same, with the tolerance in
`configs/telemetry.yaml`, and the README must state that the published speeding figure
is a tolerance-adjusted estimate rather than a true count.

---

## 5. Persistence policy

**Decision: full-population telemetry is never materialised.**

At the population scale needed for credible frequency and severity models (roughly
3,000 drivers over 18 months, about 970,000 trips at a mean duration near 1,000 seconds)
a complete telemetry table would run to around a billion rows. That is not workable on a
laptop and it is not necessary.

Instead, per trip: generate the trace as numpy arrays, run the detector, emit the trip
summary row, discard the trace. Only aggregates persist.

This is also the architecturally honest choice. In production telematics the per-second
signal is processed on the device and only trip summaries reach the insurer, so a
pipeline shaped this way mirrors the real system rather than an idealised one.

### Sample cohort

A deterministic sample of trips retains its full trace, written to
`data/generated/telemetry_sample.parquet`, for validation, plots and documentation.

- Default size: **2,000 trips** (roughly 2.1M rows, comfortably under 50 MB on disk).
- Selection must be **deterministic given the simulation seed** so that runs stay
  reproducible - for example, every trip whose sequence number is divisible by a stride
  chosen to yield the target count. Do not select by unseeded random draw.
- `data/generated/` is gitignored, so the sample never enters the repository.

---

## 6. Invariants

These are the properties a generated trace must satisfy for the trip summary derived
from it to be truthful. They are the acceptance criteria for the generator and should be
enforced by tests, not by inspection.

For each trip, the internal true trace and summary row must satisfy:

1. **Length.** `len(trace) == duration_seconds`.
2. **Distance reconciliation.** `sum(true_speed_kmh) / 3600 ≈ distance_km`, within 0.5%.
   At 1 Hz the distance is the integral of speed, so this is the single strongest check
   that generator and summary agree.
3. **Peak speed.** `max(speed_kmh) ≈ maximum_speed_kmh`, within rounding tolerance.
4. **Mean speed** follows from 1 and 2 and needs no separate assertion, since
   `average_speed_kmh` is defined as distance over duration.
5. **Non-negative speed.** No sample below zero, noise included.
6. **Heading range.** All headings in `[0, 360)`.
7. **Bounded yaw rate.** The wrapped second-to-second heading change must stay within a
   physically plausible limit. Lateral acceleration is approximately `v x yaw_rate`, so
   an unbounded heading change at motorway speed implies an impossible cornering force
   and will manufacture phantom severe events.

Heading differences must be computed **wrapped**: the change from 359 degrees to 1 degree
is +2 degrees, not -358. Naive subtraction here is a reliable source of spurious
cornering events.

The published observed trace has a different contract because it contains noise and missing
fixes:

1. Valid samples have finite speed and heading; invalid samples have `NaN` for both.
2. `gps_valid_seconds` is the count of valid samples.
3. `gps_valid_distance_km` is `sum(valid observed speed) / 3600` at 1 Hz.
4. `gps_coverage_fraction` is valid samples divided by total samples.
5. Speeding fractions use valid seconds; detected-event rates use valid observed distance.
6. If valid time or distance is zero, the corresponding behavioural rate is missing, not
   zero. No observation must not be interpreted as safe behaviour.

---

## 7. Versioned assumptions

Noise, dropout, detection thresholds and timing rules are recorded in
`configs/telemetry.yaml` and explained in `docs/telemetry_assumptions.md`. They are project
assumptions to test through sensitivity analysis, not validated hardware or actuarial
calibration. Changing them requires a configuration version change and regression tests.
