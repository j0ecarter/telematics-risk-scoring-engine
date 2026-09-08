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
| `speed_kmh` | float32 | km/h | **Observed** ground speed as the device reports it, including measurement noise. Non-negative. |
| `heading_degrees` | float32 | degrees | **Observed** compass heading in `[0, 360)`. Required for cornering detection. |
| `is_gps_valid` | bool | - | `False` marks a dropped or degraded fix. Speed and heading are undefined for those samples. |

Speeds are expressed in km/h to match the existing trip summary columns
(`average_speed_kmh`, `maximum_speed_kmh`). Conversion to m/s happens inside the
detector and nowhere else, so that a unit error cannot cross a module boundary.

### Why `second_index` is uint32 and not uint16

The current journey generator clips trip distance at 180 km and clips average speed at a
lower bound of 8 km/h, so its worst-case trip duration is `180 / 8 * 3600 = 81,000`
seconds - about 22.5 hours. That overflows uint16 (65,535).

The probability of drawing such a trip is negligible, but the type must still be safe.
It also points at a modelling problem worth fixing separately: **a 22-hour continuous
trip is not a trip.** Consider capping trip duration in the generator, or splitting long
journeys into legs with a rest gap between them. Logged as an open item, not part of
this contract.

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

The true trace feeds claim generation and is then discarded. Only the observed trace is
written to disk or passed to the detector.

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

For each trip, given its observed trace and its summary row:

1. **Length.** `len(trace) == duration_seconds`.
2. **Distance reconciliation.** `sum(speed_kmh) / 3600 ≈ distance_km`, within 0.5%.
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

---

## 7. Deliberately left open

These require research and are not settled by this document:

- **Noise magnitude** for GPS speed and heading.
- **Dropout rate and duration** - how often `is_gps_valid` is False, and for how long.
- **Detection thresholds** for each severity band, and the speeding tolerance. Values
  around 0.3g are commonly quoted for harsh braking but vary widely between vendors;
  these must be researched, cited in `configs/telemetry.yaml`, and treated as versioned
  assumptions rather than constants buried in code.
