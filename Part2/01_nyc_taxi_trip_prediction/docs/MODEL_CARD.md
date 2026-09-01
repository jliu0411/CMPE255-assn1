# Model Card — RideCast NYC v1.0.0

- **Purpose:** Educational NYC taxi trip-duration estimation.
- **Model:** Histogram Gradient Boosting trained on `log1p(trip_duration)`.
- **Metric:** RMSLE (primary), MAE and R² (secondary).
- **Inputs:** Pickup time, pickup/drop-off coordinates, passengers, vendor, store-forward flag.
- **Intended use:** Demonstrations, coursework, and low-stakes ETA exploration.
- **Not intended for:** Pricing, driver evaluation, safety-critical dispatch, or guarantees.
- **Limitations:** No live traffic, route, weather, event, or road-closure information. The Kaggle source reflects 2016 travel patterns. Synthetic sample metrics are pipeline checks, not real-world performance claims.
- **Privacy:** Exact coordinates can be sensitive. The example service does not persist requests; production telemetry should aggregate or redact location data.
