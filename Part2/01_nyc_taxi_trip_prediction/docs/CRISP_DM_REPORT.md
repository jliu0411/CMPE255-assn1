# CRISP-DM Project Report

## 1. Business understanding

RideCast predicts point-to-point NYC taxi duration at booking time. Riders gain clearer arrival expectations; dispatch and operations teams can improve ETAs and planning. The primary success metric is RMSLE, matching Kaggle and limiting the influence of very long trips. Secondary measures are MAE and R². Predictions are advisory, not guarantees.

## 2. Data understanding

The competition data is sampled from 2016 NYC Yellow Cab records. Inputs available at pickup include vendor, timestamp, passenger count, pickup/drop-off coordinates, and store-and-forward flag. The target is trip duration in seconds. Risks include GPS errors, impossible passenger counts, extreme durations, temporal drift, and incomplete traffic/weather context.

The repository includes a deterministic 12,000-row synthetic sample so every stage runs without credentials. Final performance claims must be generated from Kaggle data, not the demo sample.

## 3. Data preparation

Quality rules retain durations from 30 seconds to 6 hours, 1–8 passengers, and coordinates within a broad NYC bounding box. Features include hour, weekday, month, weekend/rush-hour flags, great-circle distance, Manhattan distance, bearing encoding, coordinates, vendor, passenger count, and store-forward status. The target is transformed with `log1p`.

## 4. Modeling

A median regressor provides a mandatory baseline. The candidate is a histogram gradient-boosting regressor, chosen for nonlinear tabular performance, fast CPU training, and minimal deployment dependencies. The last 20% of trips chronologically form validation, reducing leakage compared with a random split.

## 5. Evaluation

Training writes `artifacts/metrics.json` with reproducible RMSLE, MAE, R², row counts, feature names, and timestamp. Promote only when the candidate beats the baseline on RMSLE and service smoke tests pass. Slice monitoring should cover rush hour, distance bands, weekday/weekend, and pickup month.

## 6. Deployment

FastAPI loads a versioned joblib artifact and exposes health, metadata, prediction, and OpenAPI endpoints. Pydantic validates the service boundary. The frontend communicates uncertainty with a likely range. Production work should add authentication, request logging without precise-coordinate retention, containerization, a model registry, drift alerts, and scheduled retraining.
