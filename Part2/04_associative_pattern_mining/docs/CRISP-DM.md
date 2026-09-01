# CRISP-DM Report

## 1. Business understanding

The merchandising team wants evidence for complementary recommendations, bundles, email content, and adjacent placement. Success means rules have enough exposure, exceed chance co-occurrence, remain directionally useful on later transactions, and can be tested without harming margin or customer trust. Association is not causation; deployment decisions require experiments.

## 2. Data understanding

The Online Retail dataset contains invoice-line transactions from a UK-based non-store retailer, including invoice, product, quantity, time, unit price, customer, and country. Risks include cancellations marked by invoice prefix, negative return quantities, missing descriptions/customer IDs, service charges presented as products, duplicated lines, dominant popular products, wholesale baskets, and temporal seasonality.

The repository includes a deterministic synthetic sample with planted thematic affinities so every stage runs locally. Its rules demonstrate the pipeline only; use the Kaggle/UCI source for analysis.

## 3. Data preparation

The pipeline normalizes descriptions, removes cancellations, returns, non-positive prices, missing descriptions/dates, duplicates, and common non-product service lines. Each invoice becomes a set of unique product descriptions, preventing quantity from inflating presence. Baskets with fewer than two products cannot contribute pair rules. The last 20% of invoices chronologically form a holdout.

## 4. Modeling

A dependency-light Apriori implementation counts frequent singles and prunes pair/triple candidates by downward closure. Candidate items are capped by frequency for laptop memory safety. Rules are generated in both valid directions and filtered by minimum support, confidence, and lift. Leverage measures absolute departure from independence; conviction adds directional context.

## 5. Evaluation

Review support (business reach), confidence (conditional reliability), lift (improvement over popularity), leverage (absolute impact), conviction, and consequent popularity together. Holdout confidence is calculated only on later baskets; low eligible counts are labeled unstable. Confidence drift beyond 0.20 triggers observation rather than promotion. Seasonal and country slices should be evaluated on real data.

## 6. Deployment

FastAPI loads the immutable JSON artifact and returns ranked complements when a rule antecedent is contained in the incoming basket. Ranking combines confidence, lift, and support. Production should add inventory, margin, exclusions, diversity, and frequency caps; monitor coverage, click/add-to-cart lift, revenue per session, returns, latency, drift, and rule staleness. Validate changes through randomized A/B tests.
