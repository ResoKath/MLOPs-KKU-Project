# Complete reproducible food delivery training, serving and monitoring

The notebook depended on session variables for inference and lacked a reproducible path from raw data to an approved API model. This change adds shared validation/preprocessing, chronological splits, three baseline experiments, full held-out evaluation, MLflow provenance/config/model reload checks, versioned registry gates, a FastAPI service, a repeatable DAG, and Docker/Prometheus/GitHub Actions workflows. Existing group files and their history are retained.

The CPU serving candidate is HGB (test MAE 3.4018 minutes). A separate, actually executed TabPFN CUDA experiment evaluated all 5,929 test rows (MAE 3.1479 minutes); its foundation checkpoint exceeds the current 100 MB serving gate, so it remains a quality-passed candidate. No private credentials or foundation checkpoint are included in Git.

Promotion evidence is bound to the candidate checksum, training run and registry version. CI recomputes predictions and can deliberately reject degraded predictions; monitoring separates model versions and consumes non-overlapping label windows. Container startup preserves the approved model for restart/rollback.

Validation so far: 19 tests passed on Windows, Docker built and served externally through the host port, and Prometheus scraped the live endpoint. Hosted CI and final container latency/alert/rollback results are recorded separately as they finish. Synthetic feedback and retraining demonstrations must be identified as simulations; real stakeholder confirmation and business pilot measurements remain group work.
