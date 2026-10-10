# Complete reusable training and serving pipeline for food delivery ETA

The notebook used session globals for inference preprocessing and had no held-out TabPFN test results. This change puts validation and preprocessing in shared modules, saves the fitted configuration with MLflow models, and adds chronological validation/test evaluation, registry gates, a FastAPI service, HTTP measurements, monitoring, retraining/rollback demonstrations, and a Prefect DAG.

Validation evidence is in `artifacts/`: real dataset evaluation, preprocessing/model reload checks, local HTTP performance, schema rejection, unit/integration tests, and synthetic drift/retraining/rollback. Docker and hosted GitHub Actions still require execution on an available engine and remote repository. TabPFN GPU evaluation remains unexecuted; an explicit continuation notebook is provided without fabricated outputs.

This is a prepared PR description, **not a created pull request**.
