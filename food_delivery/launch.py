"""One-command startup; preserve the approved model across container restarts."""
import os
from pathlib import Path

import uvicorn

from food_delivery.api import create_app
from food_delivery.pipeline import local_dag


def main():
    out = Path(os.environ.get("ARTIFACT_DIR", "artifacts")).resolve()
    model = out / "model.joblib"
    if not model.exists():
        local_dag(os.environ.get("DATA_PATH", "input/Zomato Dataset.csv"), str(out))
    os.environ["MODEL_PATH"] = str(model)
    os.environ["EVENT_DB"] = str(out / "events.db")
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
