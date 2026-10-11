"""Same three checks locally and in hosted CI; emit machine-readable evidence."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from food_delivery.preprocessing import prepare_data
from food_delivery.train import write_json


def checks(data="input/Zomato Dataset.csv",out="artifacts", inject=None):
    stages = {}
    try:
        subprocess.run([sys.executable,"-m","ruff","check","food_delivery","tests"],check=True)
        stages["code_quality"] = "PASSED"
        if inject == "data":
            path = Path(out)/"bad_ci_data.csv"
            frame = pd.read_csv(data)
            frame.loc[0,"Vehicle_condition"] = 9
            frame.to_csv(path,index=False)
            data = str(path)
        prepare_data(data)
        stages["data_validation"] = "PASSED"
        quality = json.loads((Path(out)/"gate.json").read_text())
        if inject == "model":
            quality["passed"] = False
        if not quality["passed"]:
            raise ValueError("Model quality gate failed")
        stages["model_quality"] = "PASSED"
    except Exception as error:
        result = {"passed":False,"checks":stages,"error":str(error),"injection":inject,"execution":"local CLI; hosted CI needs GitHub run evidence"}
        write_json(Path(out)/f"ci_{inject or 'normal'}.json",result)
        raise
    result = {"passed":True,"checks":stages,"execution":"local CLI; hosted CI needs GitHub run evidence"}
    write_json(Path(out)/"ci_normal.json",result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inject",choices=["data","model"])
    p.add_argument("--data",default="input/Zomato Dataset.csv")
    a = p.parse_args()
    print(json.dumps(checks(a.data,inject=a.inject),indent=2))
