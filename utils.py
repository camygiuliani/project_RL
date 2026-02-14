import yaml
import os
import pandas as pd
from datetime import datetime


#loading configuration from yaml file
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_training_csv(history,path):
    if len(history) == 0:
        print("[WARN] No metrics to save.")
        return None


    df = pd.DataFrame(history)
    df.to_csv(path, index=False)
    print(f"[TRAIN] Saved training metrics to: {path}")
    return path