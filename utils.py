import yaml


#loading configuration from yaml file
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
