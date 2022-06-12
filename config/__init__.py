import yaml


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def set_config(cfg):
    global config
    for key in cfg.keys():
        config[key] = cfg[key]


config = load_config()