import argparse
import yaml

def load_cfg(cfg_path):
    with open(cfg_path, mode='r') as f:
        yaml_data = f.read()

    data = yaml.load(yaml_data, Loader=yaml.Loader)
    return data

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/config.yaml', help='cfg.yaml path')
    return parser.parse_args()
