import os
import json
import argparse
import configparser
from utils import logs

path = os.path.abspath(".")
config_path = os.path.join(path, 'configs', 'config.ini')
config = configparser.ConfigParser()
config.read(config_path,encoding='utf-8')

def make_parser():
    parser = argparse.ArgumentParser(description="road occupied clienet")
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        required=False,
                        default='{"algorithm_params":{"triton_url":"192.168.113.57:30740"},"algorithm":"road-occupied","consumer_topic":"coverall_detection","producer_topic":"capture_scene","pulsar_address":"pulsar://192.168.113.71:6650","log":{"log_path":"/app/log","log_file":"road-occupied","log_rotation":"1 week","log_retention":"3 week"}}',
                        help='The json sequence of config')
    return parser

# Extract config params
args = make_parser().parse_args()
print(args.config)
config_params = json.loads(args.config)
algorithm_params = config_params["algorithm_params"]
triton_url = algorithm_params["triton_url"]
algorithm = config_params["algorithm"]
consumer_topic = config_params["consumer_topic"]
producer_topic = config_params["producer_topic"]
pulsar_address = config_params["pulsar_address"]
log_params = config_params["log"]
log_file_name = log_params["log_file"]
log_rotation = 7*int(log_params["log_rotation"].split(" ")[0]) if log_params["log_rotation"].split(" ")[1] == "week" else int(log_params["log_rotation"].split(" ")[0])
log_retention = 7*int(log_params["log_retention"].split(" ")[0]) if log_params["log_retention"].split(" ")[1] =="week" else int(log_params["log_retention"].split(" ")[0])

config.set("Log", "rotation_days", str(log_rotation))
config.set("Log", "delete_old_logs", str(log_retention))
config.set("Pulsar", "pulsar_host", pulsar_address)
with open(config_path, "w") as config_file:
    config.write(config_file)
    config_file.close()

logger = logs.Log(log_file_name).logs_setup()