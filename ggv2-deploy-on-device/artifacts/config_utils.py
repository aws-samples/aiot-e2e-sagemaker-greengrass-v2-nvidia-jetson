import os
import sys
import json
import logging
from logging import INFO, StreamHandler, getLogger
from os import environ, path
from sys import stdout

from awsiot.greengrasscoreipc.model import QOS

def load_classes_dict(filename='classes_dict.json'):
    with open(filename, 'r') as fp:
        classes_dict = json.load(fp)

    classes_dict = {int(k):v for k,v in classes_dict.items()}        
    return classes_dict
    

# Set all the constants
USE_GPU = 0
SCORE_THRESHOLD = 0.25
MAX_NO_OF_RESULTS = 3
SHAPE = (224,224)
QOS_TYPE = QOS.AT_LEAST_ONCE
TIMEOUT = 10

# Intialize all the variables with default values
DEFAULT_PREDICTION_INTERVAL_SECS = 3
ENABLE_SEND_MESSAGE = False
TOPIC = "ml/bioplus/imgclassification"

# Get a logger
logger = logging.getLogger(__name__)
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
handler = StreamHandler(stdout)
logger.setLevel(INFO)
logger.addHandler(handler)

# Get the model directory and images directory from the env variables.
try:
    MODEL_CPU_DIR = path.expandvars(environ.get("MODEL_CPU_DIR"))
    MODEL_GPU_DIR = path.expandvars(environ.get("MODEL_GPU_DIR"))
    SAMPLE_IMAGE_DIR = path.expandvars(environ.get("SAMPLE_IMAGE_DIR"))
except TypeError:
    MODEL_CPU_DIR = f'{os.getcwd()}/model_cpu'
    MODEL_GPU_DIR = f'{os.getcwd()}/model_gpu'
    SAMPLE_IMAGE_DIR = f'{os.getcwd()}/sample_images'

LABEL_MAP = load_classes_dict()
