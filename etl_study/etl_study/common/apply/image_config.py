"""
apply的bash_operator用的image
"""
from airflow.hooks.base import BaseHook
from src.config import VERSION
from src.config import PROJ_NAME

IMAGE_REGISTRY = BaseHook.get_connection('image_registry').host
IMAGE_REPO = f'up0125/{PROJ_NAME}_etl'
IMAGE = f'{IMAGE_REGISTRY}/{IMAGE_REPO}:{VERSION}'
