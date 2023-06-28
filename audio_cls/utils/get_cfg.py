import os
import sys
import logging
import importlib

logger = logging.getLogger(__name__)


def get_cfg_by_file(cfg_file):
    try:
        sys.path.append(os.path.dirname(cfg_file))
        current_cfg = importlib.import_module(os.path.basename(cfg_file).split(".")[0])
        logger.info(f'Import {cfg_file} successfully!')
    except Exception:
        raise ImportError(f'Fail to import {cfg_file}')
    return current_cfg
