from typing import Optional

import ee
from ee import EEException

from .logging_config import get_logger

logger = get_logger(__name__)

_EE_INITIALIZED = False

def initialize_ee(project: Optional[str]=None) -> None:
    global _EE_INITIALIZED

    if _EE_INITIALIZED:
        return
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except EEException as e:
        logger.info(f"Failed to initialize the EE project: {e}; attempting authentication.")
        ee.Authenticate()

        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.info("Google Earth Engine authenticated and initialized.")

    _EE_INITIALIZED = True