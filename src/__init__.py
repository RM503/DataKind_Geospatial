from common.logging_config import configure_logging
configure_logging()

from common.logging_config import get_logger
logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()