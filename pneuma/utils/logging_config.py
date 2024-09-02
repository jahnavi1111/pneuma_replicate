import logging
import os


def configure_logging():
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.expanduser("~/Documents/Pneuma/out/pneuma.log")
            ),
            logging.StreamHandler(),
        ],
    )
