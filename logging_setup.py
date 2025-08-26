# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
import logging
from logging.handlers import RotatingFileHandler

# Empfehlung: Definiere einen benannten Logger fürs gesamte Projekt
logger = logging.getLogger("grid_trading_bot")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    'grid_trading_bot.log', maxBytes=500_000, backupCount=3, encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_info(msg):
    logger.info(msg)
