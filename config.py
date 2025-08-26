# =============================================================================
# CONFIG.PY – KONFIGURATIONSEINSTELLUNGEN UND GRUNDKONSTANTEN
# =============================================================================

import os
from dotenv import load_dotenv

# Umgebungsvariablen aus .env-Datei laden (für API-Keys usw.)
load_dotenv()

# API Keys sicher aus Umgebungsvariablen lesen
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError(
        "API-Keys nicht gefunden! Bitte überprüfe deine .env Datei und den Pfad."
    )

# Trading-Einstellungen (kann ggf. auch aus externer settings.json geladen werden)
PAIR = "BTCUSDC"
MID_PRICE = 100000.0
DIFF_VALUE = 50.0
GRID_COUNT = 3
POSITION_SIZE_BTC = 0.00016819

# Berechnete Werte
RANGE_LOW = MID_PRICE - DIFF_VALUE
RANGE_HIGH = RANGE_LOW + (GRID_COUNT * DIFF_VALUE)
TRIGGER_PRICE = MID_PRICE
GRID_SIZE = DIFF_VALUE
GRID_LINES = [RANGE_LOW + i * GRID_SIZE for i in range(GRID_COUNT + 1)]
STOP_LOSS_LINE = RANGE_LOW - GRID_SIZE

# Bot-Einstellungen
IS_LIVE = True
ENABLE_DEBUG_TRIGGER = True
PRICE_CHECK_INTERVAL = 1.0
ORDER_CHECK_INTERVAL = 10
