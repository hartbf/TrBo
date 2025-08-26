# =============================================================================
# API UTILITIES
# =============================================================================
import math
import time
import socket
import importlib
from binance.client import Client
from binance import ThreadedWebsocketManager
import asyncio
import nest_asyncio
from config import API_KEY, API_SECRET, PAIR
from logging_setup import log_info

# API Client (Singleton-Instanz an dieser Stelle ist ok, könnte ggf. parametrisiert werden)
client = Client(API_KEY, API_SECRET)

def get_step_size(symbol: str) -> float:
    """Hole die stepSize für ein Symbol (LOT_SIZE Filter)."""
    info = client.get_symbol_info(symbol)
    for f in info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            return float(f['stepSize'])
    # Fallback für unlikely cases
    return 0.00000001

STEP_SIZE = get_step_size(PAIR)

def fmt_qty_safe(q: float, step_size: float = STEP_SIZE, safety_margin: float = 0.98) -> float:
    """Formatiere Menge so, dass sie dem step_size entspricht und nie zu klein ist."""
    q = q * safety_margin
    qty_rounded = math.floor(q / step_size) * step_size
    if qty_rounded < step_size:
        qty_rounded = 0
    # logarithmische Rundung nur, wenn step_size <> 0
    if step_size > 0:
        decimals = max(0, int(-math.log(step_size, 10)))
        return round(qty_rounded, decimals)
    return qty_rounded

def api_call(func, *args, retries: int = 5, delay: int = 1, **kwargs):
    """Führe eine API-Funktion mit Wiederholungen und Exponential Backoff aus."""
    backoff = delay
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            # Rate Limit/HTTP-Fehler behandeln
            if "429" in msg or "rate limit" in msg.lower() or "418" in msg:
                log_info(f"API Rate Limit Fehler erkannt ({attempt + 1}/{retries}): {msg}")
                log_info(f"Warte {backoff} Sekunden wegen Rate Limit...")
                time.sleep(backoff)
                backoff *= 2
            else:
                log_info(f"API Fehler ({attempt + 1}/{retries}): {msg}")
                time.sleep(backoff)
                backoff *= 2
    # Detaillierte Fehlerausgabe
    raise Exception(f"Fehler bei API Call {getattr(func, '__name__', str(func))} nach {retries} Versuchen")

def get_current_price(symbol: str) -> float:
    """Hole aktuellen Preis eines Symbols von Binance."""
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

def bot_status_check(symbol: str = PAIR, timeout: int = 15, bot_ref=None) -> bool:
    """Führt Vorab-Checks für API, Modul, Internet, Stream-Test aus."""
    def log_step(msg):
        if bot_ref and hasattr(bot_ref, 'append_order_history'):
            bot_ref.append_order_history(msg)
        else:
            print(msg)

    log_step("========= BOT START VORAB-CHECK =========")
    try:
        Client(API_KEY, API_SECRET).get_account()
        log_step("✅ API-Key gültig – Account erreichbar")
    except Exception as e:
        log_step(f"❌ API-Keys ungültig oder API nicht erreichbar: {e}")
        return False

    try:
        importlib.import_module("binance")
        log_step("✅ python-binance Modul vorhanden")
    except ImportError:
        log_step("❌ python-binance Modul fehlt!")
        return False

    try:
        socket.create_connection(("api.binance.com", 443), timeout=3)
        log_step("✅ Internetverbindung OK")
    except Exception as e:
        log_step(f"❌ Keine Verbindung zu Binance API: {e}")
        return False

    log_step(f"⏳ Starte {timeout}s Stream-Test …")
    received = {"ok": False}

    def cb(_):
        received["ok"] = True

    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    try:
        if not hasattr(twm, '_ThreadedApiManager__loop') or not twm._ThreadedApiManager__loop.is_running():
            try:
                twm._ThreadedApiManager__loop = asyncio.get_event_loop()
            except RuntimeError:
                nest_asyncio.apply()
                twm._ThreadedApiManager__loop = asyncio.get_event_loop()
    except Exception:
        pass

    try:
        twm.start()
        twm.start_symbol_ticker_socket(callback=cb, symbol=symbol)
        time.sleep(timeout)
        twm.stop()
        return received["ok"]
    except Exception as e:
        log_step(f"❌ Stream-Test fehlgeschlagen: {e}")
        return False
