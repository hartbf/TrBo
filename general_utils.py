# =============================================================================
# GENERAL UTILITIES
# =============================================================================
import math
import time
import csv
import re
import json
import tempfile
import os
import traceback
import asyncio
from itertools import islice
from datetime import datetime
from typing import Any, Optional, List, Dict, Callable, Union, Iterator
from logging_setup import log_info
import importlib

# Optional imports je nach Verfügbarkeit
try:
    from pytz import timezone as pytz_timezone

    PYTZ_AVAILABLE: bool = True
except ImportError:
    PYTZ_AVAILABLE: bool = False

try:
    from plyer import notification

    PLYER_AVAILABLE: bool = True
except ImportError:
    PLYER_AVAILABLE: bool = False


def round_step_size(quantity: float, step_size: float) -> float:
    """Rundet eine Menge auf das nearest Floor Vielfache des Schrittgrößenfilters."""
    precision = int(round(-math.log(step_size, 10), 0))
    rounded_qty = math.floor(quantity / step_size) * step_size
    return round(rounded_qty, precision)


def format_price(price: float) -> str:
    """Formatieren der Preisangabe auf zwei Dezimalstellen."""
    return f"{price:.2f}"


def wait_for(seconds: int) -> None:
    """Utility zum Warten mit Pausenanzeige."""
    for i in range(seconds):
        print(f"Warte {seconds - i} Sekunden...")
        time.sleep(1)


def save_order_history_to_file(order_history: List[str], filename: str = "order_history.log") -> bool:
    """Speichert den Orderverlauf in eine Datei."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for line in order_history:
                f.write(line + "\n")
        return True
    except Exception as e:
        print(f"Fehler beim Speichern der Order-History: {e}")
        return False


def calculate_profit_loss(buy_price: float, sell_price: float, quantity: float, fee_rate: float) -> float:
    """Berechnet den Gewinn oder Verlust für einen Trade unter Berücksichtigung der Gebühren."""
    cost = buy_price * quantity * (1 + fee_rate)
    revenue = sell_price * quantity * (1 - fee_rate)
    return revenue - cost


def format_datetime(dt: Optional[datetime] = None) -> str:
    """Formatiert ein datetime-Objekt in einen lesbaren String."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_float(value: Any, default: float = 0.0) -> float:
    """Sicheres Parsen von Fließkommazahlen mit Defaultwert."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_get(dct: Optional[Dict[Any, Any]], key: Any, default: Any = None) -> Any:
    """Sichere Methode, um Werte aus einem Dictionary zu holen."""
    try:
        return dct[key]  # type: ignore
    except (KeyError, TypeError):
        return default


def current_timestamp() -> str:
    """Gibt den aktuellen Zeitstempel als String zurück."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 50,
                       fill: str = '█') -> None:
    """Druckt eine Fortschrittsanzeige in die Konsole."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


def retry(func: Callable, retries: int = 3, delay: int = 2, backoff: int = 2,
          exceptions: Union[type, tuple] = Exception, logger: Optional[Any] = None) -> Any:
    """Führt die Funktion mit Wiederholungen bei Fehlern aus."""
    attempt = 0
    while attempt < retries:
        try:
            return func()
        except exceptions as e:
            if logger:
                logger.info(f"Versuch {attempt + 1} fehlgeschlagen: {e}")
            else:
                print(f"Versuch {attempt + 1} fehlgeschlagen: {e}")
            time.sleep(delay)
            delay *= backoff
            attempt += 1
    raise Exception(f"Funktion {func.__name__} ist nach {retries} Versuchen fehlgeschlagen.")


def human_readable_size(size: float, decimal_places: int = 2) -> str:
    """Wandelt eine Größe in Bytes in ein human-readable-Format um."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024.0
    return f"{size:.{decimal_places}f} PB"


def is_price_in_range(price: float, low: float, high: float) -> bool:
    """Prüft, ob ein Preis innerhalb eines Bereichs liegt."""
    return low <= price <= high


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Sichere Division, um Division durch Null zu vermeiden."""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return default


def timestamp_to_datetime(ts: float) -> Optional[datetime]:
    """Konvertiert Unix-Timestamp in datetime-Objekt."""
    try:
        return datetime.fromtimestamp(ts)
    except Exception:
        return None


def datetime_to_timestamp(dt: datetime) -> Optional[float]:
    """Konvertiert datetime-Objekt in Unix-Timestamp."""
    try:
        return dt.timestamp()
    except Exception:
        return None


def safe_round(value: Any, decimals: int = 8) -> float:
    """Rundet einen Wert auf eine bestimmte Anzahl von Dezimalstellen."""
    try:
        return round(float(value), decimals)
    except Exception:
        return 0.0


def read_csv_prices(filepath: str) -> List[float]:
    """Liest Preise aus einer CSV-Datei und gibt sie als Liste zurück."""
    prices = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    price = parse_float(row[0], default=None)
                    if price is not None:
                        prices.append(price)
    except Exception as e:
        print(f"Fehler beim Lesen der CSV: {e}")
    return prices


def log_exception(e: Exception, logger: Optional[Any] = None) -> None:
    """Loggt eine Exception inklusive Stack-Trace."""
    exc_msg = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    if logger:
        logger.error(exc_msg)
    else:
        print(exc_msg)


def safe_cast(val: Any, to_type: Callable, default: Any = None) -> Any:
    """Sicherer Cast auf definierten Typ mit Defaultwert."""
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def timestamp_now() -> float:
    """Aktueller Zeitstempel als float."""
    return time.time()


def ensure_dir_exists(path: str) -> None:
    """Erstellt ein Verzeichnis, falls es nicht existiert."""
    if not os.path.exists(path):
        os.makedirs(path)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Text kürzen, falls zu lang."""
    if len(text) > max_length:
        return text[:max_length - 3] + "..."
    return text


def is_market_open() -> bool:
    """Prüft, ob der Markt geöffnet ist (z.B. Binance ist 24/7 offen)."""
    return True


def elapsed_seconds(start_time: float) -> float:
    """Gibt Sekunden seit start_time zurück."""
    return time.time() - start_time


def format_pct(value: float, decimals: int = 2) -> str:
    """Formatiert eine Zahl als Prozentstring."""
    try:
        return f"{value * 100:.{decimals}f}%"
    except Exception:
        return "0%"


def safe_min(*args: Optional[float]) -> Optional[float]:
    """Gibt das Minimum einer Liste zurück, ignoriert None-Werte."""
    filtered = [x for x in args if x is not None]  # type: ignore
    if not filtered:
        return None
    return min(filtered)


def safe_max(*args: Optional[float]) -> Optional[float]:
    """Gibt das Maximum einer Liste zurück, ignoriert None-Werte."""
    filtered = [x for x in args if x is not None]  # type: ignore
    if not filtered:
        return None
    return max(filtered)


def chunk_list(lst: List[Any], n: int) -> List[List[Any]]:
    """Teilt eine Liste in n ungefähr gleich große Stücke."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def format_order_side(side: str) -> str:
    """Formatiert 'BUY' oder 'SELL' in ein deutschsprachiges Wort."""
    if side.upper() == "BUY":
        return "Kaufen"
    elif side.upper() == "SELL":
        return "Verkaufen"
    return "Unbekannt"


def current_date_string() -> str:
    """Gibt das aktuelle Datum als String im YYYY-MM-DD Format zurück."""
    return datetime.now().strftime("%Y-%m-%d")


def parse_bool(value: Any) -> bool:
    """Parst einen Wert in Boolean mit tolerantem Verhalten."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ['true', '1', 'yes', 'y']
    return False


def pretty_print_order(order: Dict[str, Any]) -> str:
    """Erzeugt eine gut lesbare Stringdarstellung einer Order."""
    side = format_order_side(order.get('side', ''))
    price = order.get('price', 'N/A')
    qty = order.get('origQty', 'N/A')
    status = order.get('status', 'N/A')
    return f"Order: {side} {qty} zum Preis {price} Status: {status}"


def clear_console() -> None:
    """Löscht die Konsole unter Windows oder Unix-Betriebssystemen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def notify_user(title: str, message: str) -> None:
    """Zeigt eine einfache Desktop-Benachrichtigung an (wenn möglich)."""
    if PLYER_AVAILABLE:
        try:
            notification.notify(title=title, message=message)
        except Exception:
            print(f"{title}: {message}")
    else:
        print(f"{title}: {message}")


def timestamp_to_str(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Konvertiert Timestamp in formatierten String."""
    try:
        return datetime.fromtimestamp(ts).strftime(fmt)
    except Exception:
        return ""


def str_to_timestamp(date_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[float]:
    """Konvertiert formatierte Datum-String in Timestamp."""
    try:
        dt = datetime.strptime(date_str, fmt)
        return dt.timestamp()
    except Exception:
        return None


def is_trade_profitable(buy_price: float, sell_price: float, fee_rate: float) -> bool:
    """Gibt True zurück, wenn Trade profitabel ist unter Berücksichtigung der Gebühren."""
    return sell_price > buy_price * (1 + 2 * fee_rate)


def log_debug(message: str, enable_debug: bool = True) -> None:
    """Schreibt Debug-Nachrichten nur, wenn enable_debug True."""
    if enable_debug:
        print(f"DEBUG: {message}")


def sanitize_filename(name: str) -> str:
    """Konvertiert einen String zu einem gültigen Dateinamen."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def format_duration(seconds: int) -> str:
    """Formatiert eine Zeitdauer in Sekunden als H:M:S."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_time_diff_in_seconds(start: datetime, end: datetime) -> float:
    """Berechnet die Differenz zwischen zwei datetime-Objekten in Sekunden."""
    delta = end - start
    return delta.total_seconds()


def retry_async(func: Callable, retries: int = 3, delay: int = 2, backoff: int = 2,
                exceptions: Union[type, tuple] = Exception, logger: Optional[Any] = None) -> Callable:
    """Async Variante einer Retry-Funktion."""

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        attempt = 0
        current_delay = delay
        while attempt < retries:
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                if logger:
                    logger.info(f"Versuch {attempt + 1} fehlgeschlagen: {e}")
                else:
                    print(f"Versuch {attempt + 1} fehlgeschlagen: {e}")
                await asyncio.sleep(current_delay)
                current_delay *= backoff
                attempt += 1
        raise Exception(f"Funktion {func.__name__} ist nach {retries} Versuchen fehlgeschlagen.")

    return wrapper


def convert_to_float(value: Any, default: float = 0.0) -> float:
    """Versucht, einen Wert in float umzuwandeln, sonst Default."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_string(value: Any, default: str = "") -> str:
    """Wandelt Wert in String um, falls nicht None, sonst Default."""
    if value is None:
        return default
    return str(value)


def retry_with_callback(func: Callable, on_fail: Optional[Callable] = None, retries: int = 3, delay: int = 2) -> Any:
    """Führt Funktion mit retries aus, bei Fehler Callback aufrufen."""
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if on_fail:
                on_fail(e, attempt + 1)
            time.sleep(delay)
    raise Exception(f"Funktion {func.__name__} nach {retries} Versuchen fehlgeschlagen.")


def format_order(order: Dict[str, Any]) -> str:
    """Erstellt eine kurze Beschreibung einer Order."""
    side = safe_string(order.get('side'), 'UNKNOWN')
    price = safe_string(order.get('price'), '0')
    qty = safe_string(order.get('origQty'), '0')
    status = safe_string(order.get('status'), 'UNKNOWN')
    return f"{side} {qty} @ {price} ({status})"


def now_string() -> str:
    """Gibt die aktuelle Zeit als String zurück."""
    return datetime.now().strftime("%H:%M:%S")


def calculate_percentage_change(old: float, new: float) -> float:
    """Berechnet die prozentuale Änderung von old zu new."""
    try:
        return ((new - old) / old) * 100
    except ZeroDivisionError:
        return 0.0


def is_number(value: Any) -> bool:
    """Überprüft, ob ein Wert eine Zahl ist."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_valid_symbol(symbol: str) -> bool:
    """Prüft, ob ein Handelssymbol valide ist (z.B. BTCUSDT)."""
    return bool(re.match(r'^[A-Z]{6,12}$', symbol))


def normalize_symbol(symbol: str) -> str:
    """Normalisiert ein Symbol, Großbuchstaben und ohne Leerzeichen."""
    return symbol.upper().replace(' ', '')


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Begrenzt value auf einen Wertebereich."""
    return max(min_value, min(value, max_value))


def timestamp_to_iso(ts: float) -> str:
    """Konvertiert Unix-Timestamp in ISO 8601 Format."""
    try:
        return datetime.utcfromtimestamp(ts).isoformat() + "Z"
    except Exception:
        return ""


def iso_to_timestamp(iso_str: str) -> Optional[float]:
    """Konvertiert ISO 8601 Datumsstring in Unix-Timestamp."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", ""))
        return dt.timestamp()
    except Exception:
        return None


def flatten_list(list_of_lists: List[List[Any]]) -> List[Any]:
    """Flacht eine Liste von Listen zu einer einzigen Liste ab."""
    return [item for sublist in list_of_lists for item in sublist]


def dict_get_or_default(d: Optional[Dict[Any, Any]], key: Any, default: Any = None) -> Any:
    """Gibt d[key] zurück oder default wenn key nicht existiert."""
    if d and isinstance(d, dict):
        return d.get(key, default)
    return default


def dict_deep_get(d: Optional[Dict[Any, Any]], keys: List[Any], default: Any = None) -> Any:
    """Greift tief verschachtelte Schlüssel in einem Dictionary sicher ab."""
    cur = d
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def safe_increment(dictionary: Dict[Any, Any], key: Any, amount: Union[int, float] = 1) -> None:
    """Erhöht einen Wert in einem Dictionary sicher um amount."""
    if key in dictionary and isinstance(dictionary[key], (int, float)):
        dictionary[key] += amount
    else:
        dictionary[key] = amount


def list_unique(seq: List[Any]) -> List[Any]:
    """Gibt eine Liste mit einzigartigen Elementen zurück, Reihenfolge bleibt erhalten."""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def parse_int(value: Any, default: int = 0) -> int:
    """Konvertiert einen Wert in Integer, sonst Defaultwert."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def invert_side(side: str) -> str:
    """Invertiert den Side-Wert BUY ↔ SELL."""
    if side.upper() == "BUY":
        return "SELL"
    elif side.upper() == "SELL":
        return "BUY"
    return side


def current_utc_datetime() -> datetime:
    """Gibt das aktuelles UTC datetime Objekt zurück."""
    return datetime.utcnow()


def safe_list_get(lst: List[Any], index: int, default: Optional[Any] = None) -> Any:
    """Sicherer Listenindex-Abruf mit Default."""
    try:
        return lst[index]
    except IndexError:
        return default


def is_order_complete(order: Dict[str, Any]) -> bool:
    """Prüft, ob eine Order den Status FILLED oder PARTIALLY_FILLED hat."""
    status = order.get('status', '').upper()
    return status in ['FILLED', 'PARTIALLY_FILLED']


def clear_list(lst: List[Any]) -> None:
    """Leert eine Liste inplace."""
    lst.clear()


def safe_pop(dictionary: Dict[Any, Any], key: Any, default: Optional[Any] = None) -> Any:
    """Sicheres Entfernen eines Schlüssels aus einem Dictionary."""
    if key in dictionary:
        return dictionary.pop(key)
    return default


def is_float_string(s: Any) -> bool:
    """Überprüft, ob ein String eine gültige float Zahl repräsentiert."""
    try:
        float(s)
        return True
    except Exception:
        return False


def timestamp_now_ms() -> int:
    """Gibt aktuellen Zeitstempel in Millisekunden zurück."""
    return int(round(time.time() * 1000))


def list_diff(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Gibt die Elemente, die in list1 aber nicht in list2 sind, zurück."""
    return [item for item in list1 if item not in list2]


def split_string_by_length(s: str, length: int) -> List[str]:
    """Teilt einen String in Stücke der Länge length auf."""
    return [s[i:i + length] for i in range(0, len(s), length)]


def dict_merge(d1: Dict[Any, Any], d2: Dict[Any, Any]) -> Dict[Any, Any]:
    """Fügt zwei Dictionaries zusammen, d2 überschreibt d1 bei Konflikten."""
    result = d1.copy()
    result.update(d2)
    return result


def is_market_open_now(timezone: str = 'UTC', start_hour: int = 0, end_hour: int = 24) -> bool:
    """Überprüft, ob die aktuelle Zeit innerhalb der Marktöffnungszeiten liegt."""
    if not PYTZ_AVAILABLE:
        log_info("⚠ pytz nicht installiert, kann Marktöffnungszeiten nicht prüfen")
        return True

    try:
        now = datetime.now(pytz_timezone(timezone))
        return start_hour <= now.hour < end_hour
    except Exception:
        return True


def atomic_write(filepath: str, data: str, mode: str = 'w', encoding: str = 'utf-8') -> None:
    """Schreibt Daten atomar in eine Datei, um Korruption zu vermeiden."""
    dirpath = os.path.dirname(filepath)
    with tempfile.NamedTemporaryFile(mode=mode, encoding=encoding, dir=dirpath, delete=False) as tmpfile:
        tmpfile.write(data)
        tempname = tmpfile.name
    os.replace(tempname, filepath)


def chunked_iterable(iterable: Iterator[Any], chunk_size: int) -> Iterator[List[Any]]:
    """Erzeugt Iteratoren auf Teilstücke der Größe chunk_size."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


def calculate_ema(prices: List[float], period: int = 10) -> Optional[float]:
    """Berechnet den gleitenden Durchschnitt EMA einer Liste von Preisen."""
    if not prices or len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema_values = [sum(prices[:period]) / period]
    for price in prices[period:]:
        ema = price * k + ema_values[-1] * (1 - k)
        ema_values.append(ema)
    return ema_values[-1]


def merge_dicts(*dicts: Dict[Any, Any]) -> Dict[Any, Any]:
    """Fasst mehrere Dictionaries zusammen, spätere überschreiben frühere."""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def filter_none_values(d: Dict[Any, Any]) -> Dict[Any, Any]:
    """Filtert ein Dictionary, um Einträge mit None-Werten zu entfernen."""
    return {k: v for k, v in d.items() if v is not None}


def safe_json_loads(s: str, default: Optional[Any] = None) -> Optional[Any]:
    """Parst JSON sicher, gibt Default bei Fehlern zurück."""
    try:
        return json.loads(s)
    except Exception:
        return default


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """JSON Serienalisierung mit Fehlerbackup."""
    try:
        return json.dumps(obj, **kwargs)
    except Exception:
        return ""


def timer(func: Callable) -> Callable:
    """Dekorator zur Messung der Ausführungszeit einer Funktion."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMER] {func.__name__} dauerte {end - start:.6f} Sekunden")
        return result

    return wrapper


def ensure_list(obj: Any) -> List[Any]:
    """Stellt sicher, dass ein Objekt eine Liste ist."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def ensure_dict(obj: Any) -> Dict[Any, Any]:
    """Stellt sicher, dass ein Objekt ein Dictionary ist."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return {}


def is_time_between(start_time: datetime.time, end_time: datetime.time,
                    check_time: Optional[datetime.time] = None) -> bool:
    """Prüft ob die aktuelle Zeit innerhalb eines Bereichs liegt."""
    if check_time is None:
        check_time = datetime.now().time()
    if start_time <= end_time:
        return start_time <= check_time <= end_time
    else:
        return check_time >= start_time or check_time <= end_time


def check_all_packages(gui: bool = False, bot_ref: Optional[Any] = None) -> bool:
    """Überprüft, ob alle notwendigen Pakete installiert sind."""
    required = ['binance', 'matplotlib', 'tkinter', 'openpyxl', 'requests', 'dotenv']
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        msg = f"Fehlende Pakete: {', '.join(missing)}. Bitte installieren!"
        if gui and bot_ref:
            bot_ref.append_order_history(msg)
            import tkinter.messagebox as messagebox
            messagebox.showwarning("Fehlende Pakete", msg)
        else:
            print(msg)
        return False
    return True
