# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
import sys
import importlib
import traceback
import asyncio
import nest_asyncio
from threading import Thread
from bot_core import GridTradingBot
from gui import TradingBotGUI
from api_utils import bot_status_check

# Asyncio setup
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

nest_asyncio.apply()

if __name__ == "__main__":
    bot = GridTradingBot()
    # Bot in einem separaten Thread starten, damit GUI nicht blockiert wird
    Thread(target=bot.start, daemon=True).start()
    gui = TradingBotGUI(bot)
    gui.status_var.set("RUNNING")
    gui.run()
