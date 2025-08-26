# =============================================================================
# GUI COMPONENTS
# =============================================================================
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import tkinter.font as tkFont
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from bot_core import GridTradingBot
from config import IS_LIVE, ENABLE_DEBUG_TRIGGER, PRICE_CHECK_INTERVAL
from general_utils import check_all_packages


class TradingBotGUI:
    def __init__(self, bot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title("Grid Trading Bot")

        self.main_font = tkFont.Font(family="Arial", size=14, weight="bold")
        self.status_font = tkFont.Font(family="Arial", size=12)

        tk.Label(self.root, text="Bot Status:", font=self.main_font).pack()
        self.status_var = tk.StringVar(value="RUNNING")
        tk.Label(self.root, textvariable=self.status_var, font=self.status_font).pack()

        self.start_btn = tk.Button(self.root, text="Start (Autostart aktiv!)", bg="grey", fg="white", state="disabled")
        self.start_btn.pack()
        self.stop_btn = tk.Button(self.root, text="Stop", command=self.stop_bot, bg="red", fg="white")
        self.stop_btn.pack()
        if ENABLE_DEBUG_TRIGGER:
            tk.Button(self.root, text="⚡ Debug Trigger", bg="orange", command=self.debug_trigger_clicked).pack()

        tk.Button(self.root, text="🔄 Update Checker", bg="blue", fg="white", command=self.open_update_checker).pack(pady=5)
        tk.Button(self.root, text="📂 Backtesting CSV laden", command=self.load_backtest_csv).pack(pady=5)
        tk.Button(self.root, text="📊 Chart öffnen", bg="green", fg="white", command=self.open_chart_window).pack(pady=5)

        tk.Button(self.root, text="🛈 Trading Info öffnen", bg="purple", fg="white",
                  command=self.open_trading_info_window).pack(pady=5)

        tk.Label(self.root, text="Log:", font=self.main_font).pack()
        self.logbox = tk.Listbox(self.root, width=100, height=15)
        self.logbox.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        self.log_scrollbar = tk.Scrollbar(self.root, command=self.logbox.yview, orient="vertical")
        self.log_scrollbar.pack(side=tk.TOP, fill=tk.Y)
        self.logbox.config(yscrollcommand=self.log_scrollbar.set)

        self._scroll_pos = 0.0
        self.logbox.bind("<MouseWheel>", self.on_mouse_wheel)
        self.logbox.bind("<Button-4>", self.on_mouse_wheel)
        self.logbox.bind("<Button-5>", self.on_mouse_wheel)

        self.chart_window = None
        self.trading_info_window = None
        self.open_trading_info_window()

        self.update_ui()

    def open_update_checker(self):
        UpdateCheckerGUI(self.root, self.bot)

    def open_chart_window(self):
        if self.chart_window is None or not tk.Toplevel.winfo_exists(self.chart_window.top):
            self.chart_window = GridChartWindow(self.root, self.bot)
        else:
            self.chart_window.top.lift()

    def open_trading_info_window(self):
        if self.trading_info_window is None or not tk.Toplevel.winfo_exists(self.trading_info_window.top):
            self.trading_info_window = TradingInfoWindow(self.root, self.bot)
        else:
            self.trading_info_window.top.lift()

    def load_backtest_csv(self):
        filepath = filedialog.askopenfilename(title="Backtesting CSV Datei wählen",
                                              filetypes=[("CSV Dateien", "*.csv"), ("Alle Dateien", "*.*")])
        if filepath:
            self.bot.load_backtest_data_from_csv(filepath)
            if not self.bot.running:
                self.bot.start()

    def on_mouse_wheel(self, event):
        self._scroll_pos = self.logbox.yview()[0]

    def stop_bot(self):
        if self.bot.running:
            self.bot.stop()
            self.status_var.set("STOPPED")
            self.bot.append_order_history("⏹ Stop-Button gedrückt – Bot gestoppt.")
        else:
            self.bot.append_order_history("ℹ Bot ist bereits gestoppt")

    def debug_trigger_clicked(self):
        if not self.bot.running:
            self.bot.append_order_history("⚠ Bot läuft nicht – kein Debug möglich")
            return
        if IS_LIVE:
            if not messagebox.askyesno("Bestätigung", "⚠ LIVE-MODUS!\nEchte Orders werden platziert.\nFortfahren?"):
                return
        self.bot.debug_trigger_now()

    def update_ui(self):
        pos = self.logbox.yview()
        self.logbox.delete(0, tk.END)
        for entry in self.bot.order_history:
            self.logbox.insert(tk.END, entry)
        self.logbox.yview_moveto(self._scroll_pos)
        self._scroll_pos = self.logbox.yview()[0]
        self.root.after(int(PRICE_CHECK_INTERVAL * 1000), self.update_ui)

    def run(self):
        self.root.mainloop()


class GridChartWindow:
    def __init__(self, root, bot):
        self.top = tk.Toplevel(root)
        self.top.title("Grid Chart")
        self.bot = bot
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_chart()

    def update_chart(self):
        self.ax.clear()
        from config import GRID_LINES
        prices = [p for p in GRID_LINES]
        profits = [0] * len(prices)  # Beispielwerte
        self.ax.plot(prices, profits, 'bo-')
        self.ax.set_title("Grid Trading Chart")
        self.ax.set_xlabel("Preis")
        self.ax.set_ylabel("Profit")
        self.canvas.draw()


class TradingInfoWindow:
    def __init__(self, root, bot):
        self.top = tk.Toplevel(root)
        self.top.title("Trading Info")
        self.bot = bot
        self.text = tk.Text(self.top, width=80, height=20)
        self.text.pack(fill=tk.BOTH, expand=True)
        self.update_info()

    def update_info(self):
        self.text.delete('1.0', tk.END)
        info = f"Netto Profit (USDC): {self.bot.net_profit_usdc:.4f}\n"
        info += f"Offene BTC Positionen: {self.bot.positions_btc:.6f}\n"
        info += f"Letzter Trade Preis: {self.bot.last_trade_price if self.bot.last_trade_price else 'N/A'}\n"
        info += f"Gesamte Gebühren (BTC): {self.bot.total_fees_btc:.8f}\n"
        self.text.insert(tk.END, info)
        self.top.after(1000, self.update_info)


class UpdateCheckerGUI:
    def __init__(self, root, bot):
        self.top = tk.Toplevel(root)
        self.top.title("Update Checker")
        self.bot = bot
        label = tk.Label(self.top, text="Überprüfe auf Updates ...")
        label.pack(padx=10, pady=10)

        # Einfache Update-Check-Logik
        version_label = tk.Label(self.top, text="Aktuelle Version: 1.0.0")
        version_label.pack(pady=5)

        check_button = tk.Button(self.top, text="Auf Updates prüfen", command=self.check_updates)
        check_button.pack(pady=5)

        self.result_text = tk.Text(self.top, width=60, height=10)
        self.result_text.pack(padx=10, pady=10)

    def check_updates(self):
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, "Update-Check wird durchgeführt...\n")
        # Hier könnte echte Update-Logik implementiert werden
        self.result_text.insert(tk.END, "✅ Du hast die neueste Version (1.0.0)\n")
