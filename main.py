
import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone

# =========================
# RLA XAUUSD SIGNAL BOT
# Initial demo version
# =========================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Yahoo Gold Futures is a proxy, NOT Exness spot XAUUSD.
SYMBOL = "GC=F"

RSI_PERIOD = 14
RVI_PERIOD = 10
ATR_PERIOD = 14

MA_FAST = 21
MA_MID = 50
MA_SLOW = 200

def send_telegram(message, photo=None):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials are not configured.")
        return

    base = f"https://api.telegram.org/bot{BOT_TOKEN}/"

    try:
        if photo:
            with open(photo, "rb") as image:
                response = requests.post(
                    base + "sendPhoto",
                    data={"chat_id": CHAT_ID, "caption": message},
                    files={"photo": image},
                    timeout=30
                )
        else:
            response = requests.post(
                base + "sendMessage",
                data={"chat_id": CHAT_ID, "text": message},
                timeout=30
            )

        response.raise_for_status()
        print("Telegram delivery successful.")

    except Exception as error:
        print("Telegram delivery failed:", type(error).__name__)


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1/period, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/period, adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rvi(data, period=10):
    numerator = (
        (data["Close"] - data["Open"]) +
        2 * (data["Close"].shift(1) - data["Open"].shift(1)) +
        2 * (data["Close"].shift(2) - data["Open"].shift(2)) +
        (data["Close"].shift(3) - data["Open"].shift(3))
    ) / 6

    denominator = (
        (data["High"] - data["Low"]) +
        2 * (data["High"].shift(1) - data["Low"].shift(1)) +
        2 * (data["High"].shift(2) - data["Low"].shift(2)) +
        (data["High"].shift(3) - data["Low"].shift(3))
    ) / 6

    main = (
        numerator.rolling(period).sum() /
        denominator.rolling(period).sum().replace(0, np.nan)
    )

    signal = (
        main + 2 * main.shift(1) +
        2 * main.shift(2) + main.shift(3)
    ) / 6

    return main, signal


def atr(data, period=14):
    previous_close = data["Close"].shift(1)

    true_range = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - previous_close).abs(),
        (data["Low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    return true_range.ewm(
        alpha=1/period, adjust=False
    ).mean()


def get_data(interval):
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{SYMBOL}?range=30d&interval={interval}"
    )

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20
    )
    response.raise_for_status()

    result = response.json()["chart"]["result"]

    if not result:
        raise ValueError("No market data returned")

    item = result[0]
    quote = item["indicators"]["quote"][0]

    data = pd.DataFrame({
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Volume": quote.get("volume")
    }, index=pd.to_datetime(item["timestamp"], unit="s", utc=True))

    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    if len(data) < 220:
        raise ValueError("Insufficient candles for MA200")

    return data


def add_indicators(data):
    data = data.copy()

    data["MA21"] = data["Close"].rolling(MA_FAST).mean()
    data["MA50"] = data["Close"].rolling(MA_MID).mean()
    data["MA200"] = data["Close"].rolling(MA_SLOW).mean()

    data["RSI"] = rsi(data["Close"], RSI_PERIOD)
    data["RVI"], data["RVI_SIGNAL"] = rvi(data, RVI_PERIOD)
    data["ATR"] = atr(data, ATR_PERIOD)

    return data.dropna()


def create_chart(data, direction, entry, stop):
    chart = data.tail(70).copy()

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#101820")

    for date, row in chart.iterrows():
        rising = row["Close"] >= row["Open"]
        color = "#00d084" if rising else "#ff5364"

        ax.vlines(
            date, row["Low"], row["High"],
            color=color, linewidth=1
        )

        bottom = min(row["Open"], row["Close"])
        height = max(abs(row["Close"] - row["Open"]), 0.01)

        ax.bar(
            date, height, bottom=bottom,
            width=0.00035, color=color
        )

    ax.plot(chart.index, chart["MA21"], label="MA21")
    ax.plot(chart.index, chart["MA50"], label="MA50")
    ax.plot(chart.index, chart["MA200"], label="MA200")

    ax.axhline(entry, color="cyan", linestyle="--",
               label=f"Entry {entry:.2f}")
    ax.axhline(stop, color="red", linestyle="--",
               label=f"Initial SL {stop:.2f}")

    ax.set_title(
        f"RLA XAUUSD | {direction} | M5",
        color="white", fontsize=16
    )
    ax.legend()
    ax.grid(alpha=0.15)
    ax.tick_params(colors="white")
    fig.autofmt_xdate()

    filename = "rla_signal.png"
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return filename


def analyze():
    m5 = add_indicators(get_data("5m"))
    m15 = add_indicators(get_data("15m"))

    # Do not use the currently forming candle.
    m5 = m5.iloc[:-1]
    m15 = m15.iloc[:-1]

    previous = m5.iloc[-2]
    current = m5.iloc[-1]
    trend = m15.iloc[-1]

    buy_trend = (
        trend["MA21"] > trend["MA50"] >
        trend["MA200"]
    )

    sell_trend = (
        trend["MA21"] < trend["MA50"] <
        trend["MA200"]
    )

    buy_cross = (
        previous["RVI"] <= previous["RVI_SIGNAL"]
        and current["RVI"] > current["RVI_SIGNAL"]
    )

    sell_cross = (
        previous["RVI"] >= previous["RVI_SIGNAL"]
        and current["RVI"] < current["RVI_SIGNAL"]
    )

    direction = None

    if buy_trend and buy_cross and current["RSI"] > 50:
        direction = "BUY"

    elif sell_trend and sell_cross and current["RSI"] < 50:
        direction = "SELL"

    if not direction:
        print("No confirmed signal.")
        return

    entry = float(current["Close"])
    volatility = float(current["ATR"])

    if not np.isfinite(volatility) or volatility <= 0:
        print("Invalid ATR.")
        return

    stop = (
        entry - 2 * volatility
        if direction == "BUY"
        else entry + 2 * volatility
    )

    chart_file = create_chart(
        m5, direction, entry, stop
    )

    message = (
        f"RLA XAUUSD SIGNAL\n\n"
        f"{direction}\n"
        f"Instrument: Gold Futures proxy (GC=F)\n"
        f"Timeframe: M5 + M15\n"
        f"Entry: {entry:.2f}\n"
        f"Initial SL: {stop:.2f}\n"
        f"RSI: {current['RSI']:.2f}\n"
        f"RVI: {current['RVI']:.4f}\n"
        f"ATR: {volatility:.2f}\n\n"
        f"DEMO SIGNAL — Not verified spot XAUUSD"
    )

    send_telegram(message, chart_file)


if __name__ == "__main__":
    print("RLA Signal Bot started.")

    try:
        analyze()
    except Exception as error:
        print("Analysis error:", type(error).__name__, str(error))
