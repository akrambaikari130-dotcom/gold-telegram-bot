import os
import logging
from datetime import datetime
import io

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# التوكن كيجي من المتغير (غادي نضبطوه من بعد)
BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gold_data(interval="1h", period="1mo"):
    try:
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

def add_indicators(df):
    df = df.copy()
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    return df

def generate_ai_signal(df):
    df = df.copy().dropna()
    if len(df) < 60:
        return "انتظار 🟡", 50, "بيانات غير كافية"
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(10).std()
    df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['RSI_Norm'] = df['RSI'] / 100
    df['MACD_Norm'] = df['MACD'] / df['Close']
    features = ['RSI_Norm', 'Price_vs_SMA20', 'Price_vs_SMA50', 'MACD_Norm', 'Volatility']
    df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    df = df.dropna()
    if len(df) < 40:
        return "انتظار 🟡", 50, "بيانات غير كافية"
    X = df[features]
    y = df['Target']
    split = int(len(X) * 0.8)
    model = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    last = X.iloc[[-1]]
    proba = model.predict_proba(last)[0]
    pred = model.predict(last)[0]
    confidence = max(proba) * 100
    last_rsi = df['RSI'].iloc[-1]
    price = df['Close'].iloc[-1]
  
    sma20 = df['SMA_20'].iloc[-1]
    if pred == 1 and confidence > 55 and last_rsi < 70 and price > sma20:
        return "شراء 🟢", confidence, f"توقع صعود • RSI={last_rsi:.1f}"
    elif pred == 0 and confidence > 55 and last_rsi > 30 and price < sma20:
        return "بيع 🔴", confidence, f"توقع هبوط • RSI={last_rsi:.1f}"
    else:
        return "انتظار 🟡", confidence, f"الإشارة غير واضحة (ثقة {confidence:.0f}%)"

def create_chart(df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#1a1a2e')
    ax1, ax2 = axes
    ax1.set_facecolor('#1a1a2e')
    ax2.set_facecolor('#1a1a2e')
    ax1.plot(df['Date'], df['Close'], color='#D4AF37', linewidth=1.5, label='Gold')
    if 'SMA_20' in df.columns:
        ax1.plot(df['Date'], df['SMA_20'], color='#2196F3', linewidth=1, label='SMA 20', alpha=0.8)
    if 'SMA_50' in df.columns:
        ax1.plot(df['Date'], df['SMA_50'], color='#FF9800', linewidth=1, label='SMA 50', alpha=0.8)
    ax1.set_title('XAUUSD / Gold', color='white', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2)
    if 'RSI' in df.columns:
        ax2.plot(df['Date
