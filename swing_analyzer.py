import numpy as np
import pandas as pd

class SwingAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def detect_swing_points(self, window=5):
        """Detect swing highs and lows like TradingView"""
        highs = self.data['High']
        lows = self.data['Low']
        
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(highs) - window):
            # Swing high: current high is highest in window
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                swing_highs.append((i, highs.iloc[i]))
            
            # Swing low: current low is lowest in window  
            if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                swing_lows.append((i, lows.iloc[i]))
        
        return swing_highs, swing_lows
    
    def get_swing_direction(self):
        """Determine current swing direction"""
        swing_highs, swing_lows = self.detect_swing_points()
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0
        
        # Get last two swing points
        last_high = swing_highs[-1]
        prev_high = swing_highs[-2] if len(swing_highs) > 1 else swing_highs[-1]
        last_low = swing_lows[-1]
        prev_low = swing_lows[-2] if len(swing_lows) > 1 else swing_lows[-1]
        
        # Higher highs and higher lows = uptrend
        higher_highs = last_high[1] > prev_high[1]
        higher_lows = last_low[1] > prev_low[1]
        
        # Lower highs and lower lows = downtrend
        lower_highs = last_high[1] < prev_high[1]
        lower_lows = last_low[1] < prev_low[1]
        
        if higher_highs and higher_lows:
            return 1  # Bullish swing
        elif lower_highs and lower_lows:
            return -1  # Bearish swing
        else:
            return 0  # Sideways
    
    def calculate_swing_strength(self):
        """Calculate swing strength based on momentum"""
        close = self.data['Close']
        
        # Multi-timeframe momentum
        momentum_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        momentum_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        
        # Volume confirmation
        volume_avg = self.data['Volume'].rolling(20).mean()
        volume_strength = self.data['Volume'].iloc[-5:].mean() / volume_avg.iloc[-1]
        
        # RSI position
        rsi = self.calculate_rsi(close)
        rsi_current = rsi.iloc[-1]
        
        # Combine factors
        strength = 0
        if momentum_5 > 2: strength += 1
        if momentum_20 > 5: strength += 1
        if volume_strength > 1.2: strength += 1
        if 30 < rsi_current < 70: strength += 1
        
        return strength / 4  # Normalize to 0-1
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))