import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress

class ChartPatternAnalyzer:
    def __init__(self, data):
        self.data = data
        self.patterns = {}
        
    def detect_support_resistance(self, window=20):
        """Detect support and resistance levels"""
        highs = self.data['High'].rolling(window=window, center=True).max()
        lows = self.data['Low'].rolling(window=window, center=True).min()
        
        resistance = self.data[self.data['High'] == highs]['High'].dropna()
        support = self.data[self.data['Low'] == lows]['Low'].dropna()
        
        return {
            'resistance_levels': resistance.tail(3).tolist(),
            'support_levels': support.tail(3).tolist(),
            'current_vs_resistance': (self.data['Close'].iloc[-1] / resistance.iloc[-1] - 1) * 100 if len(resistance) > 0 else 0,
            'current_vs_support': (self.data['Close'].iloc[-1] / support.iloc[-1] - 1) * 100 if len(support) > 0 else 0
        }
    
    def detect_trend_channels(self, period=20):
        """Detect trend channels and breakouts"""
        closes = self.data['Close'].tail(period)
        x = np.arange(len(closes))
        slope, intercept, r_value, _, _ = linregress(x, closes)
        
        trend_line = slope * x + intercept
        deviations = closes - trend_line
        upper_channel = trend_line + deviations.std() * 2
        lower_channel = trend_line - deviations.std() * 2
        
        current_price = closes.iloc[-1]
        channel_position = (current_price - lower_channel[-1]) / (upper_channel[-1] - lower_channel[-1])
        
        return {
            'trend_slope': slope,
            'trend_strength': abs(r_value),
            'channel_position': channel_position,
            'breakout_signal': 1 if channel_position > 0.9 else (-1 if channel_position < 0.1 else 0)
        }
    
    def detect_candlestick_patterns(self):
        """Detect key candlestick patterns"""
        df = self.data.tail(10).copy()
        df['body'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['lower_shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['is_bullish'] = df['Close'] > df['Open']
        
        patterns = {}
        
        # Doji pattern
        avg_body = df['body'].mean()
        patterns['doji'] = (df['body'].iloc[-1] < avg_body * 0.1)
        
        # Hammer/Hanging Man
        patterns['hammer'] = (df['lower_shadow'].iloc[-1] > df['body'].iloc[-1] * 2 and 
                             df['upper_shadow'].iloc[-1] < df['body'].iloc[-1] * 0.5)
        
        # Engulfing patterns
        if len(df) >= 2:
            patterns['bullish_engulfing'] = (not df['is_bullish'].iloc[-2] and df['is_bullish'].iloc[-1] and
                                           df['body'].iloc[-1] > df['body'].iloc[-2])
            patterns['bearish_engulfing'] = (df['is_bullish'].iloc[-2] and not df['is_bullish'].iloc[-1] and
                                           df['body'].iloc[-1] > df['body'].iloc[-2])
        
        return patterns
    
    def calculate_volume_profile(self, bins=20):
        """Analyze volume at different price levels"""
        price_range = self.data['High'].max() - self.data['Low'].min()
        price_bins = np.linspace(self.data['Low'].min(), self.data['High'].max(), bins)
        
        volume_profile = []
        for i in range(len(price_bins)-1):
            mask = (self.data['Close'] >= price_bins[i]) & (self.data['Close'] < price_bins[i+1])
            volume_profile.append(self.data[mask]['Volume'].sum())
        
        max_volume_idx = np.argmax(volume_profile)
        poc_price = (price_bins[max_volume_idx] + price_bins[max_volume_idx+1]) / 2
        
        return {
            'point_of_control': poc_price,
            'current_vs_poc': (self.data['Close'].iloc[-1] / poc_price - 1) * 100,
            'volume_distribution': volume_profile
        }
    
    def detect_divergences(self, period=14):
        """Detect price-momentum divergences"""
        prices = self.data['Close'].tail(period)
        rsi = self.calculate_rsi(prices)
        
        price_peaks = argrelextrema(prices.values, np.greater, order=3)[0]
        rsi_peaks = argrelextrema(rsi.values, np.greater, order=3)[0]
        
        divergence = 0
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if (prices.iloc[price_peaks[-1]] > prices.iloc[price_peaks[-2]] and 
                rsi.iloc[rsi_peaks[-1]] < rsi.iloc[rsi_peaks[-2]]):
                divergence = -1  # Bearish divergence
            elif (prices.iloc[price_peaks[-1]] < prices.iloc[price_peaks[-2]] and 
                  rsi.iloc[rsi_peaks[-1]] > rsi.iloc[rsi_peaks[-2]]):
                divergence = 1   # Bullish divergence
        
        return {'divergence_signal': divergence}
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def analyze_all_patterns(self):
        """Run all pattern analysis"""
        results = {}
        results.update(self.detect_support_resistance())
        results.update(self.detect_trend_channels())
        results['candlestick_patterns'] = self.detect_candlestick_patterns()
        results.update(self.calculate_volume_profile())
        results.update(self.detect_divergences())
        
        # Calculate overall chart signal
        signals = []
        signals.append(results.get('breakout_signal', 0))
        signals.append(results.get('divergence_signal', 0))
        signals.append(1 if results['candlestick_patterns'].get('bullish_engulfing', False) else 0)
        signals.append(-1 if results['candlestick_patterns'].get('bearish_engulfing', False) else 0)
        
        results['chart_signal'] = np.mean([s for s in signals if s != 0]) if any(signals) else 0
        results['pattern_strength'] = len([s for s in signals if s != 0])
        
        return results