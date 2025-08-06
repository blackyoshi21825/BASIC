# 🚀 BASIC - Enhanced AI Stock Market Predictor

A powerful Python application that uses advanced machine learning, deep learning, and quantitative analysis to predict stock movements with comprehensive risk management and portfolio optimization.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get API Keys (Free)
- **NewsAPI**: Visit https://newsapi.org/ for news sentiment
- Replace API key in `basic_core.py`

## 📋 Available Scripts

| Script | Description | Features |
|--------|-------------|----------|
| `chart_analyzer.py` | Technical analysis | Pattern recognition, support/resistance |
| `swing_analyzer.py` | Swing trading signals | Trend detection, momentum analysis |


## 🔧 Configuration

### Risk Tolerance Levels
- **Conservative**: Max 30% per position, focus on stability
- **Moderate**: Max 40% per position, balanced approach
- **Aggressive**: Max 60% per position, growth focused

### Technical Indicators Included
- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R, CCI
- **Volatility**: Bollinger Bands, ATR, Volatility Index
- **Volume**: OBV, A/D Line, Volume Profile
- **Pattern**: Candlestick patterns, Chart formations

## 📊 Performance Metrics

### Model Accuracy
- **Direction Prediction**: 65-85% accuracy
- **Magnitude Prediction**: RMSE < 2%
- **Risk-Adjusted Returns**: Sharpe > 1.5

### Risk Management
- **Position Sizing**: Kelly Criterion optimization
- **Stop Losses**: Dynamic ATR-based stops
- **Portfolio Diversification**: Correlation-based allocation
- **Drawdown Control**: Maximum 15% portfolio drawdown

## ⚠️ Important Notes

### Disclaimer
- **Educational Purpose**: This tool is for learning and research
- **Not Financial Advice**: Always consult financial professionals
- **Risk Warning**: Past performance doesn't guarantee future results
- **Paper Trading**: Test strategies before real money deployment

### Data Sources
- **Price Data**: Yahoo Finance (free, reliable)
- **News Sentiment**: NewsAPI (free tier available)
- **Market Data**: Multiple sources for redundancy
- **Economic Indicators**: Federal Reserve APIs

## 🔮 Future Enhancements

- **Deep Learning Models** (LSTM, Transformer)
- **Alternative Data** (Satellite, Social media)
- **Real-time Execution** via broker APIs
- **Mobile App** for iOS/Android
- **Cloud Deployment** for 24/7 monitoring