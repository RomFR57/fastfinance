# fastfinance

### **Financial Indicators speed up with Numba**

<p align="center">
  <img src="https://imagizer.imageshack.com/img923/9808/uBE2M9.jpg" />
</p>
 
### **Indicators :**
- Aroon
- Average Directional Index [ ADX ]
- Avergage True Range [ ATR ]
- Bollinger Bands
- Chaikin Money Flow [ CMF ]
- Cumulative Moving Avergage [ CMA ]
- Donchian channel
- Double Exponential Moving Avergage [ DEMA ]
- Exponential Moving Average [ EMA ]
- Exponential Weighted Moving Average [ EWMA ]
- Heiken Ashi
- Ichimoku
- KDJ
- Keltner Channel
- Momentum
- Moving Average Convergence Divergence [ MACD ]
- On Balance Volume [ OBV ]
- Rate Of Change [ ROC ]
- Relative Strengh Index [ RSI ]
- Simple Moving Average [ SMA ]
- Stochastic
- Stochastic Relative Strengh Index [ S-RSI ]
- Triple Exponential Moving Avergage [ TRIX ]
- True Range [ TR ]
- Volatility Index [ VIX ]
- Volume Profile
- Weighted Moving Average [ WMA ]
- William %R

### **Requirements :**
- [Numba](https://github.com/numba/numba)
- [Numpy](https://github.com/numpy/numpy)

### **Install :**
```python
pip install numba
pip install numpy
```

### **Example :**
```python
import numpy as np
import fastfinance as ff

data = np.array([2.9, 0.9, 1.9, 8.5, 0.1, 0.6, 1.9, 8.8], dtype=np.float64)

print(ff.ema(data=data, period=3))
```

<h3 align="center">Thanks for ⭐ support !</h3>
