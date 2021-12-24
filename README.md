# fastfinance

**Financial Indicators speed up with Numba**

![alt text](https://imagizer.imageshack.com/img923/9808/uBE2M9.jpg)

 
**Indicators :**
- Average Directional Index [ ADX ]
- Avergage True Range [ ATR ]
- Bollinger Bands
- Double Exponential Moving Avergage [ DEMA ]
- Exponential Moving Average [ EMA ]
- Exponential Weighted Moving Average [ EWMA ]
- Heiken Ashi
- Ichimoku
- KDJ
- Moving Average Convergence Divergence [ MACD ]
- Relative Strengh Index [ RSI ]
- Simple Moving Average [ SMA ]
- Stochastic
- Stochastic Relative Strengh Index [ S-RSI ]
- Triple Exponential Moving Avergage [ TRIX ]
- True Range [ TR ]
- Volume Profile

**Requirements :**
- [Numba](https://github.com/numba/numba)
- [Numpy](https://github.com/numpy/numpy)

**Install :**
```python
pip install numba
pip install numpy
```

**Example :**
```python
import numpy as np
import fastfinance as ff

data = np.array([2.9, 0.9, 1.9, 8.5, 0.1, 0.6, 1.9, 8.8], dtype=np.float64)

print(ff.ema(data=data, period=3))
```
```python
[       nan        nan 1.75714286 5.52857143 2.75714286 1.58571429 1.27142857 5.65714286]
```
