# fastfinance
Financial Indicators speed up with Numba

Indicators:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Triple Exponential Moving Avergage (TRIX)
- Relative Strengh Index (RSI)
- Stochastic Relative Strengh Index (S-RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Ichimoku
- Heinken Ashi
- Volume Profile

Requirements:
```python
pip install numba
pip install numpy
```

Example :
```python
import numpy as np
import fastfinance as ff

input = np.array([7, 8, 4, 1, 3, 2, 5, 9])

output = ff.ema(data=imput, period=5)

print(output)
```
