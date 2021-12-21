# fastfinance
Financial Indicators speed up with Numba

Indicators:
- Simple Moving Average
- Exponential Moving Average
- Triple Exponential Moving Avergage
- Relative Strengh Index
- Stochastic Relative Strengh Index
- Moving Average Convergence Divergence
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
