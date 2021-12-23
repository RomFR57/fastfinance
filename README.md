# fastfinance
**Financial Indicators speed up with Numba**

Indicators :
- Average Directional Index
- Avergage True Range
- Bollinger Bands
- Double Exponential Moving Avergage
- Exponential Moving Average
- Heinken Ashi
- Ichimoku
- KDJ
- Moving Average Convergence Divergence
- Relative Strengh Index
- Simple Moving Average
- Stochastic
- Stochastic Relative Strengh Index
- Triple Exponential Moving Avergage
- True Range
- Volume Profile

Requirements :
- [Numba](https://github.com/numba/numba)
- [Numpy](https://github.com/numpy/numpy)

Install :
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
