# fastfinance
Financial Indicators speed up with Numba

Indicators:
- SMA
- EMA
- TRIX
- RSI
- Stochastic RSI
- MACD
- Bollinger Bands
- Ichimoku Cloud
- Heinken Ashi

Requirements:
```python
pip install numpy
pip install numba
```

Example :
```python
import numpy as np
import fastfinance as ff

input = np.array([7, 8, 4, 1, 3, 2, 5, 9])

output = ff.ema(data=imput, period=5)

print(output)
```
