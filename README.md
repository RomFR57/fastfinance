# fastfinance

### **Fast Financial Indicators speed up with Numba**

<p align="center">
  <img src="https://imagizer.imageshack.com/img923/9808/uBE2M9.jpg" />
</p>

### **Story :**
I'm a french former AI engineer and indie developer. I put my programmer career aside for few years to become a hospital nurse but french govemment don't allow me to work as a nurse anymore because I refused covid vaccine (I hope you respect my choice because I respect yours) whereas I was there for patients since the beginning of the pandemic. I had to ask my resignation but they keep me suspended and under contract from September 2021 until June 2022. It means that my salary is currently suspended and I'm not allow to fill another job until this date and nobody cares. I couldn't heal people anymore so I decided to come back to programming.

In October 2021, I quickly learned Python to be able to use machine learning for image classification and also build an automated Binance trading bot as a way to pay my rent. Python is fun but slow and I wanted my bot to be able to plot financial indicators in real-time and use them for automatic strategies. I tried at first with built-in Python methods but it was impossible to get the refresh loop time below 1 sec. So I started to convert every financial indicators algorithms to fit Numba requirements and the problem was solved.

I decided to share this library because it helped me a lot and I hope it could help someone which is in a bad situation.
If you use the library and you make a lot of money with it, you should consider to help the ones who need the most.

Now I'm making 3D games and animation controllers for Unity C#. I'm looking for Unity level design contributors for my game.

If you are interested, feel free to contact me @ rom.fr57@gmail.com

### **Example :**
Automated trading bot using fastfinance for plot output, strategies and parameters tuning

Link to Youtube > [Python Binance Trading Bot - Autotune with Bollinger Bands strategy](https://www.youtube.com/watch?v=L5t6aFAETcg)

### **Indicators :**
- Aroon
- Average Directional Index [ ADX ]
- Average True Range [ ATR ]
- Bollinger Bands
- Center Of Gravity [ COG ]
- Chaikin Money Flow [ CMF ]
- Chande Momentum Oscillator [ CMO ]
- Chopiness Index [ CHOP ]
- Cumulative Moving Average [ CMA ]
- Donchian channel
- Double Exponential Moving Average [ DEMA ]
- Entropy
- Exponential Moving Average [ EMA ]
- Exponential Weighted Moving Average [ EWMA ]
- Fourier Transform Fit Extrapolation
- Fractal Dimension Index [ FDI ]
- Heiken Ashi
- Ichimoku
- Kaufman's Adaptive Moving Average [ KAMA ]
- KDJ
- Keltner Channel
- Least Squares Moving Average [ LSMA ]
- Momentum
- Moving Average Convergence Divergence [ MACD ]
- On Balance Volume [ OBV ]
- Polynomial Fit Extrapolation
- Rate Of Change [ ROC ]
- Relative Strengh Index [ RSI ]
- Simple Moving Average [ SMA ]
- Stochastic
- Stochastic Relative Strengh Index [ S-RSI ]
- Supertrend
- Triple Exponential Moving Average [ TRIX ]
- True Range [ TR ]
- Volatility Index [ VIX ]
- Volume Profile
- Weighted Moving Average [ WMA ]
- William %R
- Zero-Lag Least Squares Moving Average [ ZLSMA ]

### **Requirements :**
- [Numba](https://github.com/numba/numba)
- [Numpy](https://github.com/numpy/numpy)

### **Install :**
```python
pip install numba
pip install numpy
```

### **Usage :**
```python
import numpy as np
import fastfinance as ff

data = np.array([2.9, 0.9, 1.9, 8.5, 0.1, 0.6, 1.9, 8.8], dtype=np.float64)

print(ff.ema(data=data, period=3))
```

<h3 align="center">Thanks for ⭐ support !</h3>
