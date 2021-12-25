from time import time_ns

import numpy as np

import fastfinance as ff


def benchmark(data=None):
    print("$$$ FASTFINANCE BENCHMARK $$$")
    init = np.random.uniform(low=1, high=1000, size=10)
    if data is None:
        data = np.random.uniform(low=1, high=1000, size=int(input("DATA SIZE : ")))
    size = len(data)
    benchmark_list = [
        ["INIT", ff.sma, (init, 1)],
        ["SMA", ff.sma, (data, size)],
        ["INIT", ff.wma, (init, 1)],
        ["WMA", ff.wma, (data, size)],
        ["INIT", ff.cma, (init,)],
        ["CMA", ff.cma, (data,)],
        ["INIT", ff.ema, (init, 1)],
        ["EMA", ff.ema, (data, size)],
        ["INIT", ff.ewma, (init, 1)],
        ["EWMA", ff.ewma, (data, size)],
        ["INIT", ff.dema, (init, 1)],
        ["DEMA", ff.dema, (data, size)],
        ["INIT", ff.trix, (init, 1)],
        ["TRIX", ff.trix, (data, size)],
        ["INIT", ff.macd, (init, 1, 1)],
        ["MACD", ff.macd, (data, size, size)],
        ["INIT", ff.stoch, (init, init, init, 2, 1)],
        ["STOCHASTIC", ff.stoch, (data, data, data, size, size)],
        ["INIT", ff.kdj, (init, init, init)],
        ["KDJ", ff.kdj, (data, data, data)],
        ["INIT", ff.rsi, (init, 1)],
        ["RSI", ff.rsi, (data, size)],
        ["INIT", ff.srsi, (init, 5)],
        ["S-RSI", ff.srsi, (data, size)],
        ["INIT", ff.bollinger_bands, (init, 1)],
        ["BOLLINGER BANDS", ff.bollinger_bands, (data, size)],
        ["INIT", ff.heiken_ashi, (init, init, init, init)],
        ["HEIKEN ASHI", ff.heiken_ashi, (data, data, data, data)],
        ["INIT", ff.ichimoku, (init, 1, 1, 1, 1)],
        ["ICHIMOKU", ff.ichimoku, [data]],
        ["INIT", ff.volume_profile, (init, init, 1)],
        ["VOLUME PROFILE", ff.volume_profile, (data, data, size)],
        ["INIT", ff.tr, (init, init, init)],
        ["TR", ff.tr, (data, data, data)],
        ["INIT", ff.atr, (init, init, init, 1)],
        ["ATR", ff.atr, (data, data, data, size)],
        ["INIT", ff.adx, (init, init, init, 1, 1)],
        ["ADX", ff.adx, (data, data, data, size, size)],
        ["INIT", ff.obv, (init, init)],
        ["OBV", ff.obv, (data, data)],
        ["INIT", ff.wpr, (init, init, init, 2)],
        ["WPR", ff.wpr, (data, data, data, size)],
        ["INIT", ff.momentum, (init, 1)],
        ["MOMENTUM", ff.momentum, (data, size)],
        ["INIT", ff.momentum, (init, 1)],
        ["ROC", ff.momentum, (data, size)]
    ]
    for i, p in enumerate(benchmark_list):
        if p[0] == "INIT":
            p[1](*p[2])
        else:
            t = time_ns()
            p[1](*p[2])
            print("".join([p[0], " [", str(round((time_ns() - t) / 1000000)), " ms]"]))
