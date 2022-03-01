"""
MIT License

Copyright (c) 2021 RomFR57 rom.fr57@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
        ["SMA", ff.sma, (data, size)],
        ["WMA", ff.wma, (data, size)],
        ["CMA", ff.cma, (data,)],
        ["EMA", ff.ema, (data, size)],
        ["EWMA", ff.ewma, (data, size)],
        ["DEMA", ff.dema, (data, size)],
        ["TRIX", ff.trix, (data, size)],
        ["MACD", ff.macd, (data, size, size)],
        ["STOCHASTIC", ff.stoch, (data, data, data, size, size)],
        ["KDJ", ff.kdj, (data, data, data)],
        ["RSI", ff.rsi, (data, size)],
        ["S-RSI", ff.srsi, (data, size)],
        ["CHANDE MO", ff.cmo, (data, size)],
        ["BOLLINGER BANDS", ff.bollinger_bands, (data, size)],
        ["KELTNER CHANNEL", ff.keltner_channel, (data, data, data, data, size)],
        ["DONCHIAN CHANNEL", ff.donchian_channel, (data, data, size)],
        ["HEIKEN ASHI", ff.heiken_ashi, (data, data, data, data)],
        ["ICHIMOKU", ff.ichimoku, [data]],
        ["VOLUME PROFILE", ff.volume_profile, (data, data, size)],
        ["TR", ff.tr, (data, data, data)],
        ["ATR", ff.atr, (data, data, data, size)],
        ["ADX", ff.adx, (data, data, data, size, size)],
        ["SUPERTREND", ff.super_trend, (data, data, data, size)],
        ["OBV", ff.obv, (data, data)],
        ["WPR", ff.wpr, (data, data, data, size)],
        ["MOMENTUM", ff.momentum, (data, size)],
        ["ROC", ff.momentum, (data, size)],
        ["AROON", ff.aroon, (data, size)],
        ["CMF", ff.cmf, (data, data, data, data, size)],
        ["VIX", ff.vix, (data, data, size)],
        ["FDI", ff.fdi, (data, size)],
        ["ENTROPY", ff.entropy, (data, data, size)],
        ["POLY FIT EXTRA", ff.poly_fit_extra, (data, 10, size)],
        ["FOURIER FIT EXTRA", ff.fourier_fit_extra, (data, 10, size)],
        ["CHOP INDEX", ff.chop, (data, data, data, data, size)],
    ]
    for i, p in enumerate(benchmark_list):
        p[1](*p[2])
        p[1](*p[2])
        t = time_ns()
        p[1](*p[2])
        print("".join([p[0], " [", str(round((time_ns() - t) / 1000000)), " ms]"]))


if __name__ == "__main__":
    benchmark()