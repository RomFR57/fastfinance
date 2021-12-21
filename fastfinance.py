from math import fabs

import numpy as np
from numba import jit
from numba.extending import overload


@overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    """
    Numba Overload of np.clip
    :type a: np.ndarray
    :type a_min: int
    :type a_max: int
    :type out: np.ndarray
    :rtype: np.ndarray
    """
    if out is None:
        out = np.empty_like(a)
    for i in range(len(a)):
        if a[i] < a_min:
            out[i] = a_min
        elif a[i] > a_max:
            out[i] = a_max
        else:
            out[i] = a[i]
    return out


@jit(nopython=True, cache=True)
def sma(data, period):
    """
    Simple Moving Average
    :type data: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    out = np.cumsum(data) / period
    out[period:] = out[period:] - out[:-period]
    out[:period - 1] = np.nan
    return out


@jit(nopython=True, cache=True)
def ema(data, period, smoothing=2.0):
    """
    Exponential Moving Average
    :type data: np.ndarray
    :type period: int
    :type smoothing: float
    :rtype: np.ndarray
    """
    size = len(data)
    out = np.array([np.nan] * size)
    w = smoothing / (period + 1)
    for i in range(period - 1, size):
        window = data[i - period + 1:i + 1]
        top = window[period - 1]
        bottom = 1
        for y in range(1, period):
            top += ((1 - w) ** y) * window[period - 1 - y]
            bottom += (1 - w) ** y
        out[i] = top / bottom
    return out


@jit(nopython=True, cache=True)
def trix(data, period, smoothing=2.0):
    """
    Triple Exponential Moving Average
    :type data: np.ndarray
    :type period: int
    :type smoothing: float
    :rtype: np.ndarray
    """
    return ((3 * ema(data, period, smoothing) - (3 * ema(ema(data, period, smoothing), period, smoothing))) +
            ema(ema(ema(data, period, smoothing), period, smoothing), period, smoothing))


@jit(nopython=True, cache=True)
def macd(data, fast, slow, smoothing=2.0):
    """
    Moving Average Convergence Divergence
    :type data: np.ndarray
    :type fast: int
    :type slow: int
    :type smoothing: float
    :rtype: np.ndarray
    """
    return ema(data, fast, smoothing) - ema(data, slow, smoothing)


@jit(nopython=True, cache=True)
def rsi(data, period, smoothing=2.0, f_sma=True, f_clip=True, f_abs=True):
    """
    Relative Strengh Index
    :type data: np.ndarray
    :type period: int
    :type smoothing: float
    :type f_sma: bool
    :type f_clip: bool
    :type f_abs: bool
    :rtype: np.ndarray
    """
    size = len(data)
    delta = np.array([np.nan] * size)
    up = np.array([np.nan] * size)
    down = np.array([np.nan] * size)
    delta = np.diff(data)
    if f_clip:
        up, down = np.clip(delta, a_min=0, a_max=np.max(delta)), np.clip(delta, a_min=np.min(delta), a_max=0)
    else:
        up, down = delta.copy(), delta.copy()
        up[delta < 0] = 0.0
        down[delta > 0] = 0.0
    if f_abs:
        for i, x in enumerate(down):
            down[i] = fabs(x)
    else:
        down = np.abs(down)
    rs = sma(up, period) / sma(down, period) if f_sma else ema(up, period - 1, smoothing) / ema(
        down, period - 1, smoothing)
    out = np.array([np.nan] * size)
    out[1:] = (100 - 100 / (1 + rs))
    return out


@jit(nopython=True, cache=True)
def srsi(data, period, smoothing=2.0, f_sma=True, f_clip=True, f_abs=True):
    """
    Stochastic Relative Strengh Index
    :type data: np.ndarray
    :type period: int
    :type smoothing: float
    :type f_sma: bool
    :type f_clip: bool
    :type f_abs: bool
    :rtype: np.ndarray
    """
    r = rsi(data, period, f_sma, f_clip, f_abs)[period:]
    out = np.array([100 * ((r[i] - np.min(r[i + 1 - period:i + 1])) / (np.max(r[i + 1 - period:i + 1]) - np.min(
        r[i + 1 - period:i + 1]))) for i in range(period - 1, len(r))])
    return np.concatenate((np.array([np.nan] * (len(data) - len(out))), out))


@jit(nopython=True, cache=True)
def bollinger_bands(data, period, dev_up=2.0, dev_down=2.0):
    """
    Bollinger Bands
    :type data: np.ndarray
    :type period: int
    :type dev_up: float
    :type dev_down: float
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    :return: middle, up, down, width
    """
    size = len(data)
    bb_up = np.array([np.nan] * size)
    bb_down = np.array([np.nan] * size)
    bb_width = np.array([np.nan] * size)
    bb_mid = sma(data, period)
    for i in range(period - 1, size):
        std_dev = np.std(data[i - period + 1:i + 1])
        bb_up[i] = bb_mid[i] + (std_dev * dev_up)
        bb_down[i] = bb_mid[i] - (std_dev * dev_down)
        bb_width[i] = (bb_up[i] - bb_down[i]) / bb_mid[i]
    return bb_mid, bb_up, bb_down, bb_width


@jit(nopython=True, cache=True)
def heiken_ashi(c_open, c_high, c_low, c_close):
    """
    Heiken Ashi
    :type c_open: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type c_close: np.ndarray
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    :return: open, high, low, close
    """
    ha_close = (c_open + c_high + c_low + c_close) / 4
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (c_open[0] + c_close[0]) / 2
    for i in range(1, len(c_close)):
        ha_open[i] = (c_open[i - 1] + c_close[i - 1]) / 2
    ha_high = np.maximum(np.maximum(ha_open, ha_close), c_high)
    ha_low = np.minimum(np.maximum(ha_open, ha_close), c_low)
    return ha_open, ha_high, ha_low, ha_close


@jit(nopython=True, cache=True)
def max_plus_min_div_2(data, period, shift=0):
    """
    MAX + MIN / 2
    :type data: np.ndarray
    :type period: int
    :type shift: int
    :rtype np.ndarray
    """
    size = len(data)
    calc = np.array([np.nan] * (size + shift))
    for i in range(period - 1, size):
        calc[i + shift] = (np.max(data[i + 1 - period:i + 1]) + np.min(data[i + 1 - period:i + 1])) / 2
    return calc


@jit(nopython=True, cache=True)
def ichimoku(data, tenkansen=9, kinjunsen=26, senkou_b=52, shift=26):
    """
    Ichimoku
    :type data: np.ndarray
    :type tenkansen: int
    :type kinjunsen: int
    :type senkou_b: int
    :type shift: int
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    :return: tenkansen, kinjunsen, chikou, senkou a, senkou b
    """
    size = len(data)
    n_tenkansen = max_plus_min_div_2(data, tenkansen)
    n_kinjunsen = max_plus_min_div_2(data, kinjunsen)
    n_chikou = np.concatenate(((data[shift:]), (np.array([np.nan] * (size - shift)))))
    n_senkou_a = np.concatenate((np.array([np.nan] * shift), ((n_tenkansen + n_kinjunsen) / 2)))
    n_senkou_b = max_plus_min_div_2(data, senkou_b, shift)
    return n_tenkansen, n_kinjunsen, n_chikou, n_senkou_a, n_senkou_b


@jit(nopython=True, cache=True)
def volume_profile(c_close, c_volume, bins=10):
    """
    Volume Profile
    :type c_close: np.ndarray
    :type c_volume: np.ndarray
    :type bins: int
    :rtype: (np.ndarray, np.ndarray)
    :return: count, price
    """
    min_close = np.min(c_close)
    max_close = np.max(c_close)
    norm = 1.0 / (max_close - min_close)
    sum_h = np.array([0.0] * bins)
    for i in range(len(c_close)):
        sum_h[int((c_close[i] - min_close) * bins * norm)] += c_volume[i] ** 2
    count = np.sqrt(sum_h)
    count /= np.linalg.norm(count)
    return count, np.linspace(min_close, max_close, bins)
