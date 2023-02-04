#### Just playin around with the feature calculators
import pandas as pd
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from scipy.signal import cwt, ricker, find_peaks_cwt, welch
import numpy as np 
from scipy.stats import linregress
import itertools
from statsmodels.tsa.stattools import pacf

feature_fcs = ComprehensiveFCParameters()

timeseries = pd.read_csv("THIS_IS_THE_INPUT_TIMESERIES_FOR_CHECKING_EXTRACTION.csv")

timeseries_sorted = timeseries.sort_values(by = ["id","kind","sort"])


def _roll(a, shift):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def _get_length_sequences_where(x):
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]

def _roll(a, shift):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def _into_subchunks(x, subchunk_length, every_n=1):
    len_x = len(x)
    assert subchunk_length > 1
    assert every_n > 0
    num_shifts = (len_x - subchunk_length) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(subchunk_length)
    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
    return np.asarray(x)[indexer]

# We are going to compute a feature by hand, for each timeseries (kind, id) pair

# This is just a helper function to print whats going on to the screen

window_length = 4
fd1_all = []
fd2_all = []
fd3_all = []
fd4_all = []
fd5_all = []
fd6_all = []
fd7_all = []
fd8_all = []
fd9_all = []
fd10_all = []
fd11_all = []

for group_name, df_group in timeseries_sorted.groupby(["id", "kind"]):

    print("Timeseries index:")
    print(group_name)
    #print("Timeseries values (chunked)")
    ts_values = df_group["val"].tolist()

    ts_values_chunks = [ts_values[x:x+window_length] for x in range(0, len(ts_values), window_length)]


    output_for_cwt_chunks = []
    output_for_friedrich_coefficients_chunks = []
    output_for_stdev_chunks = []
    output_for_linear_trend_res_chunks = []
    output_for_mean_change_res_chunks = []
    output_for_num_crossing_m_res_chunks = []
    output_for_number_peaks_res_chunks = []
    output_for_skewnes_res_chunks = []
    output_for_sym_looking_res_chunks = []
    output_for_var_larger_than_std_res_chunks = []

    for timeseries_chunk in ts_values_chunks:
        
        #print(timeseries_chunk)
        
        # Doing cwt by "hand"
        widths = [2,3,4]
        coeff = 1
        w = 2
        cwt_result = cwt(timeseries_chunk, ricker, widths)
        output_for_cwt_chunks.append(cwt_result[0, coeff])

        # Doing friedrich_coefficients by "hand"
        m = 1
        r = 2
        coeff = 1
        df = pd.DataFrame({"signal": timeseries_chunk[:-1], "delta": np.diff(timeseries_chunk)})
        df["quantiles"] = pd.qcut(df.signal, r)
        quantiles = df.groupby("quantiles")
        fred_result = pd.DataFrame(
            {"x_mean": quantiles.signal.mean(), "y_mean": quantiles.delta.mean()}
        )
        fred_result.dropna(inplace=True)
        fred_res = np.polyfit(fred_result.x_mean, fred_result.y_mean, deg=m)[coeff]
        output_for_friedrich_coefficients_chunks.append(fred_res)

        # stdev
        stdev_res = np.std(np.asarray(timeseries_chunk))
        output_for_stdev_chunks.append(stdev_res)

        # Linear trend
        linReg = linregress(range(len(timeseries_chunk)), timeseries_chunk)
        linear_trend_res = getattr(linReg, "slope")
        output_for_linear_trend_res_chunks.append(linear_trend_res)

        # mean_change
        mean_change_res = (np.asarray(timeseries_chunk)[-1] - np.asarray(timeseries_chunk)[0]) / (len(np.asarray(timeseries_chunk)) - 1)
        output_for_mean_change_res_chunks.append(mean_change_res)

        # number_crossing_m
        m = 5
        positive = np.asarray(timeseries_chunk) > m
        num_crossing_m_res = np.where(np.diff(positive))[0].size
        output_for_num_crossing_m_res_chunks.append(num_crossing_m_res)

        # number_peaks
        n = 1
        x_reduced = timeseries_chunk[n:-n]

        res = None
        for i in range(1, n + 1):
            result_first = x_reduced > _roll(timeseries_chunk, i)[n:-n]

            if res is None:
                res = result_first
            else:
                res &= result_first

            res &= x_reduced > _roll(timeseries_chunk, -i)[n:-n]

        number_peaks_res = np.sum(res)
        output_for_number_peaks_res_chunks.append(number_peaks_res)

    
        # skewness
        skewnes_res = pd.Series.skew(pd.Series(timeseries_chunk))
        output_for_skewnes_res_chunks.append(skewnes_res)

        # symmetry_looking
        mean_median_difference = np.abs(np.mean(np.asarray(timeseries_chunk)) - np.median(np.asarray(timeseries_chunk)))
        max_min_difference = np.max(np.asarray(timeseries_chunk)) - np.min(np.asarray(timeseries_chunk))
        r = 0.1
        sym_looking_res = mean_median_difference < r * max_min_difference
        output_for_sym_looking_res_chunks.append(sym_looking_res)

        # variance_larger_than_standard_deviation
        var_larger_than_std_res = np.var(timeseries_chunk) > np.sqrt(np.var(timeseries_chunk))
        output_for_var_larger_than_std_res_chunks.append(var_larger_than_std_res)


    ##### Now we have the feature timeseries, extract the feature dynamics with window = 4

    output_for_sym_looking_res_chunks = list(map(lambda x : float(x), output_for_sym_looking_res_chunks))

    # cwt_coefficients and with first_location_of_minimum
    fd1 = np.argmin(np.asarray(output_for_cwt_chunks)) / len(np.asarray(output_for_cwt_chunks))
    fd1_all.append(fd1)

    # friedrich_coefficients and index_mass_quantile
    q = 0.1  
    output_for_friedrich_coefficients_chunks = np.asarray(output_for_friedrich_coefficients_chunks)
    abs_x = np.abs(output_for_friedrich_coefficients_chunks)
    s = np.sum(abs_x)
    mass_centralized = np.cumsum(abs_x) / s
    fd2 = (np.argmax(mass_centralized >= q) + 1) / len(output_for_friedrich_coefficients_chunks)
    fd2_all.append(fd2)
    
    # standard_deviation and lempel_ziv_complexity
    bins = 2
    output_for_stdev_chunks = np.asarray(output_for_stdev_chunks)
    bins = np.linspace(np.min(output_for_stdev_chunks), np.max(output_for_stdev_chunks), bins + 1)[1:]
    sequence = np.searchsorted(bins, output_for_stdev_chunks, side="left")
    sub_strings = set()
    n = len(sequence)
    ind = 0
    inc = 1
    while ind + inc <= n:
        sub_str = tuple(sequence[ind : ind + inc])
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    fd3 = len(sub_strings) / n
    fd3_all.append(fd3)

    # linear_trend and longest_strike_above_mean
    fd4 = np.max(_get_length_sequences_where(np.asarray(output_for_linear_trend_res_chunks) > np.mean(np.asarray(output_for_linear_trend_res_chunks))))
    fd4_all.append(fd4)
    
    # Matrix profiles and mean_abs_change
    fd5 = "In progress"
    fd5_all.append(fd5)
    
    # mean_change and mean_second_derivative_central
    fd6 = (np.asarray(output_for_mean_change_res_chunks)[-1] - np.asarray(output_for_mean_change_res_chunks)[-2] - np.asarray(output_for_mean_change_res_chunks)[1] + np.asarray(output_for_mean_change_res_chunks)[0]) / (2 * (len(np.asarray(output_for_mean_change_res_chunks)) - 2))
    fd6_all.append(fd6)
    
    # number_crossing_m and number_cwt_peaks
    fd7 = len(find_peaks_cwt(vector=output_for_num_crossing_m_res_chunks, widths=np.array(list(range(1, n + 1))), wavelet=ricker))
    fd7_all.append(fd7)
    
    # number_peaks and partial_autocorrelation
    n = len(output_for_number_peaks_res_chunks)
    lag, max_demanded_lag = 1, 1
    if max_demanded_lag >= n // 2:
        max_lag = n // 2 - 1
    else:
        max_lag = max_demanded_lag
    if max_lag > 0:
        pacf_coeffs = list(pacf(output_for_number_peaks_res_chunks, method="ld", nlags=max_lag))
        pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))
    else:
        pacf_coeffs = [np.nan] * (max_demanded_lag + 1)

    fd8 = pacf_coeffs[lag]
    fd8_all.append(fd8)
    
    # skewness and spkt_welch_density
    coeff = 1
    freq, pxx = welch(output_for_skewnes_res_chunks, nperseg=min(len(output_for_skewnes_res_chunks), 256))
    fd9 = pxx[coeff]
    fd9_all.append(fd9)
    
    # symmetry_looking and time_reversal_asymmetry_statistic
    n = len(output_for_sym_looking_res_chunks)
    output_for_sym_looking_res_chunks = np.asarray(output_for_sym_looking_res_chunks)
    if 2 * lag >= n:
        fd10 = 0
    else:
        one_lag = _roll(output_for_sym_looking_res_chunks, -lag)
        two_lag = _roll(output_for_sym_looking_res_chunks, 2 * -lag)
        fd10 = np.mean((two_lag * two_lag * one_lag - one_lag * output_for_sym_looking_res_chunks * output_for_sym_looking_res_chunks)[0 : (n - 2 * lag)])

    fd10_all.append(fd10)
    
    # variance_larger_than_standard_deviation and variation_coefficient
    fd11 = np.std(output_for_var_larger_than_std_res_chunks) / np.mean(output_for_var_larger_than_std_res_chunks)
    fd11_all.append(fd11)


for feature_dynamic in [
    fd1_all,
    fd2_all,
    fd3_all,
    fd4_all,
    fd5_all,
    fd6_all,
    fd7_all,
    fd8_all,
    fd9_all,
    fd10_all,
    fd11_all
    ]:
    print("Feature vector:")
    print(feature_dynamic)
    

