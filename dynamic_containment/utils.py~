import pandas as pd
from datetime import timedelta, date
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import special
import openturns as ot
from scipy.optimize import least_squares
import warnings
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm


def hist_fit_2gauss(x, bin_cents, hist):
    
    gauss1 = normal_dist(bin_cents, x[1], mu=-x[0])
    gauss2 = normal_dist(bin_cents, x[1], mu=x[0])
    
    gauss = np.where(bin_cents<0, gauss1, gauss2)
    gauss = gauss / np.trapz(gauss, bin_cents)
    
    return gauss-hist


def hist_fit_gauss(x, bin_cents, hist):
    
    gauss = normal_dist(bin_cents, x[1], mu=x[0])
    gauss = gauss / np.trapz(gauss, bin_cents)
    
    return gauss-hist


def extract_time_block(min_t, max_t, table):
    
    return table.query("date>=@min_t & date<=@max_t")


def battery_power(delta_freq, service_power):
    
    max_percent = 4.0 / service_power
    min_percent = -5.0 / service_power
    
    if max_percent > 1.0:
        max_percent = 1.0
    if min_percent < -1.0:
        min_percent = -1.0
        
    
    charge_efficiency = 0.9
    alpha_upper = 0.95/0.3
    alpha_lower = 0.05/0.2
    
    ccharge_upper = 1 - 0.5 * alpha_upper
    
    if delta_freq >= 0.2:
        battery_charge = ccharge_upper + alpha_upper * delta_freq
    elif delta_freq <= -0.2:
        battery_charge = -ccharge_upper + alpha_upper * delta_freq
    elif delta_freq >= 0.015:
        battery_charge = alpha_lower * delta_freq
    elif delta_freq <= -0.015:
        battery_charge = alpha_lower * delta_freq
    else:
        battery_charge = 0
        
    if battery_charge > max_percent:
        battery_charge = max_percent
    elif battery_charge < min_percent:
        battery_charge = min_percent
        
    if battery_charge > 0.0:
        battery_charge = charge_efficiency * battery_charge
        
    return battery_charge


def add_charge_info(input_df, service_power):

    df = input_df.copy()
    
    df["battery_charge_percent"] = [battery_power(dfi, service_power) for dfi in df["delta_freq"]]
    df["battery_charge"] = service_power * df["battery_charge_percent"]
    df['culm_charge'] = df["battery_charge"].cumsum()/3600.0
    
    return df


def read_and_clean(path='data/task_data_1hz.csv.gz'):
    
    freq_table = pd.read_csv(path)
    freq_table = freq_table.rename(
        {
            "datetime": "date",
            "f_hz": "freq"
        }, axis=1
    )
    freq_table["date"] = freq_table["date"].astype("datetime64[ns]")
    freq_table["delta_freq"] = freq_table["freq"] - 50
    
    return freq_table


def spectral_power(df, fs=1):
    
    x, power = signal.welch(df.delta_freq, fs=fs)
    
    return x, power


def simulate_delta_f(delta_f):
    
    x, power = spectral_power(delta_f, fs=1)
    
    # Create the frequency grid:
    fmin = x[0]
    df = x[1]-x[0]
    N = len(x)
    myFrequencyGrid =  ot.RegularGrid(fmin, df, N)
    
    # Create the collection of HermitianMatrix:
    myCollection = ot.HermitianMatrixCollection()
    for k in range(N):
        frequency = myFrequencyGrid.getValue(k)
        matrix = ot.HermitianMatrix(1)
        matrix[0, 0] = power[k]
        myCollection.add(matrix)
        
    # Create the spectral model:
    mySpectralModel = ot.UserDefinedSpectralModel(myFrequencyGrid, myCollection)
    
    # define a mesh
    myTimeGrid =  ot.RegularGrid(0.0, 1.0, 2528100)
    
    # create the process
    process = ot.SpectralGaussianProcess(mySpectralModel, myTimeGrid)
    
    return process


def fill_in_spectra(input_df, sim):

    df = input_df.copy()
    
    sim = sim.rename({'delta_freq': 'sim_delta_freq'}, axis=1).drop('date', axis=1)
    
    if 'sim_delta_freq' in df.columns:
        df = df.drop('sim_delta_freq', axis=1).merge(sim, left_index=True, right_index=True)
    else:
        df = df.merge(sim, left_index=True, right_index=True)
    
    df['filled_delta_freq'] = df['delta_freq']
    df['filled_delta_freq'] = df['filled_delta_freq'].fillna(df['sim_delta_freq'])
    
    return df


def normal_dist(x, sigma, mu=0):
    
    return np.exp(-0.5 * (np.abs(x-mu)**2.0 / sigma**2.0)) / (np.sqrt(2.0 * np.pi) * sigma)


def calc_gaussian_slope(sigma, service_power):
    
    delta_freq = np.linspace(-1.5, 1.5, num=200)
    y = np.array([battery_power(deltaf, service_power) for deltaf in delta_freq])
    
    charge = np.trapz(
        (
            normal_dist(delta_freq, sigma) * y
        ),
        delta_freq
    )
    
    return charge * service_power / 3600


def calc_fitted_slope(pdf, bin_cents, service_power):
    
    delta_freq = bin_cents#np.linspace(-1.5, 1.5, num=2000)
    y = np.array([battery_power(deltaf, service_power) for deltaf in delta_freq])
    
    interped_pdf = interp1d(bin_cents, pdf, fill_value='extrapolate',  kind='linear')
    
    charge = np.trapz(
        (
            interped_pdf(delta_freq) * y
        ),
        delta_freq
    )
    
    return charge * service_power / 3600


def calc_data_slope(input_df, service_power, bins=200):

    df = input_df.copy()
    
    p, x = np.histogram(df["delta_freq"], bins=bins, density=True)
    
    delta_freq = 0.5 * (x[1:] + x[:-1])
    
    y = np.array([battery_power(deltaf, service_power) for deltaf in delta_freq])
    
    charge = np.trapz(
        (
            p * y
        ),
        delta_freq
    )
    
    return charge * service_power / 3600


def generate_sample_df(pdf, bin_cents, num_samps):
    
    delta_bin = bin_cents[1]-bin_cents[0]
        
    cdf = np.array([np.trapz(pdf[:i], bin_cents[:i]) for i in range(len(bin_cents))])
    cdf = cdf/cdf[-1]
    
    ppf = interp1d(cdf, bin_cents-delta_bin, fill_value='extrapolate',  kind='linear')
    
    our_samps = ppf(np.random.uniform(size=num_samps))
    
    return our_samps


def temporal_covariance(input_df, max_sep=None, step=100):

    df = input_df.copy()
    
    cov_work = df[["delta_freq"]]
    
    deltas = df[["delta_freq"]]
    deltas["shift_freq"] = df["delta_freq"]
    
    cov = {}
    
    if max_sep:
        num = max_sep
    else:
        num = len(deltas)-1
    
    for i in tqdm(range(0, num, step)):
        
        if not i % step:
            cov[i] = deltas.cov().iloc[0,1]
        
            deltas["shift_freq"] = deltas["shift_freq"].shift(step)
            deltas = deltas.iloc[step:]
        
    return cov


def square_cov(cov, max_time, time_step=1):
    
    cov_interp = interp1d(list(cov.keys()), list(cov.values()), fill_value='extrapolate', kind='linear')
    
    times = np.arange(0, max_time, time_step)
    
    cov_arr = np.zeros([len(times), len(times)])
    
    cov_arr[:] = cov_interp(times)
    
    for i in range(len(times)):
        cov_arr[i] = np.roll(cov_arr[i], i)
        
    for i in range(len(times)):
        for j in range(i, len(times)-1):
            cov_arr[j,i] = cov_arr[i,j]
            
    return cov_arr


def simulate_grf_using_cov(cov, max_time, time_step, num_samples, means=0):
    
    print("creating covariance")
    cov_arr = square_cov(cov, max_time, time_step=time_step)
    
    print("creating samples")
    samples = np.random.multivariate_normal([means]*len(cov_arr[0]), cov_arr, size=num_samples)
    
    return samples


def interp_samples(samples, max_time, step_samples):
    
    new_samples = []
    
    time_samples = np.arange(0, len(samples[0])) * step_samples
    new_times = np.arange(0, max_time)
    
    for s in samples:
        
        interp = interp1d(time_samples, s, fill_value='extrapolate', kind='linear')
        
        new_samples.append(interp(new_times))
        
    return new_samples


