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
from scipy.special import erf


def probability_tails(x1, x2):

    return 0.5 * erf(x1/np.sqrt(2)) + 0.5 * erf(x2/np.sqrt(2))


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


def battery_power(
        delta_freq,
        service_power,
        max_discharge_rate=5,
        max_charge_rate=4,
        charge_efficiency=0.9,
        max_cap=4
):
    
    max_percent = max_charge_rate / service_power
    min_percent = -max_discharge_rate / service_power
    
    if max_percent > 1.0:
        max_percent = 1.0
    if min_percent < -1.0:
        min_percent = -1.0

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


def add_charge_info(
        input_df,
        service_power,
        delta_time=1,
        max_discharge_rate=5,
        max_charge_rate=4,
        charge_efficiency=0.9,
        max_cap=4
):

    df = input_df.copy()
    
    df["battery_charge_percent"] = [
        battery_power(
            dfi,
            service_power,
            max_discharge_rate,
            max_charge_rate,
            charge_efficiency,
            max_cap)
        for dfi in df["delta_freq"]
    ]
    df["battery_charge"] = service_power * df["battery_charge_percent"]
    df["battery_charge_low"] = np.where(df["delta_freq"] <= 0.015, service_power * df["battery_charge_percent"], 0.0)
    df["battery_charge_high"] = np.where(df["delta_freq"] >= 0.015, service_power * df["battery_charge_percent"], 0.0)
    df['culm_charge'] = df["battery_charge"].cumsum() * delta_time/3600.0 
    df['culm_charge_low'] = df["battery_charge_low"].cumsum() * delta_time/3600.0
    df['culm_charge_high'] = df["battery_charge_high"].cumsum() * delta_time/3600.0
    
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


def calc_gaussian_slope(
        sigma,
        service_power,
        max_discharge_rate=5,
        max_charge_rate=4,
        charge_efficiency=0.9,
        max_cap=4
):
    
    delta_freq = np.linspace(-1.5, 1.5, num=200)
    y = np.array(
        [
            battery_power(
                deltaf,
                service_power,
                max_discharge_rate,
                max_charge_rate,
                charge_efficiency,
                max_cap

            ) for deltaf in delta_freq])
    
    charge = np.trapz(
        (
            normal_dist(delta_freq, sigma) * y
        ),
        delta_freq
    )
    
    return charge * service_power / 3600


def calc_fitted_slope(
        pdf,
        bin_cents,
        service_power,
        max_discharge_rate=5,
        max_charge_rate=4,
        charge_efficiency=0.9,
        max_cap=4
):
    
    delta_freq = bin_cents#np.linspace(-1.5, 1.5, num=2000)
    y = np.array(
        [
            battery_power(
                deltaf,
                service_power,
                max_discharge_rate,
                max_charge_rate,
                charge_efficiency,
                max_cap
            ) for deltaf in delta_freq
        ]
    )
    
    interped_pdf = interp1d(bin_cents, pdf, fill_value='extrapolate',  kind='linear')
    
    charge = np.trapz(
        (
            interped_pdf(delta_freq) * y
        ),
        delta_freq
    )
    
    return charge * service_power / 3600


def calc_data_slope(
        input_df,
        service_power,
        bins=200,
        max_discharge_rate=5,
        max_charge_rate=4,
        charge_efficiency=0.9,
        max_cap=4
):

    df = input_df.copy()
    
    p, x = np.histogram(df["delta_freq"], bins=bins, density=True)
    
    delta_freq = 0.5 * (x[1:] + x[:-1])
    
    y = np.array(
        [
            battery_power(
                deltaf,
                service_power,
                max_discharge_rate,
                max_charge_rate,
                charge_efficiency,
                max_cap
            ) for deltaf in delta_freq
        ]
    )
    
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


def temporal_covariance(input_df, max_sep=None, step=100, verbose=True):

    df = input_df.copy()
    
    cov_work = df[["delta_freq"]]
    
    deltas = df[["delta_freq"]]
    deltas["shift_freq"] = df["delta_freq"]
    
    cov = {}
    
    if max_sep:
        num = max_sep
    else:
        num = len(deltas)-1

    if verbose:
    
        for i in tqdm(range(0, num, step)):
        
            if not i % step:
                cov[i] = deltas.cov().iloc[0,1]
            
                deltas["shift_freq"] = deltas["shift_freq"].shift(step)
                deltas = deltas.iloc[step:]

    else:
        
        for i in range(0, num, step):
            
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
    
    cov_arr = square_cov(cov, max_time, time_step=time_step)
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


def create_continuous_timeseries(input_df):

    df = input_df.dropna()

    df.index = np.arange(len(df))

    drops = [i for i in range(698400, 698400 + 1 + 86400 - 72001)] + [i for i in range(842400, 842400 + 2 + 2*(86400 - 28801))]

    df = df.drop(drops)

    df.index = np.arange(len(df))

    return df


def simulations_for_anaylsis(
        real_space_cov,
        service_powers,
        nreals,
        time_step,
        num_days,
        block_length,
        max_discharge_rate=5,
        max_charge_rate=4,
        charge_efficiency=0.9,
        max_cap=4
):

    num_blocks = np.ceil(num_days * 24 * 3600 / float(block_length))

    reals = {}

    summaries = {}

    num_samples = [nreals, int(num_blocks)]

    for p in service_powers:

        print(f"creating simulation for service power {p}")
    
        samples = simulate_grf_using_cov(real_space_cov, max_time=10800, time_step=time_step, num_samples=num_samples)
    
        new_samples = [s.flatten() for s in samples]
    
        #new_samples = utils.interp_samples(samples, max_time=num_seconds, step_samples=time_step)
    
        count = 0
    
        reals[p] = {}

        for s in new_samples:
        
            df = pd.DataFrame()
            df["date"] = np.arange(len(s))*time_step
            df["delta_freq"] = s
            df = add_charge_info(
                df,
                p,
                time_step,
                max_discharge_rate,
                max_charge_rate,
                charge_efficiency,
                max_cap
            )

            reals[p][count] = df
        
            count+=1
        
        summaries[p] = reals[p][0].copy()
        summaries[p]['culm_charge'] = 0.0
        summaries[p]['culm_charge_sq'] = 0.0

        for s in reals[p]:
        
            summaries[p]['culm_charge'] += reals[p][s]['culm_charge']
            summaries[p]['culm_charge_sq'] += reals[p][s]['culm_charge']**2.0

        summaries[p]['culm_charge'] = summaries[p]['culm_charge'] / len(reals[p])
    
        summaries[p]['var_charge'] = (
            summaries[p]['culm_charge_sq'] / (len(reals[p]) - 1)
            - summaries[p]['culm_charge']**2.0 * len(reals[p]) / (len(reals[p]) - 1)
        )
    
        summaries[p]['std_charge'] = np.sqrt(summaries[p]['var_charge'])

    return summaries, reals


def max_date_func(E0, reals, p, max_cap, service='both'):
    
    date_end = []

    if service != 'both':
        tag = '_' + service
    else:
        tag = ''

    for s in reals[p]:

        charge = E0 + np.array(reals[p][s][f"culm_charge{tag}"])

        min_date = np.where(charge < 0.0)[0]
        if len(min_date):
            min_date = np.min(min_date)
        else:
            min_date = 1e10
        max_date = np.where(charge > max_cap)[0]
        if len(max_date):
            max_date = np.min(max_date)
        else:
            max_date = 1e10
        date_end.append(np.min([min_date, max_date]))
        
    return np.array(date_end)


def find_E0_for_max_time(
        reals,
        p,
        max_cap,
        min_charge_lim,
        max_charge_lim,
        num_charge_steps,
        poly_deg
):

    E_arr = np.linspace(min_charge_lim, max_charge_lim, num=num_charge_steps)

    d_mean = []
    
    for e in E_arr:

        dates = max_date_func(e, reals, p, max_cap)

        d_mean.append(np.mean(dates))

    res = np.polyfit(E_arr, d_mean, deg=poly_deg)

    Ebest = E_arr[np.where(np.poly1d(res)(E_arr)==np.max(np.poly1d(res)(E_arr)))[0]]

    return Ebest, d_mean, E_arr


def service_power_loop(
        reals,
        service_powers,
        max_cap,
        time_step,
        min_charge_lim,
        max_charge_lim,
        num_charge_steps,
        service='both',
        poly_deg=8,
        fit_conf=0.95,
        verbose=True
):

    results = {}

    for p in service_powers:

        results[p] = {}

        if service == 'both':
            

            E_best, d_mean, E_arr = find_E0_for_max_time(
                reals,
                p,
                max_cap,
                min_charge_lim,
                max_charge_lim,
                num_charge_steps,
                poly_deg
            )
            
        elif service=='high':
            
            E_best = [0]
            
        elif service=='low':
            
            E_best = [max_cap]
            
        dates = max_date_func(E_best, reals, p, max_cap, service)*time_step /(3600.0 * 24.0)

        dates_sort = np.sort(dates)

        sig_lower = dates_sort[int(len(dates) * (1-fit_conf)/2.0)]
        sig_upper = dates_sort[int(len(dates) * (1 - (1-fit_conf)/2.0))]

        results[p]["E_best"] = E_best
        results[p]["exp_date"] = np.mean(dates)
        results[p]["date_lower"] = sig_lower
        results[p]["date_higher"] = sig_upper
        results[p]["dates"] = dates

        if verbose:
        
            print(f"For service power {p} MW")
            print(f"\tThe expected date of termination is {np.round(np.mean(dates), 2)} days")
            print(f"\tThe {100*fit_conf}% confidence interval is {np.round(sig_lower, 2)} - {np.round(sig_upper, 2)} days")
            print(f"\tWith initial charge = {E_best[0]} MWh")

    return results
