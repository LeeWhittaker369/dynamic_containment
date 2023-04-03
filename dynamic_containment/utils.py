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
from scipy.stats import norm


def probability_tails(x1, x2):
    """The probablity that a value lands outside of two values
    of a normal distribution.

    Args:
        x1: The lower value, provided as the ratio x/sigma.
        x2: The upper value, provided as the ratio x/sigma

    Returns:
        The probability.

    """

    return 0.5 * erf(x1/np.sqrt(2)) + 0.5 * erf(x2/np.sqrt(2))


def hist_fit_2gauss(x, bin_cents, hist):
    """Creates an array of residuals between frequency model
       from two normal distribution joined together and a histogram.

    Args:
        x1: The position of peak of each normal distribution on
            either side of zero.
        bin_cents: The centres of each bin
        hist: The 

    Returns:
        The residuals.

    """
    
    gauss1 = norm.pdf(bin_cents, -x[0], x[1])
    gauss2 = norm.pdf(bin_cents, x[0], x[1])
    
    gauss = np.where(bin_cents<0, gauss1, gauss2)
    gauss = gauss / np.trapz(gauss, bin_cents)
    
    return gauss-hist


def hist_fit_gauss(x, bin_cents, hist):
    """Creates an array of residuals between frequency model
       from a normal distribution and a histogram.

    Args:
        x: The position of peak of each normal distribution on
            either side of zero.
        bin_cents: The centres of each bin
        hist: The 

    Returns:
        The residuals.

    """
    
    gauss = norm.pdf(bin_cents, x[0], x[1])
    gauss = gauss / np.trapz(gauss, bin_cents)
    
    return gauss-hist


def battery_power(
        delta_freq,
        service_power,
        max_discharge_rate=5,
        max_charge_rate=4,
        charge_efficiency=0.9,
        max_cap=4
):
    """The model of how battery charge responds to frequncy changes.

    Args:
        delta_freq: frequency-50 Hz.
        service_power: The contracted service power 
        max_discharge_rate: The maximum discharge rate of the battery (MW)
        max_charge_rate: The maximum charge rate of the battery (MW)
        charge_efficiency: The efficiency of the battery when charging
        max_cap: The maximum capacity of the battery (MWh) 

    Returns:
        The change in charge of the batter caused by a response
        to the frequency (MW).

    """
    
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
    """Adds the temporal cumulative charge information to a dataframe.

    Args:
        input dataframe: A data frame containing delta_freq.
        service_power: The contracted service power
        delta_time: The size of the time step (could be removed by using the date column)
        max_discharge_rate: The maximum discharge rate of the battery (MW)
        max_charge_rate: The maximum charge rate of the battery (MW)
        charge_efficiency: The efficiency of the battery when charging
        max_cap: The maximum capacity of the battery (MWh) 

    Returns:
        The change in charge of the batter caused by a response
        to the frequency (MW).

    """

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
    """Reads in the example data and changes the column names.

    Args:
        input dataframe: Path to the data.

    Returns:
        The example frequency data

    """
    
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
    """Calculates the specral density of a time series.

    Args:
        df: The input dataframe.
        fs: sampling frequency

    Returns:
        The spectra density

    """
    
    x, power = signal.welch(df.delta_freq, fs=fs)
    
    return x, power


def simulate_delta_f(delta_f):
    """Uses the spectra density to create Gaussian random simulations.

    Args:
        delta_f: The delta_f data.

    Returns:
        The simulated datasets

    """
    
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


def calc_gaussian_slope(
        sigma,
        service_power,
        max_discharge_rate=5,
        max_charge_rate=4,
        charge_efficiency=0.9,
        max_cap=4
):
    """Calculates the rate of change of charge assuming uncorrelated Gaussian noise.

    Args:
        sigma: The standard deviation of the noise.
        service_power: The contracted service power
        max_discharge_rate: The maximum discharge rate of the battery (MW)
        max_charge_rate: The maximum charge rate of the battery (MW)
        charge_efficiency: The efficiency of the battery when charging
        max_cap: The maximum capacity of the battery (MWh) 

    Returns:
        The rate of change of charge

    """
    
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
            norm.pdf(delta_freq, 0.0, sigma) * y
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
    """Calculates the rate of change of energy given a binned pdf

    Args:
        pdf: The probablity density binned.
        bin_cents: The bin centres
        service_power: The contracted service power
        max_discharge_rate: The maximum discharge rate of the battery (MW)
        max_charge_rate: The maximum charge rate of the battery (MW)
        charge_efficiency: The efficiency of the battery when charging
        max_cap: The maximum capacity of the battery (MWh) 

    Returns:
        The rate of change of energy

    """
    
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
    """Calculates the rate of change of energy given a dataframe with delta_freq

    Args:
        input_df: The input dataframe
        service_power: The contracted service power
        max_discharge_rate: The maximum discharge rate of the battery (MW)
        max_charge_rate: The maximum charge rate of the battery (MW)
        charge_efficiency: The efficiency of the battery when charging
        max_cap: The maximum capacity of the battery (MWh) 

    Returns:
        The rate of change of energy

    """

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


def generate_sample_delta_f(pdf, bin_cents, num_samps):
    """Genarates sample delta frequencies using a general pdf

    Args:
        pdf: The probablity density binned.
        bin_cents: The bin centres
        num_samps: Number of samples to generate

    Returns:
        Sample delta frequencies

    """
    
    delta_bin = bin_cents[1]-bin_cents[0]
        
    cdf = np.array([np.trapz(pdf[:i], bin_cents[:i]) for i in range(len(bin_cents))])
    cdf = cdf/cdf[-1]
    
    ppf = interp1d(cdf, bin_cents-delta_bin, fill_value='extrapolate',  kind='linear')
    
    our_samps = ppf(np.random.uniform(size=num_samps))
    
    return our_samps


def temporal_covariance(input_df, max_sep=None, step=100, verbose=True): 
    """Calculates the temporal covariance

    Args:
        input_df: the dataframe with delta_freq
        max: the maximum temporal separation (in number of rows in input_df)
        step: The number of steps to sample at
        verbose: Do you want a progress bar?

    Returns:
        The temporal covariance as a dict {Delta t : covariance}

    """

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


def square_cov(cov, max_sep, time_step=1):
    """Turns the covariance from `temporal_covariance` into a square matrix

    Args:
        cov: A covariance dict
        max_sep: the maximum temporal separation
        time_step: The number of seconds to sample at

    Returns:
        The a square numpy array covariance matrix

    """
    
    cov_interp = interp1d(list(cov.keys()), list(cov.values()), fill_value='extrapolate', kind='linear')
    
    times = np.arange(0, max_sep, time_step)
    
    cov_arr = np.zeros([len(times), len(times)])
    
    cov_arr[:] = np.where(times <= np.max(list(cov.keys())), cov_interp(times), 0.0)
    
    for i in range(len(times)):
        cov_arr[i] = np.roll(cov_arr[i], i)
        
    for i in range(len(times)):
        for j in range(i, len(times)-1):
            cov_arr[j,i] = cov_arr[i,j]
            
    return cov_arr


def simulate_grf_using_cov(cov, max_time, time_step, num_samples, means=0):
    """Generates a gaussian random set of random variables using a cov dict from `temporal_covariance`

    Args:
        cov: A covariance dict
        max_time: the maximum time to simulate to
        time_step: The number of seconds to sample at
        num_samples: The number of samples to generate
        means: the mean of the samples

    Returns:
        An array of samples

    """
    
    cov_arr = square_cov(cov, max_time, time_step=time_step)
    samples = np.random.multivariate_normal([means]*len(cov_arr[0]), cov_arr, size=num_samples)
    
    return samples


def create_continuous_timeseries(input_df):
    """Creates a continuous time series. Note, the dates are hardcoded!

    Args:
        input_df: The input dataframe

    Returns:
        Dataframe with continuous timeseries

    """

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
    """Generates the simulated delta_freqs for an arbitrary number of realisations

    Args:
        real_space_cov: A covariance dict
        service_powers: The contracted service powers as an array
        nreals: The number of realisations
        time_step: The time steps at which to generate samples
        num_days: The number of days of data to simulate
        block_length: The number of seconds to consider in independent blocks
        max_discharge_rate: The maximum discharge rate of the battery (MW)
        max_charge_rate: The maximum charge rate of the battery (MW)
        charge_efficiency: The efficiency of the battery when charging
        max_cap: The maximum capacity of the battery (MWh)

    Returns:
        A dictionary of summary statistics of the cumulative charge at each time step 
            for each service power: mean, variance, and standard deviation
        An dictionary containing the dataframes for each realisation for each service power

    """
    
    num_blocks = np.ceil(num_days * 24 * 3600 / float(block_length))

    reals = {}

    summaries = {}

    num_samples = [nreals, int(num_blocks)]

    for p in service_powers:

        print(f"creating simulation for service power {p}")
    
        samples = simulate_grf_using_cov(real_space_cov, max_time=block_length, time_step=time_step, num_samples=num_samples)
    
        new_samples = [s.flatten() for s in samples]
    
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
    """Finds the date at which the battery hits max energy or empty, given an inital energy and
       set of simulated charge data

    Args:
        E0: The inital energy of the battery
        reals: A dictionary of realisations from 'simulations_for_anaylsis'
        p: The serive power
        max_cap: The maximum capacity of the battery (MWh)
        service: The flavour of service - 'both', 'high', or 'low'

    Returns:
        An array of dates for each realisation

    """
    
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
        min_energy_lim,
        max_energy_lim,
        num_energy_steps,
        poly_deg
):
    """Finds the initial energy that maximises the time taken to either
       run out of energy or hit maximum capacity 

    Args:
        reals: A dictionary of realisations 'from simulations_for_anaylsis'
        p: The service power
        max_cap: The maximum capacity of the battery (MWh)
        min_energy_lim: The lower limit of the initial energy to sample
        max_energy_lim: The upper limit of the initial energy to sample
        num_energy_steps: The number of steps to sample the energy between
            min_energy_lim and max_energy_lim
        poly_deg: The degree of polynomial to fit to the energy and lifetime

    Returns:
       The optimal initial energy
       The mean termination date from the simulations
       The array og initial energies sampled

    """

    E_arr = np.linspace(min_energy_lim, max_energy_lim, num=num_energy_steps)

    d_mean = []
    
    for e in E_arr:

        dates = max_date_func(e, reals, p, max_cap)

        d_mean.append(np.mean(dates))

    res = np.polyfit(E_arr, d_mean, deg=poly_deg)

    Ebest = E_arr[np.where(np.poly1d(res)(E_arr)==np.max(np.poly1d(res)(E_arr)))[0]][0]

    return Ebest, d_mean, E_arr


def service_power_loop(
        reals,
        service_powers,
        max_cap,
        time_step,
        min_energy_lim,
        max_energy_lim,
        num_energy_steps,
        service='both',
        poly_deg=8,
        fit_conf=0.95,
        verbose=True
):
    """Loops through a list of service powers and finds the optimal initial energy in each case.
       Also provides the confidence interval of the termination date with this initial energy

    Args:
        reals: A dictionary of realisations 'from simulations_for_anaylsis'
        service_power: The contracted service powers as an array
        max_cap: The maximum capacity of the battery (MWh)
        time_step: The time step of the simulations
        min_energy_lim: The lower limit of the initial energy to sample
        max_energy_lim: The upper limit of the initial energy to sample
        num_energy_steps: The number of steps to sample the energy between
            min_energy_lim and max_energy_lim
        service: The flavour of service - 'both', 'high', or 'low'
        poly_deg: The degree of polynomial to fit to the energy and lifetime
        fit_conf: The confidence interval as a decimal - e.g. 0.95 for 95% confidence
        verbose: If True, outputs summary statistics during run time

    Returns:
        A dictionary of results: For each service power, the output is the optimal initial energy,
        the expected termination data and the confidence interval. Also, we output the full list of
        termination dates for each realisation and a dataframe with the main results.

    """

    results = {}

    E_best_list = []
    dates_list = []
    conf_list = []
    service_list = []
    df = pd.DataFrame()

    for p in service_powers:

        results[p] = {}

        if service == 'both':
            

            E_best, d_mean, E_arr = find_E0_for_max_time(
                reals,
                p,
                max_cap,
                min_energy_lim,
                max_energy_lim,
                num_energy_steps,
                poly_deg
            )
            
        elif service=='high':
            
            E_best = 0
            
        elif service=='low':
            
            E_best = max_cap
            
        dates = max_date_func(E_best, reals, p, max_cap, service)*time_step /(3600.0 * 24.0)

        dates_sort = np.sort(dates)

        sig_lower = dates_sort[int(len(dates) * (1-fit_conf)/2.0)]
        sig_upper = dates_sort[int(len(dates) * (1 - (1-fit_conf)/2.0))]

        results[p]["E_best"] = E_best
        results[p]["exp_date"] = np.mean(dates)
        results[p]["date_lower"] = sig_lower
        results[p]["date_upper"] = sig_upper
        results[p]["dates"] = dates

        if verbose:
        
            print(f"For service power {p} MW")
            print(f"\tThe expected date of termination is {np.round(np.mean(dates), 2)} days")
            print(f"\tThe {100*fit_conf}% confidence interval is {np.round(sig_lower, 2)} - {np.round(sig_upper, 2)} days")
            print(f"\tWith initial energy = {E_best[0]} MWh")

        E_best_list.append(E_best)
        dates_list.append("%s days " % (np.round(results[p]["exp_date"],2)))
        conf_list.append("%s - %s days" %(np.round(results[p]["date_lower"], 2), np.round(results[p]["date_upper"], 2)))
        service_list.append("%sMW - %s" % (p, service))
        

    df["Initial Energy / MWh"] = E_best_list
    df["Expected Termination Date"] = dates_list
    df["%s%% Confidence Interval" % (100*fit_conf)] = conf_list

    df.index = service_list
                         

    results["table"] = df

    return results


def loop_max_cap(
        reals,
        max_cap_arr,
        service_power,
        time_step,
        num_energy_steps,
        service="both",
        poly_deg=8,
        fit_conf=0.95
):
    """Finds the optimal termination data and confidence limits for a given service power given an array of maximum capacities

    Args:
        reals: A dictionary of realisations 'from simulations_for_anaylsis'
        max_cap_arr: An array of maximum capacities
        service_power: The service power
        time_step: The time step of the simulations
        num_energy_steps: The number of steps to sample the energy between
            min_energy_lim and max_energy_lim
        service: The flavour of service - 'both', 'high', or 'low'
        poly_deg: The degree of polynomial to fit to the energy and lifetime
        fit_conf: The confidence interval as a decimal - e.g. 0.95 for 95% confidence

    Returns:
        A dictionay containing the optimal energies, the expected termination dates and the 
        confidence limits

    """
    
    results = {}
    results["best_energies"] = np.zeros(len(max_cap_arr))
    results["exp_dates"] = np.zeros(len(max_cap_arr))
    results["lower_limits"] = np.zeros(len(max_cap_arr))
    results["upper_limits"] = np.zeros(len(max_cap_arr))

    for i in range(len(max_cap_arr)):
     
        results_temp = service_power_loop(
            reals,
            [service_power],
            max_cap=max_cap_arr[i],
            time_step=time_step,
            min_energy_lim=max_cap_arr[i]*0.5,
            max_energy_lim=max_cap_arr[i]*0.75,
            num_energy_steps=num_energy_steps,
            service=service,
            poly_deg=poly_deg,
            fit_conf=fit_conf,
            verbose=False,
        )

        results["best_energies"][i] = results_temp[service_power]["E_best"]
        results["exp_dates"][i] = results_temp[service_power]["exp_date"]
        results["lower_limits"][i] = results_temp[service_power]["date_lower"]
        results["upper_limits"][i] = results_temp[service_power]["date_upper"]

    return results
        
