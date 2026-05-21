import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

def periodogram_variability_score(
    time,
    flux,
    flux_err=None,
    min_period=None,
    max_period=None,
    samples_per_peak=10,
    fap_threshold=1e-3,
    peak_snr_threshold=10,
    min_power_threshold=0.05,
):

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)

    mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        mask &= np.isfinite(flux_err)

    time = time[mask]
    flux = flux[mask]

    if flux_err is not None:
        flux_err = flux_err[mask]

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    if flux_err is not None:
        flux_err = flux_err[order]

    flux = flux - np.nanmedian(flux)

    baseline = time.max() - time.min()
    cadence = np.nanmedian(np.diff(time))

    if min_period is None:
        min_period = 2 * cadence

    if max_period is None:
        max_period = baseline

    min_frequency = 1 / max_period
    max_frequency = 1 / min_period

    ls = LombScargle(
        time,
        flux,
        flux_err,
        fit_mean=True,
        center_data=True,
        normalization="standard",
    )

    frequency, power = ls.autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=samples_per_peak,
    )

    period = 1 / frequency

    best_idx = np.argmax(power)
    best_power = power[best_idx]
    best_period = period[best_idx]
    best_frequency = frequency[best_idx]

    # Exclude the main peak region when estimating background
    log_period = np.log(period)
    log_best_period = np.log(best_period)

    background_mask = np.abs(log_period - log_best_period) > 0.08

    background_power = power[background_mask]

    median_background = np.nanmedian(background_power)

    mad = np.nanmedian(np.abs(background_power - median_background))
    robust_background_sigma = 1.4826 * mad

    if robust_background_sigma == 0:
        peak_snr = np.inf
    else:
        peak_snr = (best_power - median_background) / robust_background_sigma

    if median_background > 0:
        peak_power_ratio = best_power / median_background
    else:
        peak_power_ratio = np.inf

    try:
        fap = ls.false_alarm_probability(
            best_power,
            minimum_frequency=min_frequency,
            maximum_frequency=max_frequency,
            samples_per_peak=samples_per_peak,
            method="baluev",
        )
    except Exception:
        fap = np.nan

    # Flag peaks too close to the search boundaries
    edge_fraction = 0.02
    near_low_period_edge = best_idx > (1 - edge_fraction) * len(power)
    near_high_period_edge = best_idx < edge_fraction * len(power)
    peak_near_search_edge = near_low_period_edge or near_high_period_edge

    is_periodic_variable = (
        best_power >= min_power_threshold
        and peak_snr >= peak_snr_threshold
        and fap <= fap_threshold
        and not peak_near_search_edge
    )

    return {
        "period": period,
        "frequency": frequency,
        "power": power,
        "best_period": best_period,
        "best_frequency": best_frequency,
        "best_power": best_power,
        "false_alarm_probability": fap,
        "peak_snr": peak_snr,
        "peak_power_ratio": peak_power_ratio,
        "median_background_power": median_background,
        "background_power_sigma": robust_background_sigma,
        "peak_near_search_edge": peak_near_search_edge,
        "is_periodic_variable": is_periodic_variable,
    }


def make_periodogram(
    time,
    flux,
    flux_err=None,
    min_period=None,
    max_period=None,
    samples_per_peak=10,
    plot=True,
    maxProbAllows = 1e-3
):

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)

    # Remove NaNs or infinities
    mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        mask &= np.isfinite(flux_err)

    time = time[mask]
    flux = flux[mask]

    if flux_err is not None:
        flux_err = flux_err[mask]

    # Sort by time
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    if flux_err is not None:
        flux_err = flux_err[order]

    # Normalize flux
    flux = flux - np.nanmedian(flux)

    baseline = time.max() - time.min()
    cadence = np.nanmedian(np.diff(time))

    if min_period is None:
        min_period = 2 * cadence

    if max_period is None:
        max_period = baseline

    min_frequency = 1 / max_period
    max_frequency = 1 / min_period

    ls = LombScargle(time, flux, flux_err)

    frequency, power = ls.autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=samples_per_peak,
    )

    period = 1 / frequency

    best_index = np.argmax(power)
    best_frequency = frequency[best_index]
    best_period = period[best_index]
    best_power = power[best_index]

    # INCLUDE METRIC TO DECIDE IF PERIODIC OR NOT!
    metrics = periodogram_variability_score(time, flux)

    # print("Best period:", metrics["best_period"])
    # print("Best power:", metrics["best_power"])
    # print("FAP:", metrics["false_alarm_probability"])
    # print("Peak SNR:", metrics["peak_snr"])
    # print("Peak/background ratio:", metrics["peak_power_ratio"])
    # print("Variable?", metrics["is_periodic_variable"])

    # variable = (
    #     metrics["false_alarm_probability"] < 1e-3
    #     and metrics["peak_snr"] > 10
    #     and metrics["best_power"] > 0.05
    # )
    # print('False Alarm Probaility')
    # print(metrics["false_alarm_probability"])

    if metrics["false_alarm_probability"] > maxProbAllows:
        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(period, power)
            plt.axvline(best_period, linestyle="--", label=f"Best period = {best_period:.5g}")
            plt.xscale("log")
            plt.xlabel("Period")
            plt.ylabel("Lomb-Scargle Power")
            plt.title("Periodogram")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return({
        "frequency": frequency,
        "period": period,
        "power": power,
        "best_period": best_period,
        "best_frequency": best_frequency,
        "best_power": best_power,
    }, metrics)