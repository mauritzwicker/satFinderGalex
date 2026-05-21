import numpy as np
import matplotlib.pyplot as plt
import periodogramFcts

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def make_hist_and_fitGauss(yVal, bins_data_Gaussian=20):
    # Fit Gaussian
    counts_ynew, bin_edges_ynew = np.histogram(yVal, bins=bins_data_Gaussian)
    # Convert bin edges to bin centers
    bin_centers_ynew = 0.5 * (bin_edges_ynew[:-1] + bin_edges_ynew[1:])

    amp_guess = counts_ynew.max()
    mu_guess = 0.0
    sigma_guess = np.std(yVal)

    p0 = [amp_guess, mu_guess, sigma_guess]
    try:
        popt, pcov = curve_fit(gaussian, bin_centers_ynew, counts_ynew, p0=p0)
        amp, mu, sigma = popt
        # print("Amplitude:", amp)
        # print("Mean:", mu)
        # print("Sigma:", sigma)
    except:
        print('Unable to fit Gaussian -> continue')
        return(None, None, None, None, None)

    x_fit = np.linspace(bin_edges_ynew[0], bin_edges_ynew[-1], 500)
    y_fit = gaussian(x_fit, *popt)

    return(amp, mu, sigma, x_fit, y_fit)


def smoothTimeseries(x, y, smoothingParam='savgol', window_savgol=11, poly_savgol=3, sig_gaussSmooth=0.5, degParabola_mask=3):
    # smoothingParam='savgol' or 'gaussian
    # Smoothing
    if smoothingParam == 'savgol':
        y_smooth = savgol_filter(y, window_length=window_savgol, polyorder=poly_savgol)
    elif smoothingParam == 'gaussian':
        y_smooth = gaussian_filter1d(y, sigma=sig_gaussSmooth)
    else:
        print('Smoothign Param undefined, no smoothing xD')
        y_smooth = y

    # Fit Baseline without Masking
    # Fit parabola: y = ax^2 + bx + c
    try:
        quad_coeffs = np.polyfit(x, y, deg=degParabola_mask)
        # Evaluate fitted parabola over all x
        baseline_nomask = np.polyval(quad_coeffs, x)

        # y_new = y - baseline_nomask
        # y_new = y_new[np.isfinite(y_new)]
    except:
        baseline_nomask = np.median(y)

        # y_new = y-np.median(y)
        # y_new = y_new[np.isfinite(y_new)]

    return(baseline_nomask)

def run_satFinder(overlap_results, use_OnlyCounts=False, maxRuns=1000, pltIt=False, counts_Min_rel=0.75, coordBin_arcsec=180, Xsig=5, maxProbAllows=1e-3):
    runIt=True
    cntDone = 0
    resultsCandidates= {}

    for grid_key, out in overlap_results.items():
        cube = out["cube"]
        t_edges, ra_edges, dec_edges = out["edges"]

        t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
        ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
        dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])

        n_t, n_ra, n_dec = cube.shape

        print(f"Working on {grid_key}")
        print("cube shape:", cube.shape)

        # Total RA/Dec counts over all time
        radec_distribution = np.sum(cube, axis=0).ravel()
        radec_distribution = radec_distribution[radec_distribution > 0]

        log_distribution = np.log10(radec_distribution)
        med_log_distribution = np.median(log_distribution)
        std_log_distribution = np.std(log_distribution)

        # Convert back from log-counts to counts
        min_total_counts = 10 ** (med_log_distribution - std_log_distribution)

        print("min_total_counts:", min_total_counts)

        candidatesThisGrid = []

        for ira in range(n_ra):
            if runIt == False:
                    continue
            if ira < 10:
                continue

            for idec in range(n_dec):

                if runIt == False:
                    continue
                # Time series for this RA/Dec bin
                counts_ts = cube[:, ira, idec].astype(float)

                # Remove NaN/inf safely using boolean mask
                finite = np.isfinite(counts_ts)
                t_centers_ts = t_centers[finite]
                counts_ts = counts_ts[finite]

                if len(counts_ts) < 1:
                    continue

                total_counts = counts_ts.sum()

                if total_counts <= 0:
                    continue

                if total_counts < min_total_counts:
                    continue

                xdata = np.asarray(t_centers_ts).ravel()

                if use_OnlyCounts:
                    ydata = np.asarray(counts_ts).ravel()

                else:
                    med_counts = np.median(counts_ts)

                    # Important: median can be zero if most time bins are empty
                    if med_counts <= 0 or not np.isfinite(med_counts):
                        continue

                    ydata = np.asarray(counts_ts / med_counts).ravel()

                    # Remove values below min relative counts
                    keep = ydata >= counts_Min_rel
                    xdata = xdata[keep]
                    ydata = ydata[keep]

                # Remove NaN/inf again after division
                finite = np.isfinite(ydata) & np.isfinite(xdata)
                xdata = xdata[finite]
                ydata = ydata[finite]

                if len(xdata) < 1:
                    continue

                baseline_counts = smoothTimeseries(xdata, ydata)
                y_new = ydata - baseline_counts
                x_new = xdata[np.isfinite(y_new)]
                y_new = y_new[np.isfinite(y_new)]

                amp, mu, sigma, x_fit, y_fit = make_hist_and_fitGauss(y_new)
                if amp is None:
                    continue
                # Now we want to see the histogram of the data and find the outliers

                # Now for each original corrected y_new, I want to see if they are > mu + X*sigma
                threshold = mu + Xsig * sigma
                mask_high = y_new > threshold
                high_values_x = x_new[mask_high]
                high_values = y_new[mask_high]
                high_indices = np.where(mask_high)[0]

                if len(high_values) <= 0:
                    continue

                # Make a Periodogram to See if Has this circular orbiting variability
                cand = False
                result, metrics = periodogramFcts.make_periodogram(x_new, y_new, maxProbAllows = 1e-3, plot=pltIt)
                if metrics["false_alarm_probability"] > maxProbAllows:
                    # print('Candfidate')
                    cand = True
                    cntDone +=1
                    if cntDone > maxRuns:
                        runIt=False
                
                # Plot: Extracted time vs count/med(count) and histogram(count/med(count))
                if (cand & pltIt):
                    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                    axs[0].plot(x_new, ydata, color='grey')
                    axs[1].hist(y_new, bins=30, color='grey')
                    fig.suptitle('Binsize = {0:.0f}" x {0:.0f}"'.format(coordBin_arcsec))
                    plt.show()

                if (cand & pltIt):
                    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
                    ax = axs[0]
                    ax.plot(xdata, baseline_counts, label='parabola no mask')

                    ax.plot(xdata, ydata, lw=2, color='black', ls='dashed', zorder=10, alpha=0.3)
                    ax.legend(title='Time Cuts')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Counts/med(Counts)')
                    
                    ax = axs[1]
                    ax.plot(xdata, ydata-baseline_counts, lw=2, color='black', ls='dashed', zorder=10, alpha=0.3)
                    for ix, (xxx, yyy) in enumerate(zip(high_values_x, high_values)):
                        axs[1].axvline(xxx, ls='dashed', color='green', label='Candidate Satellite:  id-{0}'.format(ix), zorder=-1, alpha=0.3)

                    fig.suptitle('Binsize = {0:.0f}" x {0:.0f}"'.format(coordBin_arcsec))
                    plt.show()

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(y_new, bins=30, alpha=0.5, label="y distribution")
                    ax.plot(x_fit, y_fit, "r-", label="Gaussian fit")
                    for xxx in [1, 2, 3, 4, 5]:
                        ax.axvline(xxx*sigma, ls='dotted', color='grey', alpha=0.4)
                    for xxx, ix in zip(high_values, high_indices):
                        ax.axvline(xxx, ls='dashed', color='green', alpha=1.0, label='Candidate Satellite:  id-{0}'.format(ix))
                    ax.set_xlabel('Counts/med(Counts)')
                    ax.legend()
                    fig.suptitle('Binsize = {0:.0f}" x {0:.0f}"'.format(coordBin_arcsec))
                    plt.show()

                if cand:
                    candidatesThisGrid.append([ira, idec, high_values_x, high_values])
        resultsCandidates[grid_key] = candidatesThisGrid
    return(resultsCandidates)