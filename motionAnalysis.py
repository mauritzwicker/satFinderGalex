import matplotlib.pyplot as plt
import numpy as np


def circular_orbit_speed(altitude_km):
    """
    Orbital speed for a circular Earth orbit.
    """
    mu_earth = 398600.4418   # km^3 / s^2
    r_earth = 6378.137       # km

    r_orbit = r_earth + np.asarray(altitude_km)
    v_km_s = np.sqrt(mu_earth / r_orbit)

    return v_km_s


def fit_radecmotion(xi, yi):
    # linear fit: y = m*x + b
    m_radec, b_radec = np.polyfit(xi, yi, 1)
    y_fit_radec = m_radec * xi + b_radec
    m_radec_arcsec = m_radec * 60 * 60
    return(m_radec_arcsec)


def plot_radec_motion(m_ra_arcsec, m_dec_arcsec, distSat, moonDist_km, geostat_sats):
    kmSec_RA = distSat * np.tan(np.deg2rad(np.abs(m_ra_arcsec) / 60 / 60))
    kmh_RA = kmSec_RA * 60 * 60
    kmSec_Dec = distSat * np.tan(np.deg2rad(np.abs(m_dec_arcsec) / 60 / 60))
    kmh_Dec = kmSec_Dec * 60 * 60

    fig, ax = plt.subplots()
    ax.plot(distSat, kmh_RA, label='in RA')
    ax.plot(distSat, kmh_Dec, label='in Dec')
    ax.set_xlabel('Distance to Satellite from GALEX [km]')
    ax.set_ylabel('Speed of Satellite [km/h]')
    ax.axvline(moonDist_km/4, label='1/4 of Moon Distance', ls='dashed')
    ax.axvline(geostat_sats, label='Geostationary Orbit', ls='dashed', color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Assuming Perp Motion [ie. lower limits]')
    ax.legend()
    fig.tight_layout()
    plt.show()
    return

def satelliteMotion(df_final_cand0):
    df_final_cand0 = df_final_cand0.sort_values('time').reset_index(drop=True)
    x = df_final_cand0["time"].astype(float).values
    y_ra = df_final_cand0["ra"].astype(float).values
    y_dec = df_final_cand0["dec"].astype(float).values

    d_GALEX = 700 # Km
    # Real Distance = Distance * tan(theta)
    moonDist_km = 384400 - d_GALEX
    geostat_sats = 35786 - d_GALEX
    maxDistPlot = 500e3
    distSat = np.logspace(0, np.log10(maxDistPlot), 1000) # km
    altSat = distSat + d_GALEX


    # # linear fit: y = m*x + b
    # m_ra, b_ra = np.polyfit(x, y_ra, 1)
    # y_fit_ra = m_ra * x + b_ra
    # m_ra_arcsec = m_ra * 60 * 60

    # m_dec, b_dec = np.polyfit(x, y_dec, 1)
    # y_fit_dec = m_dec * x + b_dec
    # m_dec_arcsec = m_dec * 60 * 60

    m_ra_arcsec = fit_radecmotion(x, y_ra)
    m_dec_arcsec = fit_radecmotion(x, y_dec)

    print('dRA/dt = {0:.3f} [arcsec / second]'.format(m_ra_arcsec))
    print('dRA/dt = {0:.3f} [arcsec / second]'.format(m_dec_arcsec))

    plot_radec_motion(m_ra_arcsec, m_dec_arcsec, distSat, moonDist_km, geostat_sats)


    # def satelliteMotion_raDec(df_final_cand0):

    # Time
    t_sec = x - x[0]
    # Coordinates in radians
    ra_rad = np.unwrap(np.deg2rad(df_final_cand0["ra"].astype(float).values))
    dec_rad = np.deg2rad(df_final_cand0["dec"].astype(float).values)

    # Linear fits in rad/s
    m_ra_rad_s, b_ra = np.polyfit(t_sec, ra_rad, 1)
    m_dec_rad_s, b_dec = np.polyfit(t_sec, dec_rad, 1)

    # Representative declination
    dec0 = np.mean(dec_rad)

    # Correct RA component on sky
    mu_ra_rad_s = m_ra_rad_s * np.cos(dec0)
    mu_dec_rad_s = m_dec_rad_s

    # Total angular speed on sky
    mu_rad_s = np.sqrt(mu_ra_rad_s**2 + mu_dec_rad_s**2)
    mu_arcsec_s = mu_rad_s * 206264.806247

    print(f"dRA/dt raw      = {m_ra_rad_s * 206264.806247:.3f} arcsec/s")
    print(f"dRA/dt on sky   = {mu_ra_rad_s * 206264.806247:.3f} arcsec/s")
    print(f"dDec/dt         = {mu_dec_rad_s * 206264.806247:.3f} arcsec/s")
    print(f"Total sky speed = {mu_arcsec_s:.3f} arcsec/s")

    D_geo_km = 35786  # altitude only; approximate
    v_geo_km_s = D_geo_km * mu_rad_s
    v_geo_km_h = v_geo_km_s * 3600

    print(f"Speed at D={D_geo_km:.0f} km: {v_geo_km_s:.3f} km/s")
    print(f"Speed at D={D_geo_km:.0f} km: {v_geo_km_h:.1f} km/h")

    v_km_s = distSat * mu_rad_s
    v_km_h = v_km_s * 3600.0

    # # For a satellite what are its velocities
    v_km_s_sat = circular_orbit_speed(altSat)
    v_km_h_sat = v_km_s_sat * 3600

    # Intersection
    # -> What we think the minimum speed / distance is
    # Difference between the two plotted curves
    diff = v_km_h - v_km_h_sat
    # Indices where the sign changes
    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    x_intersections = []
    y_intersections = []
    for i in idx:
        x0, x1 = distSat[i], distSat[i+1]
        y0, y1 = diff[i], diff[i+1]

        # Linear interpolation to find x where diff = 0
        x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)

        # Interpolate the speed at that x value
        y_cross = np.interp(x_cross, distSat, v_km_h)

        x_intersections.append(x_cross)
        y_intersections.append(y_cross)

    x_intersections = np.array(x_intersections)
    y_intersections = np.array(y_intersections)

    print("Intersections:")
    for x, y in zip(x_intersections, y_intersections):
        print(f"Distance/altitude = {x:.2f} km, speed = {y:.2f} km/h")

    fig, ax = plt.subplots()
    ax.plot(distSat, v_km_h, label="total transverse speed")
    ax.plot(distSat, v_km_h_sat, label="circular orbital speed")

    ax.set_xlabel("Distance to satellite from GALEX [km]")
    ax.set_ylabel("Transverse speed [km/h]")
    ax.axvline(moonDist_km/4, label='1/4 of Moon Distance', ls='dashed')
    ax.axvline(geostat_sats, label='Geostationary Orbit', ls='dashed', color='red')
    ax.axvline(x_intersections, ls = '-.', color='purple', label='Estimated Position')
    ax.axhline(y_intersections, ls = '-.', color='purple')

    ax.set_xlim(1, 500e3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    plt.show()

    return