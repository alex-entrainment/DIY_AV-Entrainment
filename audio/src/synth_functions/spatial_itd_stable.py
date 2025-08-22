
import numpy as np
import numba

@numba.njit(cache=True, fastmath=True)
def _coef_ms(tau_ms: float, fs: float) -> float:
    if tau_ms <= 0.0:
        return 0.0
    a = np.exp(-1.0 / (fs * (tau_ms * 1e-3)))
    if a < 0.0:
        a = 0.0
    if a > 0.999999:
        a = 0.999999
    return a

@numba.njit(cache=True, fastmath=True)
def _smooth(y_prev: float, x_t: float, a: float) -> float:
    return a * y_prev + (1.0 - a) * x_t

@numba.njit(cache=True, fastmath=True)
def _unwrap_deg(prev_deg: float, cur_deg: float) -> float:
    d = cur_deg - prev_deg
    while d >= 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return prev_deg + d

@numba.njit(cache=True, fastmath=True)
def _lagrange3(y3, y2, y1, y0, mu):
    m = mu
    m1 = 1.0 - m
    c0 = -m * (m1) * (m1) / 6.0
    c1 = (1.0 + m) * (m1) * (m1) / 2.0
    c2 = (m) * (1.0 + m) * (m1) / 2.0
    c3 = -m * (m) * (m1) / 6.0
    return c0 * y3 + c1 * y2 + c2 * y1 + c3 * y0

@numba.njit(cache=True, fastmath=True)
def stereo_itd_headmodel_stable(
    x: np.ndarray,
    fs: float,
    theta_deg_in: np.ndarray,
    distance_m_in: np.ndarray,
    head_radius_m: float = 0.0875,
    itd_scale: float = 1.0,
    ild_max_db: float = 1.5,
    ild_enable: int = 1,
    ref_distance_m: float = 1.0,
    rolloff: float = 1.0,
    min_distance_m: float = 0.1,
    dz_theta_ms: float = 40.0,
    dz_dist_ms: float = 80.0,
    max_deg_per_s: float = 90.0,
    max_delay_step_samples: float = 0.02,
    interp_mode: int = 1
):
    """Stable ITD panner with slew limiting and optional ILD."""
    N = x.shape[0]
    L = np.empty(N, np.float32)
    R = np.empty(N, np.float32)

    a_th = _coef_ms(dz_theta_ms, fs)
    a_di = _coef_ms(dz_dist_ms, fs)

    theta_deg = np.empty(N, np.float64)
    dist_m = np.empty(N, np.float64)

    prev_th = theta_deg_in[0]
    theta_deg[0] = prev_th
    for n in range(1, N):
        unw = _unwrap_deg(prev_th, theta_deg_in[n])
        max_step = max_deg_per_s / fs
        step = unw - prev_th
        if step > max_step:
            step = max_step
        if step < -max_step:
            step = -max_step
        cur = prev_th + step
        theta_deg[n] = cur
        prev_th = cur

    th = theta_deg[0]
    di = distance_m_in[0] if distance_m_in[0] > min_distance_m else min_distance_m
    theta_rad = np.empty(N, np.float64)
    distance_m = np.empty(N, np.float64)
    for n in range(N):
        th = _smooth(th, theta_deg[n], a_th)
        di = _smooth(di, distance_m_in[n] if distance_m_in[n] > min_distance_m else min_distance_m, a_di)
        theta_rad[n] = th * np.pi / 180.0
        distance_m[n] = di

    c = 343.0
    max_itd_s = (head_radius_m / c) * abs(itd_scale)
    max_d_samps = int(np.ceil(max_itd_s * fs)) + 8
    if max_d_samps < 12:
        max_d_samps = 12

    bufL = np.zeros(max_d_samps, np.float32)
    bufR = np.zeros(max_d_samps, np.float32)
    w = 0

    d_samples = 0.0
    side = 0
    gL = 1.0
    gR = 1.0

    for n in range(N):
        s = np.sin(theta_rad[n])
        tau = (head_radius_m / c) * s * itd_scale

        d_target = abs(tau) * fs
        side_t = 1 if tau > 0.0 else (-1 if tau < 0.0 else 0)

        dd = d_target - d_samples
        if dd > max_delay_step_samples:
            dd = max_delay_step_samples
        if dd < -max_delay_step_samples:
            dd = -max_delay_step_samples
        d_samples += dd

        if d_samples > 1e-6:
            side = side_t
        else:
            side = 0
            d_samples = 0.0

        d = distance_m[n]
        if d < ref_distance_m:
            gain = 1.0
        else:
            gain = (ref_distance_m / d) ** rolloff

        base = (w - 1) % max_d_samps

        if side == 1:
            riL = (base - d_samples) % max_d_samps
        else:
            riL = base * 1.0
        r0L = int(np.floor(riL))
        r1L = (r0L - 1) % max_d_samps
        fracL = riL - r0L

        if side == -1:
            riR = (base - d_samples) % max_d_samps
        else:
            riR = base * 1.0
        r0R = int(np.floor(riR))
        r1R = (r0R - 1) % max_d_samps
        fracR = riR - r0R

        if interp_mode == 0:
            yL = (1.0 - fracL) * bufL[r0L] + fracL * bufL[r1L]
            yR = (1.0 - fracR) * bufR[r0R] + fracR * bufR[r1R]
        else:
            r2L = (r1L - 1) % max_d_samps
            r3L = (r2L - 1) % max_d_samps
            r2R = (r1R - 1) % max_d_samps
            r3R = (r2R - 1) % max_d_samps
            yL = _lagrange3(bufL[r3L], bufL[r2L], bufL[r1L], bufL[r0L], fracL)
            yR = _lagrange3(bufR[r3R], bufR[r2R], bufR[r1R], bufR[r0R], fracR)

        if ild_enable != 0:
            ild_db = ild_max_db * 0.5 * s
            gL_t = 10.0 ** (+ild_db / 20.0)
            gR_t = 10.0 ** (-ild_db / 20.0)
            gL = _smooth(gL, gL_t, a_th)
            gR = _smooth(gR, gR_t, a_th)

        bufL[w] = x[n]
        bufR[w] = x[n]
        w = (w + 1) % max_d_samps

        L[n] = yL * gL * gain
        R[n] = yR * gR * gain

    out = np.empty((N, 2), np.float32)
    out[:, 0] = L
    out[:, 1] = R
    return out
