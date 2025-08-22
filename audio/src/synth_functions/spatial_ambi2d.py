import numpy as np
import numba

# Decoder identifiers
DECODER_ITD_HEAD = 0
DECODER_FOA_CARDIOID = 1

# -------------------------
# Helpers / smoothing
# -------------------------
@numba.njit(cache=True, fastmath=True)
def _frac(x): return x - np.floor(x)

@numba.njit(cache=True, fastmath=True)
def _one_pole_coef_ms(tau_ms: float, fs: float) -> float:
    if tau_ms <= 0.0:
        return 0.0
    a = np.exp(-1.0 / (fs * (tau_ms * 1e-3)))
    if a < 0.0: a = 0.0
    if a > 0.999999: a = 0.999999
    return a

@numba.njit(cache=True, fastmath=True)
def _smooth_step(state: float, target: float, a: float) -> float:
    return a * state + (1.0 - a) * target

@numba.njit(cache=True, fastmath=True)
def _smooth_angle(state: float, target: float, a: float) -> float:
    diff = np.arctan2(np.sin(target - state), np.cos(target - state))
    return state + (1.0 - a) * diff

# -------------------------
# FOA encode (2D, SN3D)
# -------------------------
@numba.njit(cache=True, fastmath=True)
def foa_encode_2d(x: np.ndarray, theta_rad: np.ndarray):
    """
    Encode mono x into FOA W,X,Y using SN3D in 2D (elev=0).
    theta_rad: per-sample azimuth (radians), 0=front, +CCW (leftwards).
    """
    N = x.shape[0]
    W = np.empty(N, np.float32)
    X = np.empty(N, np.float32)
    Y = np.empty(N, np.float32)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for n in range(N):
        c = np.cos(theta_rad[n]); s = np.sin(theta_rad[n])
        s_n = x[n]
        W[n] = inv_sqrt2 * s_n
        X[n] = s_n * c
        Y[n] = s_n * s
    return W, X, Y

# -------------------------
# FOA stereo decode (cardioids)
# -------------------------
@numba.njit(cache=True, fastmath=True)
def foa_decode_stereo_cardioid(W, X, Y, ear_angle_deg=30.0):
    """
    Simple stereo decode: two virtual cardioids at ±ear_angle_deg.
    For SN3D 2D: L = W/√2 + X*cos(a) + Y*sin(a); R = W/√2 + X*cos(-a) + Y*sin(-a)
    """
    N = W.shape[0]
    L = np.empty(N, np.float32)
    R = np.empty(N, np.float32)
    a = ear_angle_deg * np.pi / 180.0
    ca, sa = np.cos(a), np.sin(a)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for n in range(N):
        w = W[n]*inv_sqrt2
        x = X[n]; y = Y[n]
        L[n] = w + (x*ca + y*sa)
        R[n] = w + (x*ca - y*sa)  # sin(-a) = -sa
    return L, R

# -------------------------
# ITD/ILD augment (low freq)
# -------------------------
@numba.njit(cache=True, fastmath=True)
def _apply_time_varying_itd_ild(
    L_in, R_in, fs: float,
    theta_rad,                 # azimuth per sample (same used for encode)
    head_radius_m: float = 0.0875,  # ~8.75 cm
    itd_scale: float = 1.0,         # 0..1 (1 = physical)
    ild_max_db: float = 3.0,        # subtle gain toward near ear above xover
    ild_xover_hz: float = 700.0,    # shelf hinge (ignored below this)
    dz_ms: float = 20.0             # dezipper for delay/gain
):
    """
    Adds interaural time delay (fractional) and a gentle ILD shelf that
    increases toward the near ear. Designed for 0-1 kHz material.
    """
    N = L_in.shape[0]
    L = np.empty_like(L_in)
    R = np.empty_like(R_in)

    # Fractional delay lines
    c = 343.0
    max_itd = head_radius_m / c  # ~0.000255 s
    max_delay_s = max_itd * itd_scale
    max_d_samps = int(np.ceil(max_delay_s * fs)) + 4
    if max_d_samps < 8: max_d_samps = 8

    bufL = np.zeros(max_d_samps, np.float32)
    bufR = np.zeros(max_d_samps, np.float32)
    wL = 0; wR = 0

    # Smoothing
    a = _one_pole_coef_ms(dz_ms, fs)
    # Simple one-pole high-shelf approximated by mixing flat and HF emphasis
    # Here we implement ILD as a *gain only* for simplicity (since <1kHz).
    gL = 1.0; gR = 1.0

    # Precompute per-sample target ITDs (seconds)
    # Convention: positive theta -> sound on left; left ear gets earlier (negative delay)
    for n in range(N):
        # write inputs into buffers
        bufL[wL] = L_in[n]
        bufR[wR] = R_in[n]

        s = np.sin(theta_rad[n])
        tau = (head_radius_m / c) * s * itd_scale  # seconds
        # Left should be advanced when sound on left (negative delay),
        # We implement via delaying the *other* ear instead (causal).
        dL_t = max(0.0, +tau) * fs   # samples (delay left if tau positive = sound to left -> right arrives later)
        dR_t = max(0.0, -tau) * fs   # delay right if tau negative (sound to right)

        # smooth delays (samples)
        if n == 0:
            dL = dL_t; dR = dR_t
        else:
            dL = _smooth_step(dL, dL_t, a)
            dR = _smooth_step(dR, dR_t, a)

        # read with fractional delay
        # LEFT out: delayed version of Left buffer by dL
        riL = (wL - dL) % max_d_samps
        r0L = int(np.floor(riL)); r1L = (r0L - 1) % max_d_samps
        fracL = riL - r0L
        Ld = (1.0 - fracL) * bufL[r0L] + fracL * bufL[r1L]

        # RIGHT out: delayed version of Right buffer by dR
        riR = (wR - dR) % max_d_samps
        r0R = int(np.floor(riR)); r1R = (r0R - 1) % max_d_samps
        fracR = riR - r0R
        Rd = (1.0 - fracR) * bufR[r0R] + fracR * bufR[r1R]

        # ILD gain toward near ear (very mild for <1 kHz)
        # Gains vary with sin(theta): near ear gets +gain, far ear gets -gain
        g_near = ild_max_db * 0.5 * s  # +/- dB
        # Convert to linear and smooth
        gL_t = 10.0 ** (+g_near / 20.0)
        gR_t = 10.0 ** (-g_near / 20.0)
        gL = _smooth_step(gL, gL_t, a)
        gR = _smooth_step(gR, gR_t, a)

        L[n] = Ld * gL
        R[n] = Rd * gR

        # advance pointers
        wL = (wL + 1) % max_d_samps
        wR = (wR + 1) % max_d_samps

    return L, R

# -------------------------
# Main: mono -> binaural spatialize
# -------------------------
@numba.njit(cache=True, fastmath=True)
def spatialize_mono_ambi2d(
    x: np.ndarray, fs: float,
    theta_deg: np.ndarray,      # per-sample azimuth
    distance_m: np.ndarray,     # per-sample distance (>= 0.1)
    # decode & ear model
    use_itd_ild: int = 1,
    ear_angle_deg: float = 30.0,
    head_radius_m: float = 0.0875,
    itd_scale: float = 1.0,
    ild_max_db: float = 3.0,
    ild_xover_hz: float = 700.0,
    # distance model
    ref_distance_m: float = 1.0,
    rolloff: float = 1.0,       # amplitude ~ (ref / max(ref, d))^rolloff
    hf_roll_db_per_m: float = 0.0,
    # smoothing (anti-zipper)
    dz_theta_ms: float = 20.0,
    dz_dist_ms: float = 50.0,
    decoder: int = DECODER_ITD_HEAD
):
    """
    Encode mono source at azimuth theta, distance d; decode to stereo.
    decoder: 0 = ITD-only head, 1 = FOA cardioid.
    """
    N = x.shape[0]
    # Smooth theta, distance
    a_th = _one_pole_coef_ms(dz_theta_ms, fs)
    a_di = _one_pole_coef_ms(dz_dist_ms, fs)

    theta = np.empty(N, np.float64)
    dist  = np.empty(N, np.float64)

    th = theta_deg[0] * np.pi/180.0
    di = max(0.1, distance_m[0])
    for n in range(N):
        t_t = theta_deg[n] * np.pi/180.0
        d_t = max(0.1, distance_m[n])
        th = _smooth_angle(th, t_t, a_th)
        di = _smooth_step(di, d_t, a_di)
        theta[n] = th
        dist[n]  = di

    # Simple distance gain (constant-power-ish)
    g = np.empty(N, np.float32)
    for n in range(N):
        d = dist[n]
        # avoid aggressive near-field; clamp min distance
        if d < 0.1: d = 0.1
        g[n] = (ref_distance_m / max(ref_distance_m, d)) ** rolloff

    # Apply decode according to chosen model
    if decoder == DECODER_FOA_CARDIOID:
        W, X, Y = foa_encode_2d(x * g, theta)
        Lc, Rc = foa_decode_stereo_cardioid(W, X, Y, ear_angle_deg)
        if use_itd_ild != 0:
            L, R = _apply_time_varying_itd_ild(
                Lc, Rc, fs, theta,
                head_radius_m=head_radius_m,
                itd_scale=itd_scale, ild_max_db=ild_max_db,
                ild_xover_hz=ild_xover_hz, dz_ms=dz_theta_ms
            )
        else:
            L, R = Lc, Rc
    else:
        mono = x * g
        if use_itd_ild != 0:
            L, R = _apply_time_varying_itd_ild(
                mono, mono, fs, theta,
                head_radius_m=head_radius_m,
                itd_scale=itd_scale, ild_max_db=ild_max_db,
                ild_xover_hz=ild_xover_hz, dz_ms=dz_theta_ms
            )
        else:
            L = mono.copy()
            R = mono.copy()

    # Optional HF rolloff with distance (very small for <1 kHz; we approximate via gain only)
    if hf_roll_db_per_m != 0.0:
        # linearize to a simple per-sample scalar
        for n in range(N):
            att_db = hf_roll_db_per_m * (dist[n] - ref_distance_m)
            att_lin = 10.0 ** (-max(0.0, att_db)/20.0)
            L[n] *= att_lin
            R[n] *= att_lin

    out = np.empty((N,2), np.float32)
    out[:,0] = L
    out[:,1] = R
    return out

# -------------------------
# Binaural-safe spatialize for existing stereo BB
# -------------------------
@numba.njit(cache=True, fastmath=True)
def spatialize_binaural_mid_only(
    x_stereo: np.ndarray, fs: float,
    theta_deg: np.ndarray,
    distance_m: np.ndarray,
    use_itd_ild: int = 1,
    ear_angle_deg: float = 30.0,
    head_radius_m: float = 0.0875,
    itd_scale: float = 1.0,
    ild_max_db: float = 3.0,
    ild_xover_hz: float = 700.0,
    ref_distance_m: float = 1.0,
    rolloff: float = 1.0,
    hf_roll_db_per_m: float = 0.0,
    dz_theta_ms: float = 20.0,
    dz_dist_ms: float = 50.0,
    decoder: int = DECODER_ITD_HEAD
):
    """
    Preserve Δf by spatializing Mid only.
    """
    N = x_stereo.shape[0]
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    M = (x_stereo[:,0] + x_stereo[:,1]) * inv_sqrt2
    S = (x_stereo[:,0] - x_stereo[:,1]) * inv_sqrt2

    Ms = spatialize_mono_ambi2d(
        M.astype(np.float32), fs,
        theta_deg, distance_m,
        use_itd_ild, ear_angle_deg, head_radius_m, itd_scale, ild_max_db, ild_xover_hz,
        ref_distance_m, rolloff, hf_roll_db_per_m, dz_theta_ms, dz_dist_ms, decoder
    )

    out = np.empty_like(x_stereo)
    out[:,0] = (Ms[:,0] + S) * inv_sqrt2
    out[:,1] = (Ms[:,1] - S) * inv_sqrt2
    return out

# -------------------------
# Trajectory generator (arcs)
# -------------------------
def generate_azimuth_trajectory(
    duration: float, fs: float,
    segments: list
):
    """
    segments: list of dicts, each with:
      - mode: "rotate" | "oscillate"
      - start_deg: starting azimuth (deg)
      - extent_deg: for oscillate: peak ± from center; for rotate: total sweep (ignored if using speed)
      - center_deg: for oscillate only (center of arc)
      - speed_deg_per_s: rotate speed (deg/s), sign sets direction
      - period_s: for oscillate (alt to speed)
      - easing: "linear" | "sine" (optional)
      - distance_m: constant or (start,end)
      - seconds: duration of this segment
    Returns theta_deg[N], distance_m[N].
    """
    N_total = int(round(duration*fs))
    theta = np.zeros(N_total, np.float64)
    dist  = np.ones(N_total, np.float64)

    idx = 0
    for seg in segments:
        sec = float(seg.get("seconds", 0.0))
        N = int(round(sec*fs))
        if N <= 0: continue

        mode = seg.get("mode", "rotate")
        start = float(seg.get("start_deg", 0.0))
        center = float(seg.get("center_deg", start))
        extent = float(seg.get("extent_deg", 180.0))
        speed = seg.get("speed_deg_per_s", None)
        period = seg.get("period_s", None)
        easing = seg.get("easing", "linear")

        dval = seg.get("distance_m", 1.0)
        d0 = d1 = 1.0
        if isinstance(dval, (list, tuple)) and len(dval) == 2:
            d0, d1 = float(dval[0]), float(dval[1])
        else:
            d0 = d1 = float(dval)

        t = np.linspace(0.0, sec, N, endpoint=False)
        if mode == "rotate":
            if speed is None:
                # derive average speed from extent/seconds
                sp = np.sign(extent) * (abs(extent)/max(1e-9, sec))
            else:
                sp = float(speed)
            th = start + sp * t
        else:  # oscillate
            if period is None or period <= 0:
                period = sec
            omega = 2*np.pi/period
            # sine around center: center + extent*sin(omega t)
            th = center + extent * np.sin(omega*t)

        # easing: optional simple sine-in-out on angle path
        if easing == "sine":
            u = np.sin(np.pi * (t/sec - 0.5)) * 0.5 + 0.5  # 0..1 eased
            th = th[0] + (th - th[0]) * u

        # distance linear interp
        d = d0 + (d1 - d0) * (t/sec if sec > 0 else 0.0)

        end = idx+N
        if end > N_total:
            N = N_total - idx
            th = th[:N]; d = d[:N]; end = idx+N
        theta[idx:end] = th
        dist[idx:end] = d
        idx = end
        if idx >= N_total: break

    # pad remaining with last values
    if idx < N_total:
        theta[idx:] = theta[idx-1] if idx>0 else 0.0
        dist[idx:]  = dist[idx-1]  if idx>0 else 1.0

    return theta.astype(np.float64), dist.astype(np.float64)
