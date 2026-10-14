#!/usr/bin/env python3
"""
DeepPicar v3 - drive the physics with YOUR data
================================================
Single controller = the CNN. This version feeds the simulator with real logs:

  --dataset  PATH   recorded driving log  (DeepPicar / Udacity sim / Udacity challenge-2 style CSV)
  --terrain  PATH   route profile         (CSV with distance+elevation|slope[+surface], or a GPX track)
  --surfaces-json   your measured {"surface": [mu, C_rr], ...}

Utilities
  --list-datasets             show where to get suitable data
  --download URL DEST         fetch a direct-link file (Google-Drive/large files: download by hand)
  --make-sample PATH          write a SYNTHETIC test file in DeepPicar log format (pipeline test only)

Examples
  python3 deeppicar_sim_v3.py --list-datasets
  python3 deeppicar_sim_v3.py --make-sample sample_log.csv
  python3 deeppicar_sim_v3.py --dataset out-key-btn.csv
  python3 deeppicar_sim_v3.py --dataset driving_log.csv --format udacity_sim --fps 10 --use-log-throttle
  python3 deeppicar_sim_v3.py --dataset out-key-btn.csv --terrain hill.gpx --terrain-scale 0.01
  python3 deeppicar_sim_v3.py            # no dataset -> closed-loop surrogate CNN (as v2)

With a dataset the sim REPLAYS the recorded steering at the recorded frame times (real jitter)
and, if present, the recorded per-frame latency. Open-loop, so the output is tyre demand, slip,
speed and energy per terrain - not tracking error. Without a dataset it runs the closed-loop
surrogate and reports tracking error.
"""
import argparse, csv, json, math, os, sys, urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================== vehicle (report Table I)
m, L, r_w, g = 0.28, 0.155, 0.015, 9.81
I_w = 9.0e-7
DELTA_MAX = math.radians(30)
F_NOM, T_LAT_NOM = 43.7, 0.050
TAU_SERVO = 0.05                                 # ASSUMED
RHO, CD, A_FRONT = 1.225, 1.0, 0.0015            # ASSUMED
F_STALL, V_NOLOAD, THROTTLE = 0.60, 1.70, 0.65   # ASSUMED open-loop motor model

# ============================================================== surfaces (mu, C_rr)
# Sourced values are for FULL-SIZE car tyres (see --list-datasets for links); small foam/rubber
# RC tyres differ, so overwrite with your own measurements via --surfaces-json.
SURFACES = {
    "asphalt_dry":  (0.85, 0.010),   # mu 0.8-0.9 (hpwizard); C_rr 0.01-0.02 (engineersedge)
    "asphalt_wet":  (0.60, 0.015),   # mu 0.5-0.7 (hpwizard); C_rr 0.015 (engineersedge)
    "gravel_rolled":(0.60, 0.020),   # mu 0.60 (hpwizard); C_rr 0.02 (engineeringtoolbox)
    "gravel_loose": (0.55, 0.040),   # C_rr 0.04 (engineersedge); mu ~ gravel 0.55-0.60
    "soil":         (0.50, 0.080),   # C_rr 0.08 medium-hard soil; mu ASSUMED
    "sand":         (0.40, 0.200),   # C_rr 0.1-0.3; mu ASSUMED
    "tile":         (0.80, 0.015),   # indoor floor - ASSUMED, measure yours
    "carpet":       (1.00, 0.040),   # indoor floor - ASSUMED, measure yours
}
COURSE = [(3.0, "tile", 0.0), (6.0, "carpet", 4.0), (9.0, "gravel_loose", 0.0),
          (12.0, "asphalt_wet", -5.0), (15.0, "soil", 0.0)]      # demo mixed course
uniform = lambda s, slope=0.0: [(15.0, s, slope)]

def terrain_at(course, s):
    for end, surf, sl in course:
        if s < end: return surf, math.radians(sl)
    return course[-1][1], math.radians(course[-1][2])

# ============================================================== dataset registry
REGISTRY = [
    ("DeepPicar dataset (the car in your report)", "https://drive.google.com/open?id=1LjIcOVH7xmbxV58lx3BClRcZ2DACfSwh",
     "Training data from the DeepPicar paper authors (Google Drive - download by hand)."),
    ("DeepPicar-v2 repo (log format + code)", "https://github.com/mbechtel2/DeepPicar-v2",
     "Logs: out-key.csv = ts_ms,frame_id,angle(rad); out-key-btn.csv adds btn + per-frame latency (ms)."),
    ("DeepPicar-v3 repo (newer)", "https://github.com/CSL-KU/DeepPicar-v3", "Newer version from the same group."),
    ("Udacity Self-Driving Car Simulator", "https://github.com/udacity/self-driving-car-sim",
     "Record your own: driving_log.csv = center,left,right,steering,throttle,brake,speed."),
    ("Udacity challenge-2 (real car) tooling", "https://github.com/kyle-dorman/steering-angle-predictor",
     "Describes converting the ROSBAGs to interpolated.csv: angle, torque, speed, lat, long, alt."),
    ("DeepPiCar (dctian) - behaviour cloning", "https://github.com/dctian/DeepPiCar",
     "Another small-car pipeline: record your own video + steering."),
    ("Friction / rolling-resistance tables", "https://hpwizard.com/tire-friction-coefficient.html",
     "mu by surface. Also: engineeringtoolbox.com/rolling-friction-resistance-d_1303.html, engineersedge.com rolling-resistance page."),
    ("Terrain / slopes", "GPX track (e.g. from a GPS app) or SRTM-derived elevation CSV",
     "Use --terrain; --terrain-scale compresses a real route to RC-car scale while keeping grades."),
]
def list_datasets():
    for i, (n, u, d) in enumerate(REGISTRY, 1): print(f"{i}. {n}\n   {u}\n   {d}\n")

def download(url, dest):
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest); print("done,", os.path.getsize(dest), "bytes")

# ============================================================== dataset loading
@dataclass
class DrivingData:
    name: str
    fmt: str
    t: np.ndarray                      # s, starts at 0
    steer: np.ndarray                  # rad, road-wheel
    throttle: Optional[np.ndarray] = None
    speed: Optional[np.ndarray] = None # m/s
    latency: Optional[np.ndarray] = None  # s
    alt: Optional[np.ndarray] = None

def _isnum(x):
    try: float(x); return True
    except ValueError: return False

def _pick(names, options):
    for o in options:
        if o in names: return names.index(o)
    return None

def load_driving_log(path, fmt="auto", unit=None, fps=10.0, scale=1.0, norm_deg=25.0,
                     speed_scale=1.0, quantise="auto"):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    first = [c.strip().lower() for c in rows[0]]
    has_header = bool({"center", "steering", "angle", "timestamp", "ts", "wheel", "speed"} & set(first))
    th = lat = alt = spd = None
    if has_header:
        body = rows[1:]; col = lambda opts: _pick(first, opts)
        i_s = col(["steering", "angle", "wheel", "steer"]); i_t = col(["timestamp", "ts", "time", "t"])
        i_v, i_th = col(["speed"]), col(["throttle"]); i_l = col(["latency", "latency_ms"]); i_a = col(["alt", "altitude"])
        if i_s is None: sys.exit(f"No steering/angle column in header: {rows[0]}")
        get = lambda i: None if i is None else np.array([float(r[i]) for r in body])
        steer, tt, spd, th, lat, alt = get(i_s), get(i_t), get(i_v), get(i_th), get(i_l), get(i_a)
        fmt = fmt if fmt != "auto" else ("udacity_sim" if i_t is None else "header_csv")
        unit = unit or ("norm" if fmt == "udacity_sim" else "rad")
        if "udacity_sim" == fmt and "center" in first: pass
    elif _isnum(first[0]):                                   # DeepPicar: ts_ms,frame_id,angle[,btn,latency_ms]
        fmt = "deeppicar"; unit = unit or "rad"
        a = np.array([[float(c) for c in r[:5]] + [np.nan] * (5 - len(r[:5])) for r in rows])
        tt, steer = a[:, 0], a[:, 2]
        lat = a[:, 4] if not np.isnan(a[:, 4]).all() else None
    else:                                                    # Udacity sim, headerless
        fmt = "udacity_sim"; unit = unit or "norm"
        steer = np.array([float(r[3]) for r in rows]); th = np.array([float(r[4]) for r in rows])
        spd = np.array([float(r[6]) for r in rows]) if len(rows[0]) > 6 else None
        tt = None
    # time base -> seconds from 0
    if tt is None:
        t = np.arange(len(steer)) / fps
    else:
        d = float(np.median(np.diff(tt))) if len(tt) > 1 else 0
        t = (tt - tt[0]) / (1e9 if d > 1e6 else 1e3 if d >= 1 else 1.0)
    # steering -> road-wheel rad
    steer = {"rad": steer, "deg": np.radians(steer), "norm": np.radians(steer * norm_deg)}[unit] * scale
    steer = np.clip(steer, -DELTA_MAX, DELTA_MAX)
    q = (3 if fmt == "deeppicar" else 0) if quantise == "auto" else int(quantise)
    if q == 3: steer = np.where(np.abs(steer) > math.radians(15), np.sign(steer) * DELTA_MAX, 0.0)
    if lat is not None: lat = lat / 1e3 if fmt == "deeppicar" else lat   # ms -> s
    return DrivingData(os.path.basename(path), fmt, t, steer, th, None if spd is None else spd * speed_scale,
                       lat, alt)

def make_sample(path, dur=25.0, seed=0):
    """SYNTHETIC DeepPicar-format file (ts_ms,frame_id,angle_rad,btn,latency_ms). Pipeline test only."""
    rng = np.random.default_rng(seed); t = 0.0; rows = []; i = 0
    while t < dur:
        ph = t % 1.15
        a = 0.0 if ph < 0.10 else 0.52 if ph < 0.34 else 0.0 if ph < 0.44 else -0.52 if ph < 0.68 else 0.0
        lat = 50 + rng.normal(0, 6) + (120 if rng.random() < 0.02 else 0)
        rows.append(f"{int(t*1000)},{i},{a},0,{int(max(lat, 20))}")
        t += rng.normal(1 / 43.7, 0.002); i += 1
    open(path, "w").write("\n".join(rows) + "\n"); print("wrote synthetic sample:", path)

# ============================================================== terrain loading
def _haversine(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    return np.concatenate([[0], np.cumsum(2 * 6371000 * np.arcsin(np.sqrt(a)))])

def load_terrain(path, scale=1.0, surface="asphalt_dry", bin_m=0.5, smooth=5):
    surf_pts = None
    if path.lower().endswith(".gpx"):
        pts = [(float(e.get("lat")), float(e.get("lon")),
                float(next((c.text for c in e if c.tag.endswith("ele")), 0))) for e in ET.parse(path).getroot().iter()
               if e.tag.endswith("trkpt")]
        lat, lon, ele = map(np.array, zip(*pts)); d = _haversine(lat, lon)
    else:
        with open(path, newline="") as f: rd = list(csv.DictReader(f))
        d = np.array([float(r["distance_m"]) for r in rd])
        if "slope_deg" in rd[0]: sl_pts = np.array([float(r["slope_deg"]) for r in rd]); ele = None
        else: ele = np.array([float(r["elevation_m"]) for r in rd])
        if "surface" in rd[0]: surf_pts = [r["surface"].strip() for r in rd]
    if ele is not None:
        if smooth > 1:
            pad = np.pad(ele, smooth // 2, mode="edge"); ele = np.convolve(pad, np.ones(smooth) / smooth, "valid")
        sl = np.degrees(np.arctan2(np.diff(ele), np.maximum(np.diff(d), 1e-6)))       # grade is scale-invariant
        mid = (d[1:] + d[:-1]) / 2
    else:
        sl, mid = sl_pts, d
    sl = np.clip(sl, -20, 20); ds = d * scale; mid = mid * scale
    edges = np.arange(bin_m, ds[-1] + bin_m, bin_m)
    course = []
    for e in edges:
        c = e - bin_m / 2
        sf = surface if surf_pts is None else surf_pts[int(np.argmin(np.abs(ds - c)))]
        if sf not in SURFACES: sys.exit(f"Unknown surface '{sf}'. Known: {list(SURFACES)}")
        course.append((float(e), sf, float(np.interp(c, mid, sl))))
    print(f"terrain: {len(course)} bins, {edges[-1]:.1f} m, slope {min(c[2] for c in course):.1f}..{max(c[2] for c in course):.1f} deg")
    return course

# ============================================================== the single controller (surrogate)
LEVELS = np.radians([-30, 0, 30])
y_ref = lambda x: 0.40 * np.sin(2 * np.pi * x / 3.0)
psi_ref = lambda x: np.arctan(0.40 * 2 * np.pi / 3.0 * np.cos(2 * np.pi * x / 3.0))
def cnn_surrogate(x, y, th, v):
    cte = (y - y_ref(x)) * np.cos(psi_ref(x))
    d = -(th - psi_ref(x)) - np.arctan(2.5 * cte / (v + 0.3))
    return LEVELS[np.argmin(np.abs(LEVELS - d))]

# ============================================================== simulation
def simulate(course, k=1.0, data=None, use_log_thr=False, dt=1e-3, T_max=60.0, s_goal=14.8):
    replay = data is not None
    f_ctrl, t_lat = F_NOM / k, T_LAT_NOM * k
    if replay:
        T_max = data.t[-1] + 0.5; s_goal = 1e18
        Tmed = float(np.median(np.diff(data.t))); jn = 0; jlast = -1; N = len(data.t)
    pend = deque(); cmd_d = 0.0; held = 0.0; thr = THROTTLE; last = -1e9
    x = y = th = v = delta = s = z = 0.0; W = Ed = 0.0
    K = ["t", "x", "y", "s", "v", "Fd", "Flat", "util", "slip", "cte", "surf", "z", "W", "Ed"]
    lg = {kk: [] for kk in K}
    for i in range(int(T_max / dt)):
        t = i * dt
        surf, a = terrain_at(course, s); mu, Crr = SURFACES[surf]; N_ = m * g * math.cos(a)
        # ---- controller: sample -> (quantised by loader) -> latency ----
        if replay:
            while jn + 1 < N and data.t[jn + 1] <= t: jn += 1
            if ((k == 1 and jn != jlast) or (k > 1 and t - last >= k * Tmed)) and t >= data.t[0]:
                jlast = jn; held = data.steer[jn]; last = t
                if use_log_thr and data.throttle is not None: thr = float(np.clip(data.throttle[jn], 0, 1))
                lat = (data.latency[jn] if data.latency is not None and not np.isnan(data.latency[jn]) else T_LAT_NOM) * k
                pend.append((t + lat, held))
        elif t - last >= 1 / f_ctrl:
            held = cnn_surrogate(x, y, th, v); last = t; pend.append((t + t_lat, held))
        while pend and pend[0][0] <= t: cmd_d = pend.popleft()[1]
        # ---- plant ----
        delta = float(np.clip(delta + (cmd_d - delta) / TAU_SERVO * dt, -DELTA_MAX, DELTA_MAX))
        Fd = min(thr * F_STALL * (1 - v / V_NOLOAD), mu * N_)
        Frr, Fdr, Fg = Crr * N_, 0.5 * RHO * CD * A_FRONT * v * v, m * g * math.sin(a)
        Fn = Fd - Frr * (v > 1e-6) - Fdr - Fg
        if v <= 1e-6 and Fn < 0: Fn = 0.0
        v = max(v + Fn / m * dt, 0.0)
        om_g = v * math.tan(delta) / L; Freq = m * v * om_g
        Fav = math.sqrt(max((mu * N_) ** 2 - Fd ** 2, 0.0)); slip = abs(Freq) > Fav and v > 1e-3
        om = math.copysign(Fav / (m * v), om_g) if slip else om_g
        x += v * math.cos(th) * dt; y += v * math.sin(th) * dt; th += om * dt
        s += v * dt; z += v * math.sin(a) * dt; W += Fd * v * dt; Ed += (Frr + Fdr) * v * dt
        cte = float("nan") if replay else (y - y_ref(x)) * math.cos(psi_ref(x))
        for kk, val in zip(K, [t, x, y, s, v, Fd, m * v * om, math.hypot(Fd, Freq) / (mu * N_), slip, cte, surf, z, W, Ed]):
            lg[kk].append(val)
        if s >= s_goal or (not replay and abs(cte) > 1.0): break
    o = {kk: np.array(vv) for kk, vv in lg.items()}
    o["off_track"] = (not replay) and abs(o["cte"][-1]) > 1.0
    o["finished"] = bool(s >= s_goal) if not replay else True
    o["rms_cte"] = float(np.sqrt(np.nanmean(o["cte"] ** 2))) if not replay else float("nan")
    o["slip_pct"] = 100 * float(np.mean(o["slip"])); o["max_util"] = float(o["util"].max())
    return o

# ============================================================== reporting
def dataset_summary(d):
    per = np.diff(d.t)
    print("=" * 74); print(f"DATASET {d.name}  format={d.fmt}  frames={len(d.t)}  duration={d.t[-1]:.1f} s")
    print(f"  measured control rate : {1/per.mean():.1f} Hz mean, period p99 = {np.percentile(per,99)*1e3:.1f} ms (nominal {1e3/F_NOM:.1f} ms)")
    if d.latency is not None: print(f"  per-frame latency     : mean {np.nanmean(d.latency)*1e3:.0f} ms, p99 {np.nanpercentile(d.latency,99)*1e3:.0f} ms (nominal 50 ms)")
    print(f"  steering              : mean |delta| {np.degrees(np.abs(d.steer).mean()):.1f} deg, at +/-30 deg {100*np.mean(np.abs(d.steer)>=DELTA_MAX-1e-9):.0f}% of frames")
    if d.speed is not None: print(f"  logged speed          : mean {np.nanmean(d.speed):.2f}, max {np.nanmax(d.speed):.2f} (check units, see --speed-scale)")

def fig_dataset(d, out):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    ax[0, 0].plot(d.t, np.degrees(d.steer), lw=0.7); ax[0, 0].set_title("Steering command (road wheel)"); ax[0, 0].set_xlabel("t (s)"); ax[0, 0].set_ylabel("deg")
    ax[0, 1].hist(np.degrees(d.steer), bins=40); ax[0, 1].set_title("Steering distribution"); ax[0, 1].set_xlabel("deg")
    ax[1, 0].hist(np.diff(d.t) * 1e3, bins=50); ax[1, 0].axvline(1e3 / F_NOM, color="k", ls="--"); ax[1, 0].set_title("Frame period (dashed = 22.9 ms nominal)"); ax[1, 0].set_xlabel("ms")
    if d.latency is not None: ax[1, 1].hist(d.latency * 1e3, bins=50); ax[1, 1].set_title("Per-frame latency"); ax[1, 1].set_xlabel("ms")
    elif d.speed is not None: ax[1, 1].plot(d.t, d.speed); ax[1, 1].set_title("Logged speed"); ax[1, 1].set_xlabel("t (s)")
    else: ax[1, 1].axis("off")
    fig.suptitle(f"Dataset overview: {d.name}"); fig.tight_layout(); fig.savefig(os.path.join(out, "fig_dataset_overview.png"), dpi=130)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset"); ap.add_argument("--format", default="auto", choices=["auto", "deeppicar", "udacity_sim", "header_csv"])
    ap.add_argument("--unit", choices=["rad", "deg", "norm"]); ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply steering (e.g. 1/steering_ratio for steering-wheel angles)")
    ap.add_argument("--norm-deg", type=float, default=25.0, help="deg at steering=1.0 for normalised logs (ASSUMED 25)")
    ap.add_argument("--speed-scale", type=float, default=1.0); ap.add_argument("--quantise", default="auto", choices=["auto", "0", "3"])
    ap.add_argument("--use-log-throttle", action="store_true")
    ap.add_argument("--terrain"); ap.add_argument("--terrain-scale", type=float, default=1.0)
    ap.add_argument("--surface", default="asphalt_dry", help="surface for terrain files without a surface column")
    ap.add_argument("--surfaces-json"); ap.add_argument("--outdir", default=".")
    ap.add_argument("--list-datasets", action="store_true"); ap.add_argument("--download", nargs=2, metavar=("URL", "DEST"))
    ap.add_argument("--make-sample", metavar="PATH")
    a = ap.parse_args()
    if a.list_datasets: return list_datasets()
    if a.download: return download(*a.download)
    if a.make_sample: return make_sample(a.make_sample)
    if a.surfaces_json: SURFACES.update({k: tuple(v) for k, v in json.load(open(a.surfaces_json)).items()})
    os.makedirs(a.outdir, exist_ok=True)

    data = None
    if a.dataset:
        data = load_driving_log(a.dataset, a.format, a.unit, a.fps, a.scale, a.norm_deg, a.speed_scale, a.quantise)
        dataset_summary(data); fig_dataset(data, a.outdir)
    courses = {}
    if a.terrain: courses["custom terrain"] = load_terrain(a.terrain, a.terrain_scale, a.surface)
    else: courses["mixed demo course"] = COURSE
    for sname in ["asphalt_dry", "asphalt_wet", "gravel_loose", "soil", "sand"]: courses[sname + " (flat)"] = uniform(sname)
    ks = [1, 4, 11.6]
    res = {(c, kk): simulate(cr, kk, data, a.use_log_throttle) for c, cr in courses.items() for kk in ks}

    print("=" * 74)
    if data is not None:
        print("REPLAY of recorded steering on each terrain  (slip% | peak friction demand | mean speed m/s; S = car stalled)")
        print(f"{'terrain':24s}" + "".join(f"   k={kk:<5}" + " " * 14 for kk in ks))
        for c in courses:
            print(f"{c:24s}" + "".join(f"  {res[c,kk]['slip_pct']:5.1f}% | {res[c,kk]['max_util']:4.2f} | {res[c,kk]['v'].mean():4.2f}{'S' if res[c,kk]['v'].max() < 0.05 else ' '}  " for kk in ks))
        if data.speed is not None and a.use_log_throttle:
            r = res[(list(courses)[0], 1)]; err = np.nanmean(np.abs(np.interp(data.t, r["t"], r["v"]) - data.speed))
            print(f"\nSpeed validation vs logged speed (first terrain): mean |error| = {err:.2f} m/s")
    else:
        print("CLOSED-LOOP surrogate CNN  (RMS cross-track m; * left track, S = car stalled)")
        print(f"{'terrain':24s}" + "".join(f"{('k=%g' % kk):>10s}" for kk in ks))
        for c in courses:
            print(f"{c:24s}" + "".join(f"{res[c,kk]['rms_cte']:>9.3f}{'*' if res[c,kk]['off_track'] else ('S' if res[c,kk]['v'].max() < 0.05 else ' ')}" for kk in ks))
    print("(k scales control period up and latency up together; 11.6 = report's worst contention case)")

    # figures
    fig, ax = plt.subplots(figsize=(9, 5))
    if data is None:
        xs = np.linspace(0, 12, 400); ax.plot(xs, y_ref(xs), "k--", lw=1, label="reference")
    for c in list(courses)[:3]:
        r = res[(c, 1)]; ax.plot(r["x"], r["y"], lw=1.2, label=f"{c}, k=1")
    r = res[(list(courses)[0], 11.6)]; ax.plot(r["x"], r["y"], "r:", lw=1.2, label=f"{list(courses)[0]}, k=11.6")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(fontsize=7)
    ax.set_title("Trajectories (replayed log)" if data else "Trajectories (surrogate CNN)")
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "fig_paths.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(8, 4)); w = 0.25; names = list(courses)
    key, lab = ("slip_pct", "Time in slip (%)") if data is not None else ("rms_cte", "RMS cross-track error (m)")
    for j, kk in enumerate(ks):
        ax.bar(np.arange(len(names)) + j * w, [res[(c, kk)][key] for c in names], w, label=f"k={kk}")
    ax.set_xticks(np.arange(len(names)) + w); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(lab); ax.legend(); ax.set_title("Effect of terrain and control-loop degradation")
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "fig_terrain_comparison.png"), dpi=130)

    r = res[(list(courses)[0], 1)]
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axs[0].plot(r["s"], r["v"]); axs[0].set_ylabel("speed (m/s)")
    axs[1].plot(r["s"], r["util"]); axs[1].axhline(1, color="k", ls="--"); axs[1].set_ylabel("friction demand (>1 = slip)"); axs[1].set_xlabel("distance (m)")
    axs[0].set_title(f"Speed and tyre demand on '{list(courses)[0]}'"); fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "fig_speed_demand.png"), dpi=130)
    print("figures written to", os.path.abspath(a.outdir))

if __name__ == "__main__":
    main()
