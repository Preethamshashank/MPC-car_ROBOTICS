#!/usr/bin/env python3
"""
DeepPicar v4 - ONE controller: Model Predictive Control (steering + throttle together)
======================================================================================
Replaces the CNN stand-in, the PD servo logic and the PI speed loop. Each control tick
(f_ctrl = 43.7/k Hz) the MPC:
  1. predicts the car forward THROUGH the commands still in flight (latency compensation),
  2. simulates 500 candidate plans (4 throttle/brake levels x 5^3 steering profiles) ~1 s ahead with
     the same bicycle + friction-circle + slope + surface model,
  3. scores each: path error, heading error, speed error, friction-limit use, lane edge, cones,
  4. sends the best plan's FIRST (steering, throttle) - then repeats (receding horizon).
It is sampling-based MPC (preset plans, vectorised), not a QP solver.

Reuses the physics, surfaces, terrain loader and course from deeppicar_sim_v3.py.

Run:  python3 deeppicar_sim_v4_mpc.py
      python3 deeppicar_sim_v4_mpc.py --terrain data/terrain_SYNTHETIC.csv --surfaces-json data/surfaces.json
      python3 deeppicar_sim_v4_mpc.py --quick        # skip the sweeps
IDEALISATION: the MPC knows the exact model, terrain and cone positions (no camera / noise).
"""
import argparse, itertools, json, os
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import deeppicar_sim_v3 as V3
from deeppicar_sim_v3 import (m, L, g, DELTA_MAX, TAU_SERVO, SURFACES, COURSE, uniform, terrain_at,
                              load_terrain, y_ref, psi_ref, F_STALL, V_NOLOAD, RHO, CD, A_FRONT, F_NOM, T_LAT_NOM)

OBST = [(2.2, 0.0), (5.0, -0.08), (13.0, -0.06)]      # static cones: x, offset from path
# moving obstacles: two pedestrians crossing the track, one slow vehicle on the path ahead
DYN = [dict(kind="ped", x=3.6, amp=0.5, per=3.4, ph=0.0, r=0.02), dict(kind="ped", x=8.2, amp=0.5, per=2.8, ph=1.5, r=0.02),
       dict(kind="car", x0=5.5, v=0.4, r=0.05)]
def dpos(o, t):
    """(x, y, vx, vy) of a moving obstacle at time t."""
    if o["kind"] == "ped":
        w = 2 * np.pi / o["per"]
        return (o["x"], y_ref(o["x"]) + o["amp"] * np.sin(w * t + o["ph"]), 0.0, o["amp"] * w * np.cos(w * t + o["ph"]))
    x = o["x0"] + o["v"] * t
    return (x, y_ref(x), o["v"], o["v"] * 0.4 * 2 * np.pi / 3 * np.cos(2 * np.pi * x / 3))
LV = np.radians([-30, -12, 0, 12, 30]); TH = np.array([-1.0, 0.15, 0.55, 1.0])
BRK, CURB = 1.0, 0.52     # brake force 1.0 N (ASSUMED); road half-width enforced by curbs
CAND = np.array([(t, a, b, c) for t in TH for a in LV for b in LV for c in LV])
HN, HDT = 20, 0.05
EH, EP = 1 - np.exp(-HDT / TAU_SERVO), 1 - np.exp(-0.01 / TAU_SERVO)

def terrain_lookup(course):
    edges = np.array([c[0] for c in course]); mu = np.array([SURFACES[c[1]][0] for c in course])
    crr = np.array([SURFACES[c[1]][1] for c in course]); sl = np.radians([c[2] for c in course])
    def look(x):
        i = np.minimum(np.searchsorted(edges, x, side="right"), len(edges) - 1)
        return mu[i], crr[i], sl[i]
    return look

def pstep(P, cm, tc, dt, e, look):
    """Vectorised prediction model (same physics as the plant)."""
    mu, crr, a = look(P["x"]); N = m * g * np.cos(a)
    P["dl"] = P["dl"] + (cm - P["dl"]) * e
    Fd = np.where(tc < 0, np.maximum(tc * BRK, -mu * N), np.minimum(tc * F_STALL * (1 - P["v"] / V_NOLOAD), mu * N))
    ac = (Fd - np.where(P["v"] > 1e-6, crr * N, 0) - 0.5 * RHO * CD * A_FRONT * P["v"] ** 2 - m * g * np.sin(a)) / m
    ac = np.where((P["v"] <= 1e-6) & (ac < 0), 0, ac)
    P["v"] = np.maximum(P["v"] + ac * dt, 0)
    om = P["v"] * np.tan(P["dl"]) / L; Fq = m * P["v"] * om
    Fav = np.sqrt(np.maximum((mu * N) ** 2 - Fd ** 2, 0)); P["u"] = np.hypot(Fd, Fq) / (mu * N)
    o2 = np.where((np.abs(Fq) > Fav) & (P["v"] > 1e-3), np.sign(om) * Fav / np.maximum(m * P["v"], 1e-9), om)
    P["x"] = P["x"] + P["v"] * np.cos(P["th"]) * dt; P["y"] = P["y"] + P["v"] * np.sin(P["th"]) * dt
    P["th"] = P["th"] + o2 * dt

def mpc(st, pend, t, k, cmd, thr, vt, look, obst, dyn=()):
    P = {kk: np.array([st[kk]]) for kk in ("x", "y", "th", "v", "dl")}; P["u"] = np.zeros(1)
    cm, tc, qi, q, tt = cmd, thr, 0, list(pend), 0.0
    while tt < T_LAT_NOM * k - 1e-9:                      # predict through the delay
        while qi < len(q) and q[qi][0] <= t + tt: cm, tc = q[qi][1], q[qi][2]; qi += 1
        pstep(P, cm, tc, 0.01, EP, look); tt += 0.01
    n = len(CAND); Q = {kk: np.full(n, P[kk][0]) for kk in P}
    J = 0.3 * (CAND[:, 1] - cm) ** 2
    MD = np.full(n, 9.0); UM = np.zeros(n); PX, PY = [], []
    now = [dpos(d, t) for d in dyn]                       # measured position + velocity of moving obstacles
    for i in range(HN):
        u = CAND[:, 1] if i < 7 else CAND[:, 2] if i < 14 else CAND[:, 3]
        pstep(Q, u, CAND[:, 0], HDT, EH, look)
        ph = psi_ref(Q["x"]); cte = (Q["y"] - y_ref(Q["x"])) * np.cos(ph)
        J += (4 * cte ** 2 + 0.5 * (Q["th"] - ph) ** 2 + 1.5 * (Q["v"] - vt) ** 2
              + np.where(np.abs(cte) > 0.40, 600 + 8000 * (np.abs(cte) - 0.40) ** 2, 0) + 30 * np.maximum(Q["u"] - 0.9, 0) ** 2
              + 60 * np.maximum(np.abs(Q["th"] - ph) - 0.7, 0) ** 2)
        tau = T_LAT_NOM * k + (i + 1) * HDT; UM = np.maximum(UM, Q["u"]); PX.append(Q["x"].copy()); PY.append(Q["y"].copy())
        for ox, oy in obst:
            d = np.hypot(ox - Q["x"], oy - Q["y"]); MD = np.minimum(MD, d); J += 300 * np.maximum(0.26 - d, 0) ** 2 + 3000 * (d < 0.14)
        for (px, py, vx, vy), dd in zip(now, dyn):          # constant-velocity prediction
            d = np.hypot(px + vx * tau - Q["x"], py + vy * tau - Q["y"]); rr = dd["r"]; MD = np.minimum(MD, d - rr)
            J += 300 * np.maximum(0.26 + rr - d, 0) ** 2 + 3000 * (d < 0.14 + rr)
    j = int(np.argmin(J)); thr_out = -1.0 if MD[j] < 0.12 else CAND[j, 0]   # safety: best plan still hits -> full brake
    return CAND[j, 1], thr_out, float(J[j]), float(MD[j]), float(UM[j]), (np.array([p[j] for p in PX]), np.array([p[j] for p in PY]))

def simulate_mpc(course, k=1.0, obstacles=True, vt=1.2, dt=1e-3, T_max=40.0, s_goal=14.8):
    look = terrain_lookup(course); obst = [(x, y_ref(x) + o) for x, o in OBST] if obstacles else []
    hit = [False] * len(obst); hits = 0; curbs = 0; onc = False; dyn = DYN if obstacles else []; dhit = [False] * len(dyn)
    x = y = th = v = delta = s = z = W = Ed = 0.0; cmd, thr, last = 0.0, 0.5, -1e9; pend = deque(); info = {kk: [] for kk in ('t', 'J', 'md', 'um', 'thr')}; plans = []
    K = ["t", "x", "y", "s", "v", "delta", "cmd", "thr", "Fd", "Flat", "util", "slip", "cte", "W"]
    lg = {kk: [] for kk in K}
    for i in range(int(T_max / dt)):
        t = i * dt; surf, a = terrain_at(course, s); mu, Crr = SURFACES[surf]; N = m * g * np.cos(a)
        if t - last >= k / F_NOM:
            c, tc, Jb, md, um, pth = mpc(dict(x=x, y=y, th=th, v=v, dl=delta), pend, t, k, cmd, thr, vt, look, obst, dyn)
            last = t; pend.append((t + T_LAT_NOM * k, c, tc))
            for kk, val in zip(('t', 'J', 'md', 'um', 'thr'), (t, Jb, md, um, tc)): info[kk].append(val)
            if len(info['t']) % 20 == 1: plans.append(pth)
        while pend and pend[0][0] <= t: _, cmd, thr = pend.popleft()
        delta = float(np.clip(delta + (cmd - delta) / TAU_SERVO * dt, -DELTA_MAX, DELTA_MAX))
        Fd = max(thr * BRK, -mu * N) if thr < 0 else min(thr * F_STALL * (1 - v / V_NOLOAD), mu * N)
        Frr, Fdr, Fg = Crr * N, 0.5 * RHO * CD * A_FRONT * v * v, m * g * np.sin(a)
        Fn = Fd - Frr * (v > 1e-6) - Fdr - Fg
        if v <= 1e-6 and Fn < 0: Fn = 0.0
        v = max(v + Fn / m * dt, 0.0)
        om_g = v * np.tan(delta) / L; Freq = m * v * om_g
        Fav = np.sqrt(max((mu * N) ** 2 - Fd ** 2, 0.0)); slip = abs(Freq) > Fav and v > 1e-3
        om = np.sign(om_g) * Fav / (m * v) if slip else om_g
        x += v * np.cos(th) * dt; y += v * np.sin(th) * dt; th += om * dt; s += v * dt; W += max(Fd, 0.0) * v * dt
        for j, (ox, oy) in enumerate(obst):
            if not hit[j] and np.hypot(ox - x, oy - y) < 0.13: hit[j] = True; hits += 1; v *= 0.2
        ps = psi_ref(x); ct = (y - y_ref(x)) * np.cos(ps)
        if abs(ct) > CURB:                                   # curb guard: keep the car on the road
            y -= np.sign(ct) * (abs(ct) - CURB) / np.cos(ps); th = ps; v *= 0.6
            if not onc: curbs += 1; onc = True
        else: onc = False
        for j, d in enumerate(dyn):
            p = dpos(d, t)
            if not dhit[j] and np.hypot(p[0] - x, p[1] - y) < 0.13 + d["r"]: dhit[j] = True; hits += 1; v *= 0.2
        cte = (y - y_ref(x)) * np.cos(psi_ref(x))
        for kk, val in zip(K, [t, x, y, s, v, delta, cmd, thr, Fd, m * v * om, np.hypot(Fd, Freq) / (mu * N), slip, cte, W]): lg[kk].append(val)
        if s >= s_goal or abs(cte) > 1.0: break
    o = {kk: np.array(vv) for kk, vv in lg.items()}
    o.update(hits=hits, finished=bool(s >= s_goal), off_track=bool(abs(o["cte"][-1]) > 1.0),
             rms_cte=float(np.sqrt(np.mean(o["cte"] ** 2))), max_cte=float(np.abs(o["cte"]).max()),
             slip_pct=100 * float(np.mean(o["slip"])), max_util=float(o["util"].max()), obst=obst, dyn=dyn, curbs=curbs, info={kk: np.array(vv) for kk, vv in info.items()}, plans=plans)
    return o

def flag(r): return "LOST" if r["off_track"] else ("ok" if r["finished"] else "T/O")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain"); ap.add_argument("--terrain-scale", type=float, default=1.0)
    ap.add_argument("--surface", default="asphalt_dry"); ap.add_argument("--surfaces-json")
    ap.add_argument("--target-speed", type=float, default=1.2); ap.add_argument("--outdir", default="runs/v4_mpc")
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    if a.surfaces_json: SURFACES.update({kk: tuple(vv) for kk, vv in json.load(open(a.surfaces_json)).items()})
    os.makedirs(a.outdir, exist_ok=True)
    course = load_terrain(a.terrain, a.terrain_scale, a.surface) if a.terrain else COURSE
    vt = a.target_speed; ks = [1, 4, 8, 11.6] if not a.quick else [1, 11.6]
    print("=" * 76); print(f"MPC on {'custom terrain' if a.terrain else 'mixed demo course'} with {len(OBST)} cones + {len(DYN)} moving obstacles, target {vt} m/s")
    print(f"{'k':>6}{'result':>8}{'collisions':>12}{'curb':>6}{'RMS cte m':>11}{'max cte m':>11}{'slip %':>8}{'peak demand':>13}{'motor J':>9}")
    R = {}
    for kk in ks:
        r = R[kk] = simulate_mpc(course, kk, True, vt)
        print(f"{kk:>6g}{flag(r):>8}{r['hits']:>12}{r['curbs']:>6}{r['rms_cte']:>11.3f}{r['max_cte']:>11.3f}{r['slip_pct']:>8.1f}{r['max_util']:>13.2f}{r['W'][-1]:>9.3f}")
    if not a.quick:
        print("-" * 76); print("COMPARISON, no cones (RMS cross-track m)  - CNN stand-in (v3) vs MPC (v4)")
        print(f"{'k':>6}{'CNN stand-in':>16}{'MPC':>12}")
        cmp = {}
        for kk in ks:
            c = V3.simulate(course, kk); mp = simulate_mpc(course, kk, False, vt); cmp[kk] = (c, mp)
            print(f"{kk:>6g}{c['rms_cte']:>15.3f}{'*' if c['off_track'] else ' '}{mp['rms_cte']:>11.3f}{'*' if mp['off_track'] else ' '}")
        print("(* = left the track)"); print("-" * 76)
        print("MPC with cones on single flat surfaces:  result/collisions at k=1 and k=11.6")
        for sname in ["asphalt_dry", "asphalt_wet", "gravel_loose"]:
            row = [simulate_mpc(uniform(sname), kk, True, vt) for kk in (1, 11.6)]
            print(f"  {sname:14s}" + "".join(f"  k={kk:<5g}{flag(r):>5} / {r['hits']} hit(s), slip {r['slip_pct']:.0f}%" for kk, r in zip((1, 11.6), row)))
    # ---- figures
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    fig, ax = plt.subplots(figsize=(9, 4)); xs = np.linspace(0, 15, 400); ax.plot(xs, y_ref(xs), "k--", lw=1, label="reference path")
    for kk, col in [(ks[0], "tab:blue"), (ks[-1], "tab:red")]:
        ax.plot(R[kk]["x"], R[kk]["y"], color=col, lw=1.3, label=f"MPC, k={kk:g}")
    for ox, oy in R[ks[0]]["obst"]: ax.add_patch(plt.Circle((ox, oy), 0.07, color="orange", zorder=3))
    for d, col in zip(R[ks[0]]["dyn"], ["tab:olive", "tab:olive", "tab:purple"]):
        tt = R[ks[0]]["t"]; pp = np.array([dpos(d, t_) for t_ in tt[::50]]); ax.plot(pp[:, 0], pp[:, 1], ":", color=col, lw=1.2, label="moving: " + d["kind"])
    for n_, (px, py) in enumerate(R[ks[0]]["plans"]): ax.plot(px, py, color="tab:green", lw=0.8, alpha=0.6, label="MPC predicted plans" if n_ == 0 else None)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(fontsize=7, ncol=2); ax.set_title("Fig 1. MPC trajectory: cones (orange), moving obstacles (dotted), predicted plans (green)")
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "fig1_mpc_trajectory.png"))
    r = R[ks[0]]; fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(r["t"], r["v"], color="tab:green"); axs[0].axhline(vt, color="k", ls="--", lw=.8); axs[0].set_ylabel("speed (m/s)")
    axs[1].plot(r["t"], np.degrees(r["cmd"]), label="MPC command", color="tab:blue"); axs[1].plot(r["t"], np.degrees(r["delta"]), label="actual", color="tab:orange"); axs[1].set_ylabel("steering (deg)"); axs[1].legend(fontsize=7)
    axs[2].plot(r["t"], r["thr"], color="tab:purple"); axs[2].set_ylabel("MPC throttle")
    axs[3].plot(r["t"], r["util"], color="tab:red"); axs[3].axhline(1, color="k", ls="--", lw=.8); axs[3].set_ylabel("friction demand")
    axs[4].plot(r["t"], r["W"], color="tab:brown"); axs[4].set_ylabel("motor work (J)"); axs[4].set_xlabel("time (s)")
    axs[0].set_title("Fig 2. MPC outputs and vehicle response (k=1)"); fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "fig2_mpc_response.png"))
    fig, axs = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True)
    for kk, col in [(ks[0], "tab:blue"), (ks[-1], "tab:red")]:
        i_ = R[kk]["info"]; axs[0].plot(i_["t"], i_["J"], color=col, lw=1, label=f"k={kk:g}")
        axs[1].plot(i_["t"], i_["md"], color=col, lw=1); axs[2].plot(i_["t"], i_["um"], color=col, lw=1); axs[3].plot(i_["t"], i_["thr"], color=col, lw=1)
    axs[0].set_ylabel("best-plan cost J*"); axs[0].set_yscale("log"); axs[0].legend(fontsize=7)
    axs[1].axhline(0.14, color="k", ls="--", lw=.8); axs[1].set_ylabel("nearest obstacle in plan (m)"); axs[1].set_ylim(0, 1)
    axs[2].axhline(1, color="k", ls="--", lw=.8); axs[2].set_ylabel("predicted friction use")
    axs[3].set_ylabel("MPC throttle / brake"); axs[3].set_xlabel("time (s)")
    axs[0].set_title("Fig 4. MPC internals: what the controller predicts at each tick"); fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "fig4_mpc_internals.png"))
    if not a.quick:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(ks, [cmp[kk][0]["rms_cte"] for kk in ks], "o-", color="tab:red", label="CNN stand-in (sampled, quantised)")
        ax.plot(ks, [cmp[kk][1]["rms_cte"] for kk in ks], "s-", color="tab:blue", label="MPC (delay-compensated)")
        ax.set_xlabel("control-loop slowdown k"); ax.set_ylabel("RMS cross-track error (m)"); ax.legend(); ax.set_title("Fig 3. Effect of control-loop degradation (no cones)")
        fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "fig3_cnn_vs_mpc.png"))
    print("figures written to", os.path.abspath(a.outdir))

if __name__ == "__main__":
    main()
