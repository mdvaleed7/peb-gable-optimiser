"""
demo_two_span.py - example driver for peb_frame_model.py
Two-span multigable PEB frame, 2 x 24 m, eave 8 m, 1:10, bay 7.5 m, leaning interior column.
ALL SECTION SIZES ARE SEED VALUES AND ALL WIND Cpe ARE DEMO PLACEHOLDERS - replace before use.
"""
import json
import math
import os
import sys

import numpy as np

from peb_frame_model import (ULS_RULES, FrameInput, FrameModel, LinearFrame2D, Loads, Plates,
                             Profile, WindCase, design_wind_pressure, ll_patterns, RHO_STEEL)

OUT = sys.argv[1] if len(sys.argv) > 1 else "demo"
os.makedirs(OUT, exist_ok=True)


def P(tw, bf, tf, bf2=None, tf2=None):
    return Plates(tw, bf, tf, bf2 if bf2 else bf, tf2 if tf2 else tf)


# ------------------------------------------------------------------ wind (IS 875-3:2015)
pd, pz, Vz = design_wind_pressure(Vb=44, k1=1.0, k2=1.0, k3=1.0, k4=1.0, Kd=0.9, Ka=0.8, Kc=0.9)
# DEMO Cpe - placeholders only. Take walls from IS 875-3 wall table and roof pieces
# (R0..R3 = span-1 windward/leeward, span-2 windward/leeward) from the MULTISPAN roof table.
demo = dict(wx=([0.7, -0.3], [-0.9, -0.5, -0.4, -0.3]),
            wxn=([-0.3, 0.7], [-0.3, -0.4, -0.5, -0.9]),
            wlong=([-0.6, -0.6], [-0.8, -0.8, -0.8, -0.8]))
winds = []
for key, label in (("wx", "+X"), ("wxn", "-X"), ("wlong", "LONG")):
    walls, roof = demo[key]
    for cpi in (+0.2, -0.2):                   # IS 875-3 Cl. 7.3.2: +/-0.2 low permeability
        winds.append(WindCase(f"{label} CPI{cpi:+.1f}", cpi, walls[0], walls[1], roof))

# ------------------------------------------------------------------ seed sections (m)
SR_ext = Profile([(0.0, 0.80), (0.30, 0.45), (0.80, 0.55), (1.0, 0.55)],
                 [P(0.006, 0.20, 0.012), P(0.005, 0.18, 0.008), P(0.005, 0.18, 0.008)])
SR_int = Profile([(0.0, 0.90), (0.30, 0.45), (0.80, 0.55), (1.0, 0.55)], SR_ext.plates)
COL_E = Profile([(0.0, 0.30), (1.0, 0.80)], [P(0.006, 0.22, 0.012)])
COL_I = Profile([(0.0, 0.25), (1.0, 0.25)], [P(0.006, 0.20, 0.010)])

fi = FrameInput(
    name="PEB_2S", spans=[24.0, 24.0], eave_height=8.0, slope=0.1, roof="multigable", bay=7.5,
    purlin_spacing=1.5, stations_per_purlin=1, girt_spacing=1.5, interior_col_subdiv=4,
    base_ext="PINNED", base_int="PINNED", interior_columns="leaning", fy=345.0,
    rafter_SR_ext=SR_ext, rafter_SR_int=SR_int, col_ext=COL_E, col_int=COL_I,
    loads=Loads(q_sdl_slope=0.15, q_coll_plan=0.10, q_ll_plan=0.75, q_wall=0.10,
                sw_factor=1.0, pd_wind=pd, wind_cases=winds, cpe_verified=False))

mdl = FrameModel(fi)
for kind in ("ULS", "SLS", "BUCKLING"):
    tag = {"ULS": "ULS_PDELTA", "SLS": "SLS", "BUCKLING": "BUCKLING"}[kind]
    mdl.write_std(os.path.join(OUT, f"{fi.name}_{tag}.std"), kind)
smap = mdl.station_map()
with open(os.path.join(OUT, "station_map.json"), "w") as f:
    json.dump(smap, f, indent=1, default=float)

# ------------------------------------------------------------------ independent linear check
fe = LinearFrame2D(mdl)
prim = sorted(mdl.cases)
sol = {lc: fe.stations(fe.solve({lc: 1.0})) for lc in prim}       # unit primaries (superposition)
disp = {lc: fe.solve({lc: 1.0})["u"] for lc in prim}
reac = {lc: fe.solve({lc: 1.0})["R"] for lc in prim}


def comb_stations(pairs):
    out = {}
    for name in sol[1]:
        rows = []
        for j, st in enumerate(sol[1][name]):
            M = sum(f * np.mean(sol[lc][name][j]["M"]) for lc, f in pairs)
            N = sum(f * np.mean(sol[lc][name][j]["N"]) for lc, f in pairs)
            rows.append(dict(x=st["x"], y=st["y"], s=st["s"], D=st["D"], M=M, N=N))
        out[name] = rows
    return out


def comb_disp(pairs, nd):
    return sum(f * np.array(disp[lc][nd]) for lc, f in pairs)


roof = [l for l in mdl.lines if l["type"] == "raf"]
env = {l["name"]: dict(Mmax=np.full(len(l["stations"]), -np.inf), Mmin=np.full(len(l["stations"]), np.inf))
       for l in mdl.lines}
for c in mdl.uls:
    cs = comb_stations(c["pairs"])
    for name, rows in cs.items():
        Ms = np.array([r["M"] for r in rows])
        env[name]["Mmax"] = np.maximum(env[name]["Mmax"], Ms)
        env[name]["Mmin"] = np.minimum(env[name]["Mmin"], Ms)

grav = next(c for c in mdl.uls if c["title"].startswith("U01"))
uplift_all = [c for c in mdl.uls if "0.9D+1.5W" in c["title"]]
cs_uplift = {c["lc"]: comb_stations(c["pairs"]) for c in uplift_all}
worst_up = max(uplift_all, key=lambda c: max(r["M"] for l in roof for r in cs_uplift[c["lc"]][l["name"]]))
cs_g, cs_u = comb_stations(grav["pairs"]), cs_uplift[worst_up["lc"]]

# Horne sway estimate for the gravity combination: lambda_cr ~ (H/V)(h/delta_N) = h/(200 delta_N)
notional_pairs = [(lc, f) for lc, f in grav["pairs"] if lc == 301 or 311 <= lc < 400]
dN = max(abs(comb_disp(notional_pairs, mdl.col_top(c))[0]) for c in (0, len(fi.spans)))
lam_horne = fi.eave_height / (200 * dN)

# ------------------------------------------------------------------ preview figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

fig, ax = plt.subplots(3, 1, figsize=(8, 9.6), gridspec_kw=dict(height_ratios=[0.62, 0.72, 1.15]))
a0 = ax[0]
cols = {"col": "#4a6fa5", "raf": "#c0703a"}
for l in mdl.lines:
    nx, ny = l["outer_normal"]
    pts = [(s["x"], s["y"], s["D"]) for s in l["stations"]]
    outer = [(x + nx * D / 2, y + ny * D / 2) for x, y, D in pts]
    inner = [(x - nx * D / 2, y - ny * D / 2) for x, y, D in pts]
    a0.add_patch(Polygon(outer + inner[::-1], closed=True, fc=cols[l["type"]], ec="k", lw=0.4, alpha=0.55))
    xs = [s["x"] for s in l["stations"]]
    ys = [s["y"] for s in l["stations"]]
    a0.plot(xs, ys, "k-", lw=0.5)
    a0.plot(xs, ys, "|", color="k", ms=4, mew=0.6)
    for k in l["knots"]:
        st = l["stations"][k["j"]]
        a0.plot(st["x"], st["y"], "o", color="crimson", ms=4.5, zorder=5)
    mid = l["stations"][len(l["stations"]) // 2]
    off = (0.9 * nx, 0.9 * ny + (0.9 if l["type"] == "raf" else 0))
    a0.text(mid["x"] + off[0], mid["y"] + off[1], l["name"], fontsize=8, ha="center", va="center")
for nd, s in mdl.supports.items():
    x, y = mdl.nodes[nd]
    a0.plot(x, y - 0.35, "^" if s == "PINNED" else "s", color="k", ms=9)
for mid, end in mdl.releases:
    m = mdl.mem_index[mid]
    x, y = mdl.nodes[m["start"] if end == "START" else m["end"]]
    a0.plot(x, y - 0.55, "o", mfc="white", mec="k", ms=7, zorder=6)
a0.set_aspect("equal")
a0.set_xlim(-2, sum(fi.spans) + 2)
a0.set_ylim(-1.2, fi.eave_height + fi.slope * max(fi.spans) / 2 + 2.2)
a0.set_title(f"{fi.name}: {len(mdl.nodes)} nodes, {len(mdl.members)} tapered sub-members "
             f"(true-scale depth)\nred = knots (snapped to stations), ticks = stations/purlins, "
             f"o = MZ release (leaning column)", fontsize=9)
a0.set_xlabel("x (m)", fontsize=8)
a0.tick_params(labelsize=7)

a1 = ax[1]
for l in roof:
    a1.plot([s["x"] for s in l["stations"]], [1000 * s["D"] for s in l["stations"]], "-", color="#c0703a")
    for k in l["knots"]:
        st = l["stations"][k["j"]]
        a1.plot(st["x"], 1000 * st["D"], "o", color="crimson", ms=4)
a1.set_ylabel("rafter depth D (mm)", fontsize=8)
a1.set_title("Rafter depth profile: support -> valley (min) -> sagging zone -> ridge", fontsize=9)
a1.grid(alpha=0.3)
a1.tick_params(labelsize=7)

a2 = ax[2]
for l in roof:
    xs = [s["x"] for s in l["stations"]]
    a2.fill_between(xs, env[l["name"]]["Mmin"], env[l["name"]]["Mmax"], color="0.85", lw=0)
    a2.plot(xs, [r["M"] for r in cs_g[l["name"]]], "-", color="#1f4e79", lw=1.4)
    a2.plot(xs, [r["M"] for r in cs_u[l["name"]]], "--", color="#2e7d32", lw=1.4)
    for k in l["knots"]:
        a2.axvline(l["stations"][k["j"]]["x"], color="crimson", lw=0.5, alpha=0.5)
a2.axhline(0, color="k", lw=0.6)
a2.plot([], [], "-", color="#1f4e79", label=grav["title"])
a2.plot([], [], "--", color="#2e7d32", label=worst_up["title"])
a2.fill_between([], [], [], color="0.85", label="envelope, all ULS combos")
a2.legend(fontsize=7, loc="lower center")
a2.set_ylabel("M_ic (kNm)  + = inner flange in compression", fontsize=8)
a2.set_xlabel("x along frame (m)", fontsize=8)
a2.set_title("Rafter moment (independent LINEAR check, not STAAD P-Delta): knots vs envelope valleys",
             fontsize=9)
a2.grid(alpha=0.3)
a2.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "frame_preview.png"), dpi=150)

# ------------------------------------------------------------------ summary
W = mdl.steel_weight()
Wtot = sum(W.values())
area = sum(fi.spans) * fi.bay
L = []
L.append(f"{fi.name} - model summary (generated by demo_two_span.py)")
L.append("=" * 78)
L.append(f"Spans {fi.spans} m | eave {fi.eave_height} m | slope 1:{1 / fi.slope:g} | roof {fi.roof} | "
         f"bay {fi.bay} m")
L.append(f"Interior columns {fi.interior_columns} | bases ext {fi.base_ext}, int {fi.base_int}")
L.append(f"Wind IS 875-3: Vz {Vz:.1f} m/s, pz {pz:.4f} kN/m2, pd = max(Kd Ka Kc pz, 0.7 pz) = {pd:.4f} kN/m2")
if not fi.loads.cpe_verified:
    L.append("!!! Cpe values are DEMO placeholders - replace with IS 875-3 wall + multispan roof tables !!!")
L.append("")
L.append("LINES  (knots snapped to stations; D in mm)")
for l in mdl.lines:
    kn = ", ".join(f"s={k['s']:.2f} m D={1000 * k['D']:.0f}" for k in l["knots"])
    L.append(f"  {l['name']:<3} {l['kind']:<6} L={l['length']:6.3f} m  n_sub={l['n_sub']:<3} "
             f"mass={W[l['name']]:7.1f} kg  | {kn}")
L.append(f"Total steel (members only) {Wtot:.0f} kg = {Wtot / area:.2f} kg/m2 of plan "
         f"({sum(fi.spans)} x {fi.bay} m)")
L.append(f"Members reversed to satisfy STAAD f1 >= f3: {sum(m['reversed'] for m in mdl.members)} "
         f"of {len(mdl.members)}")
L.append("")
L.append("PRIMARY LOAD CASES (totals for statics check; self-weight = own estimate)")
for lc in prim:
    t = mdl.case_totals(lc)
    L.append(f"  {lc:>4}  {mdl.cases[lc]['title']:<36} sumFX {t['FX']:9.3f}  sumFY {t['FY']:9.3f} kN")
L.append("")
pats = ll_patterns(len(fi.spans))
L.append(f"Live-load patterns: {pats}")
L.append(f"ULS REPEAT LOAD combos: {len(mdl.uls)} | SLS combos: {len(mdl.sls)} | "
         f"buckling combos: {sum(c['gravity_only'] for c in mdl.uls)}")
for c in mdl.uls:
    L.append(f"  {c['lc']}  {c['title']}")
L.append("")
L.append("BENCHMARKS - independent linear solver (compare with STAAD SLS/linear results;")
L.append("expect a few % difference: this uses mid-length properties per sub-member)")
R1 = reac[1]
for nd in sorted(R1):
    L.append(f"  DL reaction node {nd:>3}: FX {R1[nd][0]:8.3f}  FY {R1[nd][1]:8.3f} kN")
for name, rows in cs_g.items():
    if name.startswith("R"):
        L.append(f"  {grav['title']:<30} {name}: M_ic start {rows[0]['M']:8.1f}, "
                 f"end {rows[-1]['M']:8.1f}, N start {rows[0]['N']:7.1f} kN(m)")
comp = {c: abs(disp[401 + c][mdl.col_top(c)][0]) for c in (0, len(fi.spans))}
L.append(f"  eave lateral compliance (unit FX at column top): "
         + ", ".join(f"C{c} {1000 * v:.2f} mm/kN" for c, v in comp.items())
         + "  <- Cpe-independent stiffness benchmark")
for w_lc in mdl.lc_wl:
    ux = max(abs(disp[w_lc][mdl.col_top(c)][0]) for c in range(len(fi.spans) + 1))
    L.append(f"  eave |ux| {mdl.cases[w_lc]['title']:<22}: {1000 * ux:6.1f} mm "
             f"(H/{fi.eave_height / ux:.0f}; Table 6 no-crane elastic cladding H/150)")
L.append("    (net drift = wall push minus sway from asymmetric roof uplift; with DEMO Cpe the two")
L.append("     largely cancel - do not read drift adequacy from these numbers)")
L.append(f"  Horne sway estimate for {grav['title']}: lambda_cr ~ h/(200 dN) = "
         f"{fi.eave_height:.1f}/(200 x {1000 * dN:.2f} mm) = {lam_horne:.1f}")
L.append("    (valid while rafter compression is small; compare with STAAD buckling factor)")
L.append("")
L.append("KNOTS vs LINEAR ULS ENVELOPE (step U1 preview; re-run with STAAD P-Delta envelope)")
for l in roof:
    e = env[l["name"]]
    absM = np.maximum(np.abs(e["Mmax"]), np.abs(e["Mmin"]))
    j_v = int(np.argmin(absM[1:-1])) + 1
    j_s = int(np.argmin(e["Mmin"]))
    st = l["stations"]
    L.append(f"  {l['name']}: knots at s = {[round(k['s'], 2) for k in l['knots']]} m | envelope valley "
             f"s = {st[j_v]['s']:.2f} m (|M|={absM[j_v]:.1f}) | peak sagging s = {st[j_s]['s']:.2f} m "
             f"(M_ic={e['Mmin'][j_s]:.1f})")
L.append("")
L.append("WARNINGS")
L += [f"  - {w}" for w in mdl.warnings] or ["  - none"]
L.append("  - Section sizes are SEED values; load factors are the ULS_RULES table (verify IS 800 Table 4).")
L.append("  - Roof LL 0.75 kN/m2 assumed (IS 875-2 Table 2, slope <= 10 deg, no access) - verify.")
L.append("  - Knee/apex connection eccentricities and rigid zones are not modelled (centre-line model).")
with open(os.path.join(OUT, "model_summary.txt"), "w") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L[:12]))
print("...")
print(f"lambda_cr (Horne) = {lam_horne:.2f}; worst uplift combo {worst_up['title']}")
