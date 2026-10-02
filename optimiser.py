"""
optimiser.py - IS 800:2007 weight optimiser for the multi-span PEB gable frame (peb_frame_model).

Inner loop (fast, no STAAD):
    design (plate catalogue indices) -> FrameModel -> LinearFrame2D (unit primaries)
    -> P-Delta per ULS combo: (K - Kg(N)) u = F, N updated until converged (as STAAD PDELTA)
    -> is800 ratios at both ends of every sub-member
    -> SLS (first order): rafter deflection relative to column tops (LL combos), eave drift (wind combos)
    -> lambda_cr of the gravity combos (eigenvalue) for the report.
Search (discrete, greedy):
    repair  : step up the variable with the largest violation drop per kg until feasible
    trim    : step each variable down while it stays feasible
    shake   : step one variable down and repair; keep it if the frame gets lighter
Verify (STAAD.Pro, headless SProStaad): ULS P-Delta + SLS files with PRINT MEMBER FORCES / JOINT
    DISPLACEMENTS and STAAD's own IS800 LSD CHECK CODE. Our checks are re-run on STAAD's forces; if
    they exceed the target the target is tightened by the ratio and the search is repeated.

OPTIMISATION PROBLEM (discrete)
    x = catalogue indices of the design variables: knot depths D and segment plates (tw, bf, tf) of every
        active template (raf_ext, raf_int, raf_ss, col_ext, col_int)
    min  f(x) = rho * sum_members integral A(s) ds  +  m_stiffeners  +  p * n_stiffeners
    s.t. g_ULS  : u(i, c) = max IS 800 D/C at station i, combination c        <= eta   (dc_target)
         g_SLS  : rafter deflection / (span / limit), eave drift / (H / limit)  <= 1
         g_stiff: stiffener force F / Fqd (8.7.2.5, 8.5.1 end stiffener)        <= 1
         h_joint: at every node where two roof pieces meet (ridge, valley, single-ridge column)
                  D_left(end) = D_right(end) and (tw, bf, tf)_left(end seg) = (tw, bf, tf)_right(end seg)
                  -> the two rafters mirror each other at the apex splice; each side keeps its own taper,
                     so the interior (valley) haunch may be deeper than the eave haunch
         h_bf   : options.constant_bf -> one flange width per template (plate stock per member)
         x in catalogue CAT (bounds and standard plate sizes)
    Six-section rafters (options.rafter_six_sections, templates raf_ext / raf_int):
         knots k0..k6 from the ULS |M| envelope (six_knots): k0 haunch, k1 / k2 at 1/3 and 2/3 of the moment drop,
         k3..k4 the low-moment zone (|M| <= M_min + 0.2 (M_max - M_min)), k5 halfway up to the ridge moment, k6 ridge
         h_zone : D3 = D4 (uniform section in the low-moment zone)
         h_pl   : plates (tw, bf, tf) equal in segments 0-1, 2-3, 4-5 (plate changes only at k2 and k4)
         g_shape: D0 >= D1 >= D2 >= D3,  D4 <= D5 <= D6 <= min(D0 of the rafter templates)
         g_apex : D5 >= D4 + (D6 - D4)(s5 - s4)/(s6 - s4)  (taper toward the ridge never steepens: no apex spike)
         g_taper: |D(k+1) - D(k)| / length <= limits.taper_max (mm/m) on every segment except the haunch one
         the g_ are enforced by projection in decode() (_shape), the h_ by linking; knots are re-derived from the
         envelope of the optimised design and the search is repeated once when they move.
    Cranes (inp.crane, IS 875-2 6.3 / 6.4, IS 800 Table 4 / 6): bracket cantilevers are members like any other;
         g_crane: rail drift / (H_rail / crane_drift) <= 1 and |u_R - u_L| / rail_spread <= 1 under CL and 0.8(CL+W)
    Equalities are eliminated by linking: linked slots are one design variable (variables()).
    Inequalities are handled by the repair / trim / shake search (total violation above eta -> 0).
Stiffeners and tension field are decided per sub-member inside check_design (not design variables).

ponytail: SLS covers LL-only and wind-only combos (Table 6 style); D+L deflection is not limited.
"""
from __future__ import annotations

import copy
import functools
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

# One BLAS thread per process: the solves are small (~150-500 DOF) and the search already runs one process
# per core; a multithreaded BLAS in every worker (OpenBLAS / Intel MKL, e.g. Anaconda's numpy, / OpenMP /
# Apple Accelerate) oversubscribes the CPU - measured 100x slower (4 workers x 4 BLAS threads).
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np                                    # noqa: E402
try:                                                  # numpy already loaded by the caller: limit it at runtime
    from threadpoolctl import threadpool_limits       # noqa: E402
    threadpool_limits(1)
except ImportError:
    pass
from scipy.linalg import LinAlgError, eigh  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, LinearOperator, eigsh  # noqa: E402

from banded import BandedSPD  # noqa: E402

from is800 import (GM0, angle_capacities, angle_ratios, angle_ratios_v, capacities, double_angle, end_post, ratios,
                   ratios_v, shear_capacity, stiffener, stiffener_fqd, tf_weld, tube_capacities, tube_ratios, tube_ratios_v,
                   web_point_load)
from peb_frame_model import (RHO_STEEL, Crane, FrameInput, FrameModel, LinearFrame2D, Loads, Plates, Profile, Truss,
                             WindCase, _ranges, design_wind_pressure, end_forces_to_stations)

# plate catalogue, mm
CAT = dict(D=list(range(200, 1501, 25)), tw=[4, 5, 6, 8, 10, 12, 14, 16],
           bf=list(range(125, 451, 25)), tf=[5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32])
# portal truss: IS 4923 tubes (tubes_is4923.json, from STAAD's IS 4923-2017 database; sorted by kg/m)
TUBES = json.loads((Path(__file__).with_name("tubes_is4923.json")).read_text())["tubes"]
TUBE = {t["name"]: t for t in TUBES}
ANGLE1 = {a["name"]: a for a in json.loads((Path(__file__).with_name("angles_is808.json")).read_text())["angles"]}
CAT.update(tube=[t["name"] for t in TUBES], angle=[], h0=list(range(400, 2601, 100)), bs=[0.0, 0.5, 1.0],
           web=["pratt", "howe"])
ROLES = ("top", "bot", "vert", "diag")
SCHEME_NAMES = dict(peb="PEB tapered I", truss="Truss SHS/RHS", angle="Truss 2-angles")


@functools.lru_cache(maxsize=None)
def angle_chain(fy: float, tg: float) -> tuple:
    """Double equal angles usable as truss members at fy: Table 2 axial compression class 3 ((b + d)/t <= 25 eps),
    chained by kg/m like the tubes (each step +2 % area, Ze not lower, r_min >= 0.98 x best lighter)."""
    eps, chain, rbest = math.sqrt(250.0 / fy), [], 0.0
    for d in sorted((double_angle(a, tg) for a in ANGLE1.values() if 2 * a["b"] / a["t"] <= 25 * eps + 1e-9),
                    key=lambda d: d["w"]):
        r = math.sqrt(min(d["Iz"], d["Iy"]) / d["A"])
        if not chain or (d["w"] > chain[-1]["w"] and d["A"] > 1.02 * chain[-1]["A"] and r >= 0.98 * rbest
                         and d["Zez"] >= chain[-1]["Zez"]):
            chain.append(d)
            rbest = max(rbest, r)
    return tuple(chain)


def truss_family(inp: dict) -> str:
    return inp.get("truss", {}).get("family", "tube")


def gusset(inp: dict) -> float:
    return float(inp.get("truss", {}).get("gusset_mm", 8.0)) / 1000


def sec_props(inp: dict, name: str) -> dict:
    """Truss section by name: IS 4923 tube or '2-ISEA ...' double angle on the input's gusset."""
    return TUBE[name] if name in TUBE else double_angle(ANGLE1[name[2:]], gusset(inp))


def _name_w(name: str) -> float:
    return TUBE[name]["w"] if name in TUBE else 2 * ANGLE1[name[2:]]["w"]
ENGINES = [Path(r"C:\Program Files\Bentley\Engineering\STAAD.Pro 2026\STAAD\SProStaad\SProStaad.exe"),
           Path(r"C:\Program Files\Bentley\StaadPro\STAAD\SProStaad\SProStaad.exe")]
TEMPLATE_NAMES = dict(raf_ext="Rafter, exterior support -> ridge", raf_int="Rafter, interior support -> ridge",
                      raf_ss="Rafter, support -> support", col_ext="Exterior column, base -> top",
                      col_int="Interior column, base -> top", bracket="Crane bracket, column -> tip",
                      truss="Portal truss (IS 4923 tubes)")

DEFAULT_INPUT = dict(
    name="PEB_2S", spans=[24.0, 24.0], eave_height=8.0, slope=0.1, roof="multigable", bay=7.5,
    purlin_spacing=1.5, girt_spacing=1.5, interior_columns="leaning", base_ext="PINNED", base_int="PINNED",
    fy=345.0, fu=490.0,
    loads=dict(q_sdl_slope=0.15, q_coll_plan=0.10, q_ll_plan=0.75, q_wall=0.10, sw_factor=1.0),
    wind=dict(Vb=44.0, k1=1.0, k2=1.0, k3=1.0, k4=1.0, Kd=0.9, Ka=0.8, Kc=0.9, cpi=[0.2, -0.2],
              cpe_verified=False,
              cases=[dict(label="+X", walls=[0.7, -0.3], roof=[-0.9, -0.5, -0.4, -0.3]),
                     dict(label="-X", walls=[-0.3, 0.7], roof=[-0.3, -0.4, -0.5, -0.9]),
                     dict(label="LONG", walls=[-0.6, -0.6], roof=[-0.8, -0.8, -0.8, -0.8])]),
    restraint=dict(fly_every=2),
    # purlin load into the rafter web (8.7.3.1 / 8.7.4): b1 = stiff bearing length of the purlin cleat on the
    # flange (8.7.1.3; 8 mm cleat + 2 x 6 mm welds), web_KL = KL/d of the web strut (8.7.1.5: 0.7 restrained, 1.0 not)
    purlin=dict(b1_mm=20.0, web_KL=1.0),
    # EOT crane on column brackets (IS 875-2 6.3): spans = 1-based span numbers with a crane ([] = no crane);
    # loads in kN, lengths in m; impact 6.3(a) (0.25 class III/IV, 0.10 class I/II columns); surge 6.3(c)
    scheme="peb",              # "peb" tapered welded I rafters | "truss" portal truss (optimise_both runs both)
    # portal truss alternative: IS 4923 cold-formed tubes, IS 800 7.2.4 lengths, Table 10 curve b
    truss=dict(compare="always",   # "always" optimise every scheme | "on_trigger" trusses only if the PEB crosses peb_limits
               family="both",       # "tube" SHS/RHS IS 4923 | "angle" double angles IS 808 on gussets | "both"
               K=1.0,               # in-plane effective length / distance between connections, 7.2.4: 0.7 - 1.0
               fly_every=2,         # bottom chord held out of plane at every n-th panel point (ties / fly braces)
               gusset_mm=8.0),      # double angles: gusset between the angles
    # ONE steel grade for every scheme (fy / fu above): plates, IS 4923 tubes (order the YSt grade with yield >= fy)
    # and IS 808 angles - so the schemes are compared on the same material
    # installed cost per frame. PLACEHOLDER INR RATES - replace with your fabricator's and erector's rates
    cost=dict(plate_rs_kg=65.0, tube_rs_kg=75.0, angle_rs_kg=68.0, fab_builtup_rs_kg=20.0, fab_truss_rs_kg=25.0,
              stiffener_rs=300.0, joint_rs=500.0, gusset_kg=6.0, erect_rs_kg=10.0, paint_rs_m2=120.0),
    # PEB economic / fabrication limits: any one crossed -> reported, and the truss is run when compare = on_trigger
    peb_limits=dict(D_max_mm=1200, plate_max_mm=25, kg_m2_max=20.0, stiffeners_max=20, lam_cr_min=5.0),
    crane=dict(spans=[], bracket_level=6.0, e=0.6, capacity_kN=98.1, bridge_kN=110.0, crab_kN=25.0, a_min=1.0,
               wheel_base=3.2, wheels_per_rail=2, gantry_kNm=1.2, gantry_depth=0.6, impact=0.25, surge=0.05),
    options=dict(slender_web=False,      # True: Class 4 webs allowed via IS 800 8.2.1.1(a) + 8.6.1 limits
                 stiffeners=False,       # True: transverse web stiffeners where needed (8.4.2.2(a), 8.6.1(b), 8.7.2)
                 stiffener_penalty_kg=0.0,   # fabrication cost of one stiffener, in kg of steel (objective only)
                 tension_field=False,    # True: 8.4.2.2(b) in anchored stiffened panels (needs stiffeners)
                 end_posts=False,        # True: 8.5.2(b) end posts let knee / haunch panels use tension field
                 constant_bf=True,       # practical: one flange width per template (member)
                 rafter_six_sections=True,   # six tapered sections from the moment envelope, shape rules
                 screen_combos=True),        # search screens with the governing ULS combinations (> 40 combinations)
    limits=dict(dc_target=1.0, rafter_defl=180.0, eave_drift=150.0, crane_drift=200.0, rail_spread_mm=10.0,
                taper_max=150.0,      # mm depth change per m, six-section segments outside the haunch segment
                time_limit_min=60.0),  # per scheme: stop and keep the best feasible design found (0 = no limit)
    # knots: s/L along the template; D in mm at each knot; plates [tw, bf, tf] mm per segment
    templates=dict(
        raf_ext=dict(knots=[0, 0.1, 0.2, 0.35, 0.55, 0.75, 1], D=[800, 700, 600, 450, 450, 500, 550],
                     plates=[[6, 200, 12], [6, 200, 12], [5, 200, 8], [5, 200, 8], [5, 200, 8], [5, 200, 8]]),
        raf_int=dict(knots=[0, 0.1, 0.2, 0.35, 0.55, 0.75, 1], D=[900, 750, 600, 450, 450, 500, 550],
                     plates=[[6, 200, 12], [6, 200, 12], [5, 200, 8], [5, 200, 8], [5, 200, 8], [5, 200, 8]]),
        raf_ss=dict(knots=[0, 0.5, 1], D=[800, 450, 800], plates=[[6, 200, 12], [6, 200, 12]]),
        col_ext=dict(knots=[0, 1], D=[300, 800], plates=[[6, 225, 12]]),
        col_int=dict(knots=[0, 1], D=[250, 250], plates=[[6, 200, 10]]),
        bracket=dict(knots=[0, 1], D=[450, 300], plates=[[6, 200, 12]]),
        # h0 mm, bottom chord slope factor, web type; tubes per half-truss side: [top, bottom, verticals, diagonals]
        truss=dict(h0=[1200], bs=[0.0], web=["pratt"], ext=["SHS 113.5x113.5x4", "SHS 113.5x113.5x4", "SHS 60x60x3",
                                                             "SHS 60x60x3"],
                   int=["SHS 113.5x113.5x4", "SHS 113.5x113.5x4", "SHS 60x60x3", "SHS 60x60x3"])))


# ============================================================================= input -> model
def _profile(t: dict) -> Profile:
    return Profile([(float(s), d / 1000) for s, d in zip(t["knots"], t["D"])],
                   [Plates(tw / 1000, bf / 1000, tf / 1000, bf / 1000, tf / 1000) for tw, bf, tf in t["plates"]])


def frame_input(inp: dict, design: dict) -> FrameInput:
    w = inp["wind"]
    pd, _, _ = design_wind_pressure(w["Vb"], w["k1"], w["k2"], w["k3"], w["k4"], w["Kd"], w["Ka"], w["Kc"])
    winds = [WindCase(f"{c['label']} CPI{cpi:+.1f}", cpi, c["walls"][0], c["walls"][1], list(c["roof"]))
             for c in w["cases"] for cpi in w["cpi"]]
    t = {k: _profile(v) for k, v in design.items() if k != "truss"}
    tr = None
    if inp.get("scheme") == "truss":
        T = design["truss"]
        tr = Truss(h0=T["h0"][0] / 1000, bs=float(T["bs"][0]), web=T["web"][0],
                   sections={sd: {r: sec_props(inp, n) for r, n in zip(ROLES, T[sd])} for sd in ("ext", "int") if sd in T})
    cr = inp.get("crane", {})
    cranes = [Crane(int(sp) - 1, cr["bracket_level"], cr["e"], cr["capacity_kN"], cr["bridge_kN"], cr["crab_kN"],
                    cr["a_min"], cr["wheel_base"], int(cr["wheels_per_rail"]), cr["gantry_kNm"], cr["gantry_depth"],
                    cr["impact"], cr["surge"]) for sp in cr.get("spans", [])]
    return FrameInput(
        name=inp["name"], spans=list(inp["spans"]), eave_height=inp["eave_height"], slope=inp["slope"],
        roof=inp["roof"], bay=inp["bay"], purlin_spacing=inp["purlin_spacing"], girt_spacing=inp["girt_spacing"],
        base_ext=inp["base_ext"], base_int=inp["base_int"], interior_columns=inp["interior_columns"], fy=inp["fy"],
        rafter_SR_ext=t.get("raf_ext"), rafter_SR_int=t.get("raf_int"), rafter_SS=t.get("raf_ss"),
        col_ext=t.get("col_ext"), col_int=t.get("col_int"), bracket=t.get("bracket"), cranes=cranes, truss=tr,
        loads=Loads(pd_wind=pd, wind_cases=winds, cpe_verified=w["cpe_verified"], **inp["loads"]))


def active_templates(inp: dict) -> list:
    """Templates the geometry actually uses (from the roof pieces of a seed model)."""
    mdl = FrameModel(frame_input(inp, inp["templates"]))
    used = {"col_ext"} | ({"col_int"} if len(inp["spans"]) > 1 else set())
    used |= {"bracket"} if inp.get("crane", {}).get("spans") else set()
    for pc in ([] if inp.get("scheme") == "truss" else mdl.pieces):
        used.add({"S-R": "raf_int" if pc["int_a"] else "raf_ext", "R-S": "raf_int" if pc["int_b"] else "raf_ext",
                  "S-S": "raf_ss"}[pc["kind"]])
    return [k for k in TEMPLATE_NAMES if k in used | ({"truss"} if inp.get("scheme") == "truss" else set())]


def truss_sides(inp: dict) -> list:
    """'int' exists when a roof piece touches no eave column: more than 2 pieces (multigable 2 per span,
    single ridge 1 per span + 1 when the ridge is not on a column)."""
    X = [0.0]
    for L_ in inp["spans"]:
        X.append(X[-1] + L_)
    n = (2 * len(inp["spans"]) if inp["roof"] == "multigable"
         else len(inp["spans"]) + (0 if any(abs(X[-1] / 2 - x) < 1e-6 for x in X) else 1))
    return ["ext", "int"] if n > 2 else ["ext"]


def _truss_lengths(inp: dict) -> tuple:
    t = inp.get("truss", {})
    return float(t.get("K", 1.0)), int(t.get("fly_every", 2))


def n_roof_pieces(inp: dict) -> int:
    return len(FrameModel(frame_input(inp, inp["templates"])).pieces)


def slender_web(inp: dict) -> bool:
    return bool(inp.get("options", {}).get("slender_web", False))


def stiffeners_on(inp: dict) -> bool:
    return bool(inp.get("options", {}).get("stiffeners", False))


def tension_field_on(inp: dict) -> bool:
    return stiffeners_on(inp) and bool(inp.get("options", {}).get("tension_field", False))


def end_posts_on(inp: dict) -> bool:
    return tension_field_on(inp) and bool(inp.get("options", {}).get("end_posts", False))


def knee_ends(mdl: FrameModel, ln: dict) -> set:
    """Line ends at a knee or rafter haunch: support end(s) of a rafter piece (eave knee, valley haunch) and
    the top of an exterior (or rigid interior) column. Ridge and base ends are not knees."""
    if ln["type"] in ("brk", "bot", "web"):
        return set()
    if ln["type"] == "col":
        return {"end"} if ln["kind"] != "int" or mdl.fi.interior_columns == "rigid" else set()
    return {"S-R": {"start"}, "R-S": {"end"}, "S-S": {"start", "end"}}[ln["kind"]]


def purlin_reactions(mdl: FrameModel) -> dict:
    """Roof load delivered by each purlin, per primary case: {(line, j): {lc: kN}} for interior purlin stations
    of rafter lines (line ends carry connection plates). + = pulls the outer flange outward (uplift), - = pushes
    it into the web. Component normal to the rafter; the rafter's own self-weight is not a purlin load."""
    out = {}
    for lc, case in mdl.cases.items():
        for it in case["items"]:
            if it[0] != "M":
                continue
            m = mdl.mem_index[it[1]]
            ln = mdl.lines[m["line"]]
            if ln["type"] != "raf" or m.get("sec"):                       # truss top chord: purlins on the chord
                continue
            gx, gy = mdl.load_vector(m, it[2], it[3])
            qn = gx * ln["outer_normal"][0] + gy * ln["outer_normal"][1]
            for j in (m["j"], m["j"] + 1):
                if 0 < j < ln["n_sub"] and ln["stations"][j]["purlin"]:
                    d = out.setdefault((ln["name"], j), {})
                    d[lc] = d.get(lc, 0.0) + qn * m["L"] / 2
    return out


def restraint_lengths(mdl: FrameModel, fly_every: int, K: float = 1.0, bot_every: int = 2) -> dict:
    """{member: (L_in, L_out, L_y, L_z)}. Outer flange held by purlins (rafters) / girts (exterior columns);
    inner flange by fly braces at every fly_every-th of those; line ends restrain both flanges.
    Interior columns: ends only. L_z = line length node to node (column height, support -> ridge):
    the non-sway length, because sway effects are in the P-Delta member forces.
    Truss members (7.2.4, 7.5.2.1, 7.5.3): in plane K x distance between connections (double-angle webs
    min(K, 0.85), 7.5.2.1: 0.7 - 0.85); out of plane the distance between restraints - purlins (top chord),
    every bot_every-th panel point (bottom chord), member length (webs)."""
    out = {}
    for ln in mdl.lines:
        n, st = ln["n_sub"], ln["stations"]
        if mdl.mem_index[ln["members"][0]].get("sec"):
            S = sorted(set([j for j in range(n + 1) if st[j]["purlin"]] if ln["type"] == "raf" else
                           list(range(0, n + 1, max(1, bot_every))) if ln["type"] == "bot" else []) | {0, n})
            for mid in ln["members"]:
                j = mdl.mem_index[mid]["j"]
                Ly = st[min(x for x in S if x >= j + 1)]["s"] - st[max(x for x in S if x <= j)]["s"]
                Kw = min(K, 0.85) if ln["type"] == "web" and mdl.mem_index[mid]["sec"]["shape"] == "DA" else K
                out[mid] = (Ly, Ly, Ly, Kw * mdl.mem_index[mid]["L"])
            continue
        if ln["type"] == "brk":        # ponytail: cantilever, 2 x length for LTB and buckling (tip held by the
            for mid in ln["members"]:  # gantry girder) - check IS 800 8.3.3 / Table 16 for the actual detail
                out[mid] = (2 * ln["length"],) * 4
            continue
        if ln["type"] == "raf":
            outer = [j for j in range(n + 1) if st[j]["purlin"]]
        else:
            outer = [] if ln["kind"] == "int" else list(range(n + 1))
        outer = sorted(set(outer) | {0, n})
        inner = sorted(set(outer[::fly_every] if fly_every > 0 and ln["kind"] != "int" else []) | {0, n})
        Lz = ln["length"]
        for mid in ln["members"]:
            j = mdl.mem_index[mid]["j"]

            def bracket(S):
                return st[min(s for s in S if s >= j + 1)]["s"] - st[max(s for s in S if s <= j)]["s"]
            out[mid] = (bracket(inner), bracket(outer), bracket(inner), Lz)
    return out


# ============================================================================= checks
K_STIFF = (1, 2, 3)     # stiffener spacing c = sub-member length / k  (k = 1: at the purlin / girt stations)


VEC_MIN = 24          # points per station above which the IS 800 checks run vectorised (numpy call overhead)


def check_design(mdl: FrameModel, inp: dict, ends, disp, stiff: dict | None = None, posts=None, uls=None) -> dict:
    """ends(lc) -> {member: (start[6], end[6])} STAAD local, second order; disp(lc) -> {node: (ux, uy)} m.
    Each sub-member is one decision: k (stiffeners at spacing L/k, 0 = none) and the shear method, 8.4.2.2(a)
    simple post-critical or (b) tension field. The cheapest passing choice is taken in the order
    (0,a), (1,a), (1,b), (2,a), (2,b), (3,a), (3,b). Then:
      anchorage : every chain of tension-field panels must end in a panel designed by (a) that passes the
                  8.5.1 beam check for R_tf = Hq/2 and M_tf = Hq d/10 (8.5.3), or - at a knee / haunch with
                  options.end_posts - in a designed 8.5.2(b) double-stiffener end post; a chain reaching any
                  other support (ridge, base) or failing its check drops its end member to (a)
      stiffeners: sized for Is (8.7.2.4) and F <= Fqd, F = V - Vcr/gm0 (8.7.2.5) in tension-field panels,
                  plus M_tf / c_Q on the end stiffener of each anchoring panel (8.5.1)
    Candidates are ranked by stiffener count, an end post counting as one more stiffener, and (a) before
    (b) on a tie - so tension field + end post is only used when it saves a stiffener or passes where (a) fails.
    stiff: fixed {member: k} and posts: members with an end post (STAAD verify) - same physical stiffeners,
    the (a)/(b) choice is re-made, and knee tension field only where the design has the end post."""
    target, fy = inp["limits"]["dc_target"], inp["fy"]
    sw, st_on, tf_on = slender_web(inp), stiffeners_on(inp), tension_field_on(inp)
    knee = {ln["name"]: knee_ends(mdl, ln) for ln in mdl.lines} if end_posts_on(inp) else {}
    Ls = restraint_lengths(mdl, int(inp["restraint"]["fly_every"]), *_truss_lengths(inp))
    E = {c["lc"]: ends(c["lc"]) for c in (mdl.uls if uls is None else uls)}
    pts = {m["id"]: [(j, N, V, M, lc) for lc, Ec in E.items() for j, N, V, M in end_forces_to_stations(m, *Ec[m["id"]])]
           for m in mdl.members}
    pts_all = pts
    trials = {}

    def trial(m, k, tf):
        key = (m["id"], k, tf)
        if key in trials:
            return trials[key]
        mid, ln, P = m["id"], mdl.lines[m["line"]], Plates(**m["plates"])
        c = m["L"] / k if k else None
        d_web = max(m["D_a"], m["D_b"]) - P.tf_in - P.tf_out
        sd = stiffener(d_web, P.tw, min(P.bf_in, P.bf_out), fy, c) if k else None
        if (k and sd is None) or (tf and c < d_web):                    # 8.4.2.2(b) needs c/d >= 1
            trials[key] = None
            return None
        caps = {j: capacities(ln["stations"][j]["D"], P, fy, *Ls[mid], sw, c, tf) for j in {pt[0] for pt in pts[mid]}}
        worst, vmax, vtf = {}, 0.0, math.inf
        if len(pts[mid]) <= VEC_MIN * len(caps):   # few combinations (screened search): scalar checks
            for j, N, V, M, lc in pts[mid]:
                r = ratios(caps[j], N, V, M)
                chk = max(r, key=r.get)
                if j not in worst or r[chk] > worst[j]["ratio"]:
                    worst[j] = dict(line=m["line_name"], member=mid, j=j, D_mm=round(ln["stations"][j]["D"] * 1000),
                                    check=chk, ratio=r[chk], lc=lc, N=N, V=V, M=M, stiff_k=k, tension_field=tf)
                vmax = max(vmax, abs(V))
        else:                                      # every combination of a station at once (numpy)
            Pt = np.array(pts[mid], dtype=float)
            J = Pt[:, 0].astype(int)
            ck, val, names = np.zeros(len(J), dtype=int), np.zeros(len(J)), None
            for jj in caps:
                sel = np.nonzero(J == jj)[0]
                r = ratios_v(caps[jj], Pt[sel, 1], Pt[sel, 2], Pt[sel, 3])
                names = names or list(r)
                R = np.stack([r[q] for q in names])
                ck[sel] = np.argmax(R, axis=0)     # first maximum, as max(r, key=r.get)
                val[sel] = R[ck[sel], np.arange(len(sel))]
            order = np.lexsort((-val, J))          # per station: largest ratio, first in point order on ties
            first = np.r_[True, J[order][1:] != J[order][:-1]]
            pick = {int(J[i]): int(i) for i in order[first]}
            for jj in dict.fromkeys(J.tolist()):
                i = pick[jj]
                j, N, V, M, lc = pts[mid][i]
                worst[jj] = dict(line=m["line_name"], member=mid, j=j, D_mm=round(ln["stations"][j]["D"] * 1000),
                                 check=names[ck[i]], ratio=float(val[i]), lc=lc, N=N, V=V, M=M, stiff_k=k,
                                 tension_field=tf)
            vmax = float(np.max(np.abs(Pt[:, 2])))
        if tf:                                     # tension-field shear is not monotone in N, M: all combinations
            vtf = min(shear_capacity(caps[j], N, M) for j, N, V, M, lc in pts_all[mid])
        trials[key] = t = dict(member=mid, k=k, tf=tf, c=c, sd=sd, caps=caps, worst=worst, vmax=vmax, vtf=vtf,
                               vd_a=min(cp["Vd"] for cp in caps.values()),
                               mx=max(w["ratio"] for w in worst.values()))
        return t

    def select(m, allow_tf=True, post_cost=0):
        """Cheapest passing (k, method); post_cost = 1 when tension field here needs an end post."""
        ks = [stiff.get(m["id"], 0)] if stiff is not None else ([0, *K_STIFF] if st_on else [0])
        cands = [(k, False, k) for k in ks]
        cands += [(k + post_cost, True, k) for k in ks if k and tf_on and allow_tf]
        best = None
        for _, tf, k in sorted(cands):
            t = trial(m, k, tf)
            if t is None:
                continue
            if best is None or t["mx"] < best["mx"] - 1e-9:
                best = t
            if t["mx"] <= target:
                return t
        return best

    ilines = [ln for ln in mdl.lines if not mdl.mem_index[ln["members"][0]].get("sec")]    # welded I members
    lines = {ln["name"]: ln["members"] for ln in ilines}               # sub-members in station order
    def tf_allowed(ln, i):                                            # a support end needs an end post
        n, kn = len(ln["members"]), knee.get(ln["name"], ())
        at_sup = (i == 0) or (i == n - 1)
        if stiff is not None and at_sup and ln["members"][i] not in (posts or ()):
            return False                                               # verify: only where the design has a post
        return (i > 0 or "start" in kn) and (i < n - 1 or "end" in kn)

    choice = {}
    for ln in ilines:
        for i, mid in enumerate(ln["members"]):
            n = len(ln["members"])
            choice[mid] = select(mdl.mem_index[mid], allow_tf=tf_allowed(ln, i),
                                 post_cost=int(i == 0 or i == n - 1))

    def anchor_forces(P, mP, step):
        """8.5.3: R_tf, M_tf of tension-field member P at its end on side step (-1 start / +1 end)."""
        cp = P["caps"][mP["j"] if step < 0 else mP["j"] + 1]
        d, tw, f = cp["d"], cp["tw"], cp["f"]
        Vp, Vcr = d * tw * f / math.sqrt(3), d * tw * cp["tau_b"]
        Hq = 1.25 * Vp * math.sqrt(max(0.0, 1 - Vcr / Vp))
        Vcr_d = Vcr / GM0
        if P["vmax"] < P["vtf"] and P["vtf"] > Vcr_d:                 # 8.5.3 reduction when V < Vtf
            Hq *= min(1.0, max(0.0, (P["vmax"] - Vcr_d) / (P["vtf"] - Vcr_d)))
        return Hq / 2, Hq * d / 10, cp

    def anchor(name, mids, e, step):
        """Anchorage of tension-field member mids[e] on side step: 8.5.1 check of the next panel, or at a
        knee / haunch an 8.5.2(b) end post. None = not anchorable (ridge / base / no end post possible)."""
        P, mP = choice[mids[e]], mdl.mem_index[mids[e]]
        Rtf, Mtf, cp = anchor_forces(P, mP, step)
        nb = e + step
        if not 0 <= nb < len(mids):                                    # member end is a support
            if ("start" if step < 0 else "end") not in knee.get(name, ()):
                return None
            pl = mP["plates"]
            ep = end_post(cp["d"], cp["tw"], min(pl["bf_in"], pl["bf_out"]), fy, P["c"], Rtf, Mtf)
            if ep is None:
                return None
            u = mP["j"] + (ep["e"] / mP["L"] if step < 0 else 1 - ep["e"] / mP["L"])
            return dict(line=name, member=mids[e], side=step, Rtf=Rtf, Mtf=Mtf, cQ=ep["e"], ratio=ep["ratio"],
                        far=None, F_end=0.0, end_post=ep, ep_pos=(name, round(u, 4)),
                        sup_pos=(name, round(mP["j"] + (0 if step < 0 else 1), 4)))
        mQ = mdl.mem_index[mids[nb]]
        if choice[mids[nb]]["k"]:                                      # Q = first panel of the stiffened neighbour
            kq = choice[mids[nb]]["k"]
            cQ, twQ = mQ["L"] / kq, mQ["plates"]["tw"]
            far = (name, round(mQ["j"] + (1 - 1 / kq if step < 0 else 1 / kq), 4))
        else:                                                          # Q = whole unstiffened stretch
            q, cQ, twQ = nb, 0.0, math.inf
            while 0 <= q < len(mids) and choice[mids[q]]["k"] == 0:
                cQ, twQ = cQ + mdl.mem_index[mids[q]]["L"], min(twQ, mdl.mem_index[mids[q]]["plates"]["tw"])
                q += step
            far = None if not 0 <= q < len(mids) else (name, round(mdl.mem_index[mids[q]]["j"] + (1 if step < 0 else 0), 4))
        f = cp["f"]
        ratio = max(Rtf / (cQ * twQ * f / (math.sqrt(3) * GM0)), Mtf / (twQ * cQ ** 2 / 4 * f / GM0))
        return dict(line=name, member=mids[e], side=step, Rtf=Rtf, Mtf=Mtf, cQ=cQ, ratio=ratio, far=far,
                    F_end=Mtf / cQ)

    anchors, changed = [], True
    while changed:
        changed, anchors = False, []
        for name, mids in lines.items():
            j = 0
            while j < len(mids) and not changed:
                if not choice[mids[j]]["tf"]:
                    j += 1
                    continue
                a = j
                while j + 1 < len(mids) and choice[mids[j + 1]]["tf"]:
                    j += 1
                for e, step in ((a, -1), (j, +1)):
                    info = anchor(name, mids, e, step)
                    if info is None or info["ratio"] > 1.0:
                        choice[mids[e]] = select(mdl.mem_index[mids[e]], allow_tf=False)
                        changed = True
                        break
                    anchors.append(info)
                j += 1
            if changed:
                break

    rows = [w for t in choice.values() for w in t["worst"].values()] + _truss_rows(mdl, inp, pts, Ls, pts_all)
    # stiffener axial forces: 8.7.2.5 in tension-field panels, 8.5.1 end stiffener of each anchoring panel
    force = {}
    for t in choice.values():
        if t["tf"]:
            m = mdl.mem_index[t["member"]]
            for i in range(t["k"] + 1):
                key = (m["line_name"], round(m["j"] + i / t["k"], 4))
                force[key] = max(force.get(key, 0.0), max(0.0, t["vmax"] - t["vd_a"]))
    for an in anchors:
        if an["far"] is not None:
            force[an["far"]] = max(force.get(an["far"], 0.0), an["F_end"])
        if an.get("end_post"):                                         # end plate takes the other flange force
            force[an["sup_pos"]] = max(force.get(an["sup_pos"], 0.0), an["end_post"]["F"])
    # purlin loads (8.7.3.1 web buckling, 8.7.4 web bearing, 8.7.8 pull): a purlin on a stiffener loads the
    # stiffener (load-carrying, 8.7.5); elsewhere the bare web is checked and, with the stiffener option, a
    # load-carrying stiffener is added at a purlin whose web fails
    pu = inp.get("purlin", {})
    b1, keff = float(pu.get("b1_mm", 20.0)) / 1000, float(pu.get("web_KL", 1.0))
    skeys = {(mdl.mem_index[t["member"]]["line_name"], round(mdl.mem_index[t["member"]]["j"] + i / t["k"], 4))
             for t in choice.values() if t["k"] for i in range(t["k"] + 1)}
    skeys |= {an["ep_pos"] for an in anchors if an.get("end_post")}
    carry, extra, purlin_rows, pmax = {}, {}, [], [0.0, 0.0]
    preac = purlin_reactions(mdl)
    if preac:                              # purlin load of every ULS combination at once: Q = C (combos x cases) P
        pkeys, plcs = list(preac), sorted({lc for d in preac.values() for lc in d})
        li = {lc: i for i, lc in enumerate(plcs)}
        Cm = np.zeros((len(mdl.uls), len(plcs)))
        for a, c in enumerate(mdl.uls):
            for lc, f in c["pairs"]:
                if lc in li:
                    Cm[a, li[lc]] += f
        Q = Cm @ np.array([[preac[k].get(lc, 0.0) for k in pkeys] for lc in plcs])
        a_in, a_out = np.argmax(-Q, axis=0), np.argmax(Q, axis=0)     # first combination reaching the extreme
    for n_k, (key, prim) in enumerate(preac.items()):
        Pin, Pout = max(0.0, -Q[a_in[n_k], n_k]), max(0.0, Q[a_out[n_k], n_k])
        lin = mdl.uls[a_in[n_k]]["lc"] if Pin > 0 else 0
        lout = mdl.uls[a_out[n_k]]["lc"] if Pout > 0 else 0
        pmax[0], pmax[1] = max(pmax[0], Pin), max(pmax[1], Pout)
        if key in skeys:
            force[key] = max(force.get(key, 0.0), Pin)                  # 8.7.5.1 with 8.7.2.5: max(Fq, Fx)
            carry[key] = max(Pin, Pout)
            continue
        ln = next(l for l in mdl.lines if l["name"] == key[0])
        j, D = int(round(key[1])), ln["stations"][int(round(key[1]))]["D"]
        worst = None
        for jj in (j - 1, j):                                           # the two sub-members at the purlin
            mm = mdl.mem_index[ln["members"][jj]]
            p = mm["plates"]
            Fw, Fcdw = web_point_load(D, D - p["tf_in"] - p["tf_out"], p["tw"], p["tf_out"], fy, b1, keff)
            r = max((max(Pin, Pout) / Fw, "web bearing 8.7.4", lin if Pin >= Pout else lout),
                    (Pin / Fcdw, "web buckling 8.7.3.1", lin))
            if worst is None or r[0] > worst[0][0]:
                worst = (r, mm)
        (ratio, chk, lc), mm = worst
        if ratio > target and st_on:
            extra[key] = mm["id"]
            force[key] = max(force.get(key, 0.0), Pin)
            carry[key] = max(Pin, Pout)
        else:
            purlin_rows.append(dict(line=key[0], member=mm["id"], j=j, D_mm=round(D * 1000), check=chk, ratio=ratio,
                                    lc=lc, N=Pin if Pin >= Pout else -Pout, V=0.0, M=0.0, stiff_k=0, tension_field=False))
    rows += purlin_rows
    pos = {}
    for t in choice.values():
        if not t["k"]:
            continue
        m, k = mdl.mem_index[t["member"]], t["k"]
        P = Plates(**m["plates"])
        st = mdl.lines[m["line"]]["stations"]
        a_, b_ = st[m["j"]], st[m["j"] + 1]
        d_web = max(m["D_a"], m["D_b"]) - P.tf_in - P.tf_out
        for i in range(k + 1):
            u = i / k
            key = (m["line_name"], round(m["j"] + u, 4))
            F = force.get(key, 0.0)
            sd = stiffener(d_web, P.tw, min(P.bf_in, P.bf_out), fy, t["c"], F)
            if sd is None:                                             # even the thickest plate buckles
                bs = min((P.bf_in - P.tw) / 2, 20 * 0.016 * math.sqrt(250 / fy))
                ratio = F / stiffener_fqd(d_web, P.tw, 0.016, bs, fy)
                rows.append(dict(line=m["line_name"], member=m["id"], j=m["j"], D_mm=round(d_web * 1000),
                                 check="stiffener 8.7.2.5", ratio=ratio, lc=0, N=F, V=0.0, M=0.0, stiff_k=k,
                                 tension_field=t["tf"]))
                sd = stiffener(d_web, P.tw, min(P.bf_in, P.bf_out), fy, t["c"]) or t["sd"]
            if key not in pos or sd["mass"] > pos[key]["mass"]:
                pos[key] = dict(sd, F=F, line=m["line_name"], member=m["id"], s=a_["s"] + u * (b_["s"] - a_["s"]),
                                x=a_["x"] + u * (b_["x"] - a_["x"]), y=a_["y"] + u * (b_["y"] - a_["y"]),
                                D=a_["D"] + u * (b_["D"] - a_["D"]), c=t["c"])
    for key, mid in extra.items():                                     # load-carrying stiffeners at purlins
        m = mdl.mem_index[mid]
        P = Plates(**m["plates"])
        st = mdl.lines[m["line"]]["stations"][int(round(key[1]))]
        d_web = st["D"] - P.tf_in - P.tf_out
        sd = stiffener(d_web, P.tw, min(P.bf_in, P.bf_out), fy, m["L"], force[key])
        if sd is None:
            rows.append(dict(line=key[0], member=mid, j=st["j"], D_mm=round(st["D"] * 1000), check="stiffener 8.7.5.1",
                             ratio=99.0, lc=0, N=force[key], V=0.0, M=0.0, stiff_k=0, tension_field=False))
            continue
        pos[key] = dict(sd, line=key[0], member=mid, s=st["s"], x=st["x"], y=st["y"], D=st["D"], c=m["L"],
                        purlin=True)
    for an in anchors:                                                 # end-post stiffeners (8.5.2(b))
        if an.get("end_post"):
            ep, m = an["end_post"], mdl.mem_index[an["member"]]
            st = mdl.lines[m["line"]]["stations"]
            a_, b_ = st[m["j"]], st[m["j"] + 1]
            u = an["ep_pos"][1] - m["j"]
            pos[an["ep_pos"]] = dict(ep, line=m["line_name"], member=m["id"], s=a_["s"] + u * (b_["s"] - a_["s"]),
                                     x=a_["x"] + u * (b_["x"] - a_["x"]), y=a_["y"] + u * (b_["y"] - a_["y"]),
                                     D=a_["D"] + u * (b_["D"] - a_["D"]), c=ep["e"], end_post=True)
    for key, Fx in carry.items():                                      # 8.7.5.2 bearing of load-carrying stiffeners
        if key in pos:
            it = pos[key]
            ratio = Fx / (it["ts"] * it["bs"] * fy * 1e3 / (0.8 * GM0))
            it["Fx"] = Fx
            rows.append(dict(line=key[0], member=it["member"], j=int(round(key[1])), D_mm=round(it["D"] * 1000),
                             check="stiffener bearing 8.7.5.2", ratio=ratio, lc=0, N=Fx, V=0.0, M=0.0,
                             stiff_k=0, tension_field=False))
    tfm = sorted(t["member"] for t in choice.values() if t["tf"])
    stiffeners = dict(layout={mid: t["k"] for mid, t in choice.items()}, tension_field=tfm,
                      end_posts=sorted(an["member"] for an in anchors if an.get("end_post")),
                      items=sorted(pos.values(), key=lambda v: (v["line"], v["s"])), anchors=anchors,
                      weld_tf=max((tf_weld(cp) for mid in tfm for cp in choice[mid]["caps"].values() if cp["tf"]),
                                  default=0.0),
                      n=len(pos), mass=sum(v["mass"] for v in pos.values()),
                      purlin=dict(b1=b1, web_KL=keff, n_extra=len(extra), n_carry=len(carry),
                                  web_max=max((x["ratio"] for x in purlin_rows), default=0.0),
                                  P_push=pmax[0], P_pull=pmax[1]))
    # SLS
    X, lim_v, lim_h = mdl.col_x(), inp["limits"]["rafter_defl"], inp["limits"]["eave_drift"]
    lim_c, lim_s = inp["limits"].get("crane_drift", 200.0), inp["limits"].get("rail_spread_mm", 10.0)
    tops = [mdl.col_top(c) for c in range(len(X))]
    sls = {}
    for c in mdl.sls:
        lcs, U = {lc for lc, _ in c["pairs"]}, disp(c["lc"])
        if lcs <= set(mdl.lc_ll):
            for ln in mdl.lines:
                if ln["type"] != "raf":
                    continue
                sp = ln["span"]
                xa, xb, ua, ub = X[sp], X[sp + 1], U[tops[sp]][1], U[tops[sp + 1]][1]
                for st in ln["stations"]:
                    rel = U[st["node"]][1] - (ua + (ub - ua) * (st["x"] - xa) / (xb - xa))
                    r = abs(rel) / ((xb - xa) / lim_v)
                    k = f"span {sp + 1}"
                    if r > sls.get(k, {"ratio": -1})["ratio"]:
                        sls[k] = dict(line=k, member=0, j=st["j"], D_mm=0, check=f"deflection L/{lim_v:g}",
                                      ratio=r, lc=c["lc"], N=0.0, V=0.0, M=1000 * rel)
        elif lcs <= set(mdl.lc_wl):
            for ci, top in enumerate(tops):
                r = abs(U[top][0]) / (mdl.nodes[top][1] / lim_h)
                k = f"C{ci}"
                if r > sls.get(k, {"ratio": -1})["ratio"]:
                    sls[k] = dict(line=k, member=0, j=0, D_mm=0, check=f"drift H/{lim_h:g}",
                                  ratio=r, lc=c["lc"], N=0.0, V=0.0, M=1000 * U[top][0])
        elif lcs & set(mdl.lc_cl):                                     # Table 6 crane + wind, lateral
            for ci in mdl.crane_info:
                uL, uR = U[ci["tips"][0]][0], U[ci["tips"][1]][0]
                for k, u, r, chk in ((f"K{ci['crane'] + 1}L", uL, abs(uL) / (ci["rail"] / lim_c), f"crane drift H/{lim_c:g}"),
                                     (f"K{ci['crane'] + 1}R", uR, abs(uR) / (ci["rail"] / lim_c), f"crane drift H/{lim_c:g}"),
                                     (f"K{ci['crane'] + 1}gauge", uR - uL, abs(uR - uL) / (lim_s / 1000),
                                      f"rail spread {lim_s:g}mm")):
                    if r > sls.get(k, {"ratio": -1})["ratio"]:
                        sls[k] = dict(line=k, member=0, j=0, D_mm=0, check=chk, ratio=r, lc=c["lc"], N=0.0, V=0.0,
                                      M=1000 * u)
    rows += list(sls.values())
    tmax = float(inp["limits"].get("taper_max", 150.0))
    if six_on(inp) and tmax > 0:                                       # g_taper: gradual six-section tapers
        for ln in mdl.lines:
            if ln["type"] != "raf" or len(ln["knots"]) != 7:
                continue
            kn, skip = ln["knots"], (0 if ln["kind"] == "S-R" else 5)  # the haunch segment itself is exempt
            for i in range(6):
                a, b = kn[i], kn[i + 1]
                if i != skip:
                    sl = abs(b["D"] - a["D"]) * 1000 / (b["s"] - a["s"])
                    rows.append(dict(line=ln["name"], member=0, j=a["j"], D_mm=round(1000 * max(a["D"], b["D"])),
                                     check=f"taper {tmax:g}mm/m", ratio=sl / tmax, lc=0, N=0.0, V=0.0, M=sl))
    return dict(rows=rows, max_ratio=max(r["ratio"] for r in rows),
                viol=sum(max(0.0, r["ratio"] - target) for r in rows), stiffeners=stiffeners)


def _truss_rows(mdl: FrameModel, inp: dict, pts: dict, Ls: dict, pts_all: dict | None = None) -> list:
    """Truss members at the frame's steel grade: IS 800 station ratios (is800 tube_ / angle_ratios), Table 3
    slenderness - 180 if compressed in a combination without wind (i), 250 if compressed only with wind (iii),
    400 if never compressed (vi) - and for tubes the joint rules b_web / b_chord in [0.35, 1] (web welded onto
    the chord face; CIDECT RHS-joint validity range, joint resistance not checked).
    Compression = N > max(1 kN, 2 % of the member's max |N|)."""
    fy, fu = float(inp["fy"]), float(inp.get("fu", 490.0))
    wind = {c["lc"] for c in mdl.uls if any(lc in mdl.lc_wl for lc, _ in c["pairs"])}
    rows = []
    for m in mdl.members:
        if not m.get("sec"):
            continue
        mid, H, da = m["id"], round(m["sec"]["H"] * 1000), m["sec"]["shape"] == "DA"
        cp = (angle_capacities if da else tube_capacities)(m["sec"], fy, fu, Ls[mid][3], Ls[mid][2])
        rat = angle_ratios_v if da else tube_ratios_v
        allp = (pts_all or pts)[mid]                 # Table 3 needs every combination's sign of N
        worst = {}
        if len(pts[mid]) <= VEC_MIN * 2:             # few combinations (screened search): scalar checks
            rat_s = angle_ratios if da else tube_ratios
            for j, N, V, M, lc in pts[mid]:
                r = rat_s(cp, N, V, M)
                chk = max(r, key=r.get)
                if j not in worst or r[chk] > worst[j]["ratio"]:
                    worst[j] = dict(line=m["line_name"], member=mid, j=j, D_mm=H, check=chk, ratio=r[chk], lc=lc,
                                    N=N, V=V, M=M, stiff_k=0, tension_field=False)
        else:                                        # every station and combination at once (numpy)
            Pt = np.array(pts[mid], dtype=float)
            r = rat(cp, Pt[:, 1], Pt[:, 2], Pt[:, 3])
            names = list(r)
            R = np.stack([r[k] for k in names])      # (checks, points); argmax = first maximum, as max(r, key=r.get)
            ck = np.argmax(R, axis=0)
            val = R[ck, np.arange(R.shape[1])]
            J = Pt[:, 0].astype(int)
            order = np.lexsort((-val, J))            # per station: the largest ratio, first in point order on ties
            first = np.r_[True, J[order][1:] != J[order][:-1]]
            pick = {int(J[i]): int(i) for i in order[first]}
            for jj in dict.fromkeys(J.tolist()):     # stations in order of first appearance
                i = pick[jj]
                j, N, V, M, lc = pts[mid][i]
                worst[jj] = dict(line=m["line_name"], member=mid, j=j, D_mm=H, check=names[ck[i]], ratio=float(val[i]),
                                 lc=lc, N=N, V=V, M=M, stiff_k=0, tension_field=False)
        Na = np.array([q[1] for q in allp], dtype=float)
        tol = max(1.0, 0.02 * float(np.max(np.abs(Na)))) if len(Na) else 1.0
        comp = {q[4] for q, n in zip(allp, Na > tol) if n}
        cw, cg = bool(comp & wind), bool(comp - wind)
        lim = 180 if cg else 250 if cw else 400
        rows += list(worst.values()) + [dict(line=m["line_name"], member=mid, j=m["j"], D_mm=H, check=f"KL/r <= {lim}",
                                             ratio=cp["KLr"] / lim, lc=0, N=0.0, V=0.0, M=cp["KLr"], stiff_k=0,
                                             tension_field=False)]
    tr = mdl.fi.truss
    for sd, S in (tr.sections.items() if tr and truss_family(inp) == "tube" else ()):
        b0 = min(S["top"]["B"], S["bot"]["B"])
        for role in ("vert", "diag"):
            beta = S[role]["B"] / b0
            for chk, ratio in (("joint b1/b0 <= 1", beta), ("joint b1/b0 >= 0.35", 0.35 / beta)):
                rows.append(dict(line=f"{sd} {role}", member=0, j=0, D_mm=round(S[role]["H"] * 1000), check=chk,
                                 ratio=ratio, lc=0, N=0.0, V=0.0, M=beta, stiff_k=0, tension_field=False))
    return rows


def _g_local(e: dict) -> np.ndarray:
    """Local geometric stiffness per unit compression (softening), [fx1 fy1 m1 fx2 fy2 m2]."""
    L, g = e["m"]["L"], np.zeros((6, 6))
    if not e["r"]:
        g[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = np.array(
            [[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L],
             [-36, -3 * L, 36, -3 * L], [3 * L, -L * L, -3 * L, 4 * L * L]]) / (30 * L)
    else:                                       # hinged end: string stiffness only
        g[np.ix_([1, 4], [1, 4])] = np.array([[1, -1], [-1, 1]]) / L
    return g


def _fail(msg: str) -> dict:
    return dict(ok=False, weight=1e9, cost=1e9, max_ratio=99.0, viol=1e6, rows=[], error=msg)


def evaluate(inp: dict, design: dict, only=None, buckling: bool = True) -> dict:
    """P-Delta analysis of every ULS combo + IS 800 checks. Never raises for a bad design (ratio 99).
    only: ULS combination numbers to analyse (screening in the search; the result is then 'screened' and
    every design the search accepts is re-checked with all combinations).
    buckling: elastic lambda_cr of the gravity combinations (reported, not a design check - the P-Delta
    analysis itself rejects an unstable frame); the search skips it, the final design gets it."""
    try:
        mdl = FrameModel(frame_input(inp, design))
    except ValueError as ex:
        return _fail(str(ex))
    uls = mdl.uls if only is None else [c for c in mdl.uls if c["lc"] in only]
    fe = LinearFrame2D(mdl, shear=True)                                # web shear deformation, as STAAD
    mids = [m["id"] for m in mdl.members]
    free, nf = fe.free, len(fe.free)
    Kl = np.array([e["k"] for e in fe.el])                              # (nm, 6, 6) local, condensed
    Gl = np.array([_g_local(e) for e in fe.el])
    Tm = np.array([e["T"] for e in fe.el])
    idx = np.array([e["idx"] for e in fe.el])
    pos = {d: i for i, d in enumerate(free)}
    gv, ga, gb = [], [], []                                             # global Kg per unit N, element blocks
    for i, e in enumerate(fe.el):
        gg = Tm[i].T @ Gl[i] @ Tm[i]
        keep = [a for a in range(6) if idx[i, a] in pos]
        gv.append([gg[a, b] for a in keep for b in keep])
        ga.append([pos[idx[i, a]] for a in keep for b in keep])
        gb.append([pos[idx[i, b]] for a in keep for b in keep])
    g_el = np.repeat(np.arange(len(fe.el)), [len(v) for v in gv])
    gv, ga, gb = np.concatenate(gv), np.concatenate(ga).astype(np.intp), np.concatenate(gb).astype(np.intp)
    # Banded Cholesky in reverse Cuthill-McKee order (the element blocks give the pattern of K and Kg):
    # K factorised once for every primary load case; K - Kg(N) once per P-Delta iteration.
    B = BandedSPD(nf, ga, gb)
    kp = np.unique(ga * nf + gb)
    kr, kc = kp // nf, kp % nf
    Kv = fe.K[free[kr], free[kc]]
    Kband = B.band(Kv, B.scatter(kr, kc))
    flatG, size = B.scatter(ga, gb), Kband.size
    keepG = flatG >= 0

    def kband(N):                                                       # band of K - Kg(N)
        return Kband - np.bincount(flatG[keepG], weights=(N[g_el] * gv)[keepG], minlength=size).reshape(Kband.shape)
    try:
        fK = B.factor(Kband)
    except LinAlgError:
        return _fail("singular stiffness (mechanism) - check releases / connectivity")
    fe.solver = lambda b: B.solve(fK, b)
    need = sorted({lc for c in mdl.uls + mdl.sls for lc, _ in c["pairs"]})
    sol = {lc: fe.solve({lc: 1.0}) for lc in need}
    U = {lc: np.array([sol[lc]["u"][nd] for nd in fe.ids]).ravel() for lc in need}      # dof order = ids

    def local(u):                                                       # element local displacements
        return np.einsum("eij,ej->ei", Tm, u[idx])

    def to_ends(f):                                                     # (nm, 6) -> STAAD (nm, 2, 6)
        out = np.zeros((len(f), 2, 6))
        out[:, :, [0, 1, 5]] = f.reshape(-1, 2, 3)
        return out

    uls_prim = sorted({lc for c in uls for lc, _ in c["pairs"]})
    Ff = {lc: B.matvec_band(Kband, U[lc][free]) for lc in uls_prim}
    FEF = {}
    for lc in uls_prim:
        f = np.array([sol[lc]["ends"][mid] for mid in mids])[:, :, [0, 1, 5]].reshape(-1, 6)
        FEF[lc] = f - np.einsum("eij,ej->ei", Kl, local(U[lc]))
    E2 = {}
    for c in uls:                                                       # P-Delta, N iterated
        F = sum(f * Ff[lc] for lc, f in c["pairs"])
        fef = sum(f * FEF[lc] for lc, f in c["pairs"])
        N = np.zeros(len(mids))                                         # first pass = first order
        for it in range(10):
            try:
                cf = B.factor(kband(N)) if it else fK
            except LinAlgError:
                return _fail(f"unstable under {c['title']} (lambda_cr < 1)")
            u = np.zeros(fe.ndof)
            u[free] = B.solve(cf, F)
            f = np.einsum("eij,ej->ei", Kl - N[:, None, None] * Gl, local(u)) + fef
            if np.max(np.abs(f[:, 0] - N)) < 1e-3 * max(1.0, np.max(np.abs(N))):
                break
            N = f[:, 0]
        E2[c["lc"]] = to_ends(f)
    lam = {}
    Ksp = Kinv = None
    for c in uls:                                                       # K phi = lambda Kg phi
        if c["gravity_only"] and only is None and buckling:             # reported only, not a design check
            N = E2[c["lc"]][:, 0, 0]
            G = csr_matrix((N[g_el] * gv, (ga, gb)), shape=(nf, nf))
            if Ksp is None:
                Ksp = csr_matrix((Kv, (kr, kc)), shape=(nf, nf))
                Kinv = LinearOperator((nf, nf), matvec=fe.solver, dtype=float)
            try:                                                        # largest mu of Kg phi = mu K phi (Lanczos)
                mu = eigsh(G, k=1, M=Ksp, Minv=Kinv, which="LA", tol=1e-12, ncv=min(nf - 1, 24),
                           return_eigenvectors=False)[0]
            except (ArpackNoConvergence, ArpackError):
                Kff = fe.K[np.ix_(free, free)]
                mu = eigh(G.toarray(), Kff, eigvals_only=True, subset_by_index=[nf - 1, nf - 1])[0]
            lam[c["lc"]] = 1 / mu if mu > 1e-12 else math.inf
    cmap = {c["lc"]: c for c in mdl.sls}

    def ends(lc):
        return {mid: (E2[lc][i, 0], E2[lc][i, 1]) for i, mid in enumerate(mids)}

    def disp(lc):
        arr = sum(f * U[l] for l, f in cmap[lc]["pairs"]).reshape(-1, 3)
        return {nd: arr[i, :2] for i, nd in enumerate(fe.ids)}

    try:
        res = check_design(mdl, inp, ends, disp, uls=uls)
    except ValueError as ex:
        return _fail(str(ex))
    w = mdl.steel_weight()
    sm = res["stiffeners"]
    weight = sum(w.values()) + sm["mass"]
    pen = float(inp.get("options", {}).get("stiffener_penalty_kg", 0.0))
    res.update(ok=res["viol"] <= 0, weight=weight, cost=weight + pen * sm["n"], line_mass=w, lam_cr=lam,
               lam_cr_gravity=min(lam.values()) if lam else math.inf, model=mdl, ends=E2, mids=mids,
               screened=only is not None)
    return res


# ============================================================================= search
def piece_template(pc: dict):
    """(template, mirrored) used by a roof piece - same rule as FrameModel._default_profile."""
    return {"S-R": ("raf_int" if pc["int_a"] else "raf_ext", False),
            "R-S": ("raf_int" if pc["int_b"] else "raf_ext", True),
            "S-S": ("raf_ss", False)}[pc["kind"]]


def rafter_joints(inp: dict) -> list:
    """[((tpl, knot), (tpl, knot)), ...] for every node where two roof pieces meet (ridge, valley,
    single-ridge column); knot 0 = template start (support end), -1 = template end (ridge end)."""
    pcs = FrameModel(frame_input(inp, inp["templates"])).pieces
    out = []
    for p, q in zip(pcs, pcs[1:]):
        (tp, mp), (tq, mq) = piece_template(p), piece_template(q)
        out.append(((tp, 0 if mp else -1), (tq, -1 if mq else 0)))
    return out


SIX = ("raf_ext", "raf_int")
ZONE = 0.2          # low-moment zone: |M| <= M_min + ZONE (M_max - M_min)
SIX_DEFAULT = (0, 0.1, 0.2, 0.35, 0.55, 0.75, 1)     # knots when no envelope is available (unstable seed)


def six_on(inp: dict) -> bool:
    return bool(inp.get("options", {}).get("rafter_six_sections", True))


def _six(inp: dict, d: dict) -> list:
    """Rafter templates of d that are six-section (7 knots) with the option on."""
    return [t for t in SIX if six_on(inp) and t in d and len(d[t]["D"]) == 7]


def _shape(inp: dict, d: dict) -> dict:
    """g_shape / g_apex of the module docstring by projection (in place). A raised knot pushes its neighbours
    toward the haunch / ridge up with it, so every single-variable step of the search stays effective:
    D6 <= min D0, D3 = D4 = min(D3, D0, D6), D2 in [D3, D0], D1 in [D2, D0], D5 in [line D4-D6 (catalogue up), D6]."""
    six = _six(inp, d)
    if not six:
        return d
    top = min(d[t]["D"][0] for t in six)
    for t in six:
        D, s = d[t]["D"], d[t]["knots"]
        D[6] = min(D[6], top)
        m = min(D[3], D[4], D[0], D[6])
        D[3] = D[4] = m
        D[2] = min(max(D[2], m), D[0])
        D[1] = min(max(D[1], D[2]), D[0])
        line = m + (D[6] - m) * (s[5] - s[4]) / (s[6] - s[4])
        D[5] = min(max(D[5], next(v for v in CAT["D"] if v >= line - 1e-9)), D[6])
    return d


def envelope(r: dict) -> dict:
    """{template: [(s, |M|max), ...]} ULS moment envelope per rafter piece, in template direction support ->
    ridge (mirrored pieces reversed), from an evaluate() result."""
    mdl, out = r["model"], {}
    idx = {mid: i for i, mid in enumerate(r["mids"])}
    for pc in mdl.pieces:
        tpl, mir = piece_template(pc)
        ln = mdl.line_by_name(f"R{pc['idx']}")
        env = np.zeros(ln["n_sub"] + 1)
        for mid in ln["members"]:
            m, i = mdl.mem_index[mid], idx[mid]
            for E in r["ends"].values():
                for j, _, _, M in end_forces_to_stations(m, E[i, 0], E[i, 1]):
                    env[j] = max(env[j], abs(M))
        sf = np.array([st["s_frac"] for st in ln["stations"]])
        out.setdefault(tpl, []).append((1 - sf[::-1], env[::-1]) if mir else (sf, env))
    return out


def _spread(k: list) -> list:
    """Strictly increasing station indices with fixed ends (needs k[-1] >= len(k) - 1)."""
    k = list(k)
    for i in range(1, len(k) - 1):
        k[i] = max(k[i], k[i - 1] + 1)
    for i in range(len(k) - 2, 0, -1):
        k[i] = min(k[i], k[i + 1] - 1)
    return k


def six_knots(env: list, n: int) -> list:
    """Seven knot stations 0..n from the envelope [(s, M), ...] of one template: k1 / k2 where |M| has dropped
    1/3 and 2/3 of the way from the haunch to the low-moment zone k3..k4, k5 where it has risen halfway from the
    zone to the ridge. Each section at least one station long."""
    g = np.linspace(0, 1, n + 1)
    M = np.max([np.interp(g, s, e) for s, e in env], axis=0)
    jz = int(np.argmin(M))
    thr = M[jz] + ZONE * (max(M[0], M[n]) - M[jz])
    a = b = jz
    while a > 0 and M[a - 1] <= thr:
        a -= 1
    while b < n and M[b + 1] <= thr:
        b += 1
    drop = M[0] - M[a]
    k1 = next(j for j in range(a + 1) if M[j] <= M[0] - drop / 3)
    k2 = next(j for j in range(a + 1) if M[j] <= M[0] - 2 * drop / 3)
    k5 = next(j for j in range(b, n + 1) if M[j] >= (M[b] + M[n]) / 2)
    return _spread([0, k1, k2, a, b, k5, n])


def to_six(inp: dict, design: dict, r: dict | None = None, log=print) -> dict:
    """design with raf_ext / raf_int re-knotted to six sections at the ULS |M| envelope of design (r = its
    evaluate() result, computed if missing). Depths interpolated, each new segment takes the plates of the old
    segment at its middle; decode() then applies the shape rules. Pieces shorter than 6 stations keep their knots."""
    tp = [t for t in SIX if t in design]
    if not six_on(inp) or not tp:
        return design
    r = r if r and "model" in r else evaluate(inp, design)
    env = envelope(r) if "model" in r else {}
    mdl = r.get("model") or FrameModel(frame_input(inp, design))
    new = copy.deepcopy(design)
    for t in tp:
        n = min(mdl.line_by_name(f"R{pc['idx']}")["n_sub"] for pc in mdl.pieces if piece_template(pc)[0] == t)
        if n < 6:
            log(f"  {t}: only {n} roof sub-members - six sections need 6; knots kept")
            continue
        k = six_knots(env[t], n) if t in env else _spread([round(f * n) for f in SIX_DEFAULT])
        sn = [round(j / n, 6) for j in k]
        old = design[t]
        segs = [max(i for i in range(len(old["plates"])) if old["knots"][i] <= (a + b) / 2) for a, b in zip(sn, sn[1:])]
        new[t] = dict(knots=sn, D=[CAT["D"][_nearest(CAT["D"], float(np.interp(x, old["knots"], old["D"])))] for x in sn],
                      plates=[list(old["plates"][i]) for i in segs])
    return new


def variables(inp: dict, tpls: list, T: dict | None = None) -> list:
    """Design variables as groups of linked slots (tpl, key, i) sharing one catalogue value - the equality
    constraints h_joint, h_bf, h_zone and h_pl of the module docstring eliminated by linking (union-find).
    T: templates defining the shape (knot / segment counts), default inp['templates']."""
    T = T or inp["templates"]
    CAT["angle"] = [d["name"] for d in angle_chain(float(inp["fy"]), gusset(inp))]   # class depends on the grade
    slots = []
    for t in tpls:
        if t == "truss":                        # h0, bottom chord slope, web type, 4 sections per side (family)
            slots += [("truss", key, 0) for key in ("h0", "bs", "web")]
            slots += [("truss", f"{sd}@{truss_family(inp)}", i) for sd in truss_sides(inp) for i in range(len(ROLES))]
        else:
            slots += [(t, "D", i) for i in range(len(T[t]["D"]))]
            slots += [(t, key, s_) for s_ in range(len(T[t]["plates"])) for key in ("tw", "bf", "tf")]
    root = {x: x for x in slots}

    def find(x):
        while root[x] != x:
            x = root[x]
        return x

    def link(x, y):
        if x in root and y in root:
            root[find(x)] = find(y)

    if inp.get("options", {}).get("constant_bf", True):                   # h_bf
        for t in [t for t in tpls if t != "truss"]:
            for sg in range(1, len(T[t]["plates"])):
                link((t, "bf", sg), (t, "bf", 0))
    for t in _six(inp, {t: T[t] for t in tpls}):                          # h_zone, h_pl
        link((t, "D", 4), (t, "D", 3))
        for a, b in ((1, 0), (3, 2), (5, 4)):
            for key in ("tw", "bf", "tf"):
                link((t, key, a), (t, key, b))
    for (ta, ka), (tb, kb) in rafter_joints(inp):                         # h_joint
        if ta not in tpls or tb not in tpls:
            continue
        ia, ib = ka % len(T[ta]["D"]), kb % len(T[tb]["D"])
        link((ta, "D", ia), (tb, "D", ib))
        sa, sb = min(ia, len(T[ta]["plates"]) - 1), min(ib, len(T[tb]["plates"]) - 1)
        for key in ("tw", "bf", "tf"):
            link((ta, key, sa), (tb, key, sb))
    groups = {}
    for x in slots:
        groups.setdefault(find(x), []).append(x)
    return list(groups.values())


def vlabel(g: list) -> str:
    t, key, i = g[0]
    return f"{t} {key}{i}" + (f" (+{len(g) - 1} linked)" if len(g) > 1 else "")


def _slot(design: dict, x) -> float:
    t, key, i = x
    if t == "truss":
        return design["truss"][key.split("@")[0]][i]
    return design[t]["D"][i] if key == "D" else design[t]["plates"][i][("tw", "bf", "tf").index(key)]


def link_mismatches(inp: dict, design: dict) -> list:
    """Linked slots with different values in a design (e.g. an old file): violated equalities."""
    return [g for g in variables(inp, [t for t in design if t in inp["templates"]], design)
            if len({_slot(design, x) for x in g}) > 1]


def compatible(inp: dict, design: dict) -> dict:
    """Design snapped to the catalogue with every equality satisfied (linked slots take their largest value)
    and the six-section shape rules applied."""
    tpls = active_templates(inp)
    T = {t: design.get(t, inp["templates"][t]) for t in tpls}
    var = variables(inp, tpls, T)
    return decode(inp, var, encode(inp, T, var), T)


def _catof(g: list) -> list:
    """Catalogue of a variable group (truss section keys carry it: 'ext@angle')."""
    key = g[0][1]
    return CAT[key.split("@")[1] if "@" in key else key]


def _nearest(cat, val):
    if isinstance(val, str):                              # section name; another family -> nearest kg/m
        return cat.index(val) if val in cat else min(range(len(cat)), key=lambda i: abs(_name_w(cat[i]) - _name_w(val)))
    return min(range(len(cat)), key=lambda i: abs(cat[i] - val))


def encode(inp: dict, design: dict, var: list) -> list:
    """Catalogue index per variable group; a group whose slots differ takes the largest (conservative)."""
    return [_nearest(_catof(g), max(_slot(design, x) for x in g)) for g in var]


def decode(inp: dict, var: list, x: list, T: dict | None = None) -> dict:
    T = T or inp["templates"]
    d = {t: copy.deepcopy(T[t]) for t in {sl[0] for g in var for sl in g}}
    for g, k in zip(var, x):
        for t, key, i in g:
            if t == "truss":
                d["truss"][key.split("@")[0]][i] = CAT[key.split("@")[1] if "@" in key else key][k]
            elif key == "D":
                d[t]["D"][i] = CAT["D"][k]
            else:
                d[t]["plates"][i][("tw", "bf", "tf").index(key)] = CAT[key][k]
    return _shape(inp, d)


def _eval_stripped(args):
    """Worker: evaluate without the (unpicklable-heavy) model object."""
    inp, design, only = args
    r = evaluate(inp, design, only, buckling=False)
    for k in ("model", "ends", "mids"):
        r.pop(k, None)
    return r


def optimise(inp: dict, start: dict | None = None, log=print, stop=lambda: False, shake_passes=1,
             workers: int | None = None) -> dict:
    """Discrete greedy minimisation of cost = steel weight (+ stiffener penalty). Returns dict(design, result, evals, seconds).
    Six-section rafters: knots from the start design's moment envelope, search, knots re-derived from the result's
    envelope and, if they moved, one more search from there."""
    t0 = time.time()
    limit = 60.0 * float(inp.get("limits", {}).get("time_limit_min", 0) or 0)
    user_stop, said = stop, []

    def stop():                                    # user Stop, or the time limit: keep the best feasible design so far
        if limit and time.time() - t0 > limit and not said:
            said.append(1)
            log(f"time limit {limit / 60:g} min reached - finishing with the best feasible design found")
        return user_stop() or bool(limit and time.time() - t0 > limit)
    tpls = active_templates(inp)
    start = {t: copy.deepcopy((start or {}).get(t, inp["templates"][t])) for t in tpls}
    six = six_on(inp) and any(t in SIX for t in tpls)
    if six:
        start = to_six(inp, start, log=log)
        _log_knots(inp, start, log)
    out = _search(inp, start, log, stop, shake_passes, workers)
    if six and not stop():
        d1 = to_six(inp, out["design"], r=out["result"], log=log)
        if any(d1[t]["knots"] != out["design"][t]["knots"] for t in SIX if t in d1):
            log("six-section knots moved with the optimised moment envelope -> search once more")
            _log_knots(inp, d1, log)
            evals = out["evals"]
            out = _search(inp, d1, log, stop, shake_passes, workers)
            out["evals"] += evals
    out["seconds"] = time.time() - t0
    return out


def _log_knots(inp, d, log):
    for t in _six(inp, d):
        log(f"  {t}: six sections, knots s/L " + " ".join(f"{s:.3f}" for s in d[t]["knots"]))


def _search(inp: dict, start: dict, log, stop, shake_passes, workers) -> dict:
    """Repair / trim / shake over the catalogue indices; neighbour scans in parallel processes (workers=1 ->
    in-process). start defines the template shapes (knots)."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    t0 = time.time()
    tpls = list(start)
    var = variables(inp, tpls, start)

    def canon(y):                                          # projected (shape rules) catalogue indices
        return encode(inp, decode(inp, var, list(y), start), var)
    x = canon(encode(inp, start, var))
    memo = {}                                              # full checks (all ULS combinations)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    # Combination screening: neighbour scans analyse only the ULS combinations that govern some station
    # (plus the gravity-only ones); a design is accepted only after a full check with every combination,
    # whose governing combinations join the screen. Crane frames have hundreds of combinations, few govern.
    mdl0 = FrameModel(frame_input(inp, decode(inp, var, list(x), start)))
    uls_all = {c["lc"] for c in mdl0.uls}
    screen = {"on": len(uls_all) > 40 and inp.get("options", {}).get("screen_combos", True),
              "active": {c["lc"] for c in mdl0.uls if c["gravity_only"]}, "ver": 0}
    smemo = {}

    def learn(r):
        new = {row["lc"] for row in r.get("rows", ()) if row.get("lc") in uls_all} - screen["active"]
        if new:
            screen["active"] |= new
            screen["ver"] += 1

    def fmany(xs, full=False):
        nonlocal pool
        scr = screen["on"] and not full
        key = (lambda y: (tuple(y), screen["ver"])) if scr else (lambda y: tuple(y))
        store = smemo if scr else memo

        def have(y):
            return tuple(y) in memo or key(y) in store
        todo = list(dict.fromkeys(tuple(y) for y in xs if not have(y)))
        only = frozenset(screen["active"]) if scr else None
        args = [(inp, decode(inp, var, list(k), start), only) for k in todo]
        try:
            res = list(pool.map(_eval_stripped, args)) if pool else list(map(_eval_stripped, args))
        except BrokenProcessPool:                     # e.g. caller has no importable __main__
            log("  worker processes failed - continuing in-process")
            pool = None
            res = list(map(_eval_stripped, args))
        for k, rr in zip(todo, res):
            store[key(k)] = rr
            if not scr:
                learn(rr)
        return [memo.get(tuple(y)) or store[key(y)] for y in xs]

    def f(x):
        return fmany([x], full=True)[0]

    def step(x, i, d):
        return canon(x[:i] + [x[i] + d] + x[i + 1:])

    def ahead(x, i, d):
        """Up to `workers` successive steps of variable i in direction d (catalogue bounds respected)."""
        out, y = [], x
        while len(out) < (workers if pool else 1) and 0 <= y[i] + d < len(_catof(var[i])):
            y = step(y, i, d)
            out.append(y)
        return out

    def repair(x):
        r = f(x)
        while r["viol"] > 0 and not stop():
            cand = [i for i in range(len(x)) if x[i] + 1 < len(_catof(var[i]))]
            best, best_score = None, 0.0
            for i, ry in zip(cand, fmany([step(x, i, 1) for i in cand])):
                if ry["viol"] < r["viol"]:
                    score = (r["viol"] - ry["viol"]) / max(ry["cost"] - r["cost"], 0.5)
                    if score > best_score:
                        best, best_score = i, score
            if best is None:
                log("  repair stuck: no single step reduces the violation (catalogue limit?)")
                return x, r
            x, r = step(x, best, 1), f(step(x, best, 1))
            while r["viol"] > 0:                                          # line search on the same variable:
                ys = ahead(x, best, 1)                                    # next steps evaluated in parallel,
                n_ok = 0                                                  # accepted in order as before
                for y, ry in zip(ys, fmany(ys)):
                    if ry["viol"] >= r["viol"] or (r["viol"] - ry["viol"]) / max(ry["cost"] - r["cost"], 0.5) < best_score / 2:
                        break
                    x, r, n_ok = y, ry, n_ok + 1
                if r.get("screened"):
                    r = f(x)                                              # full check of the accepted step
                if not ys or n_ok < len(ys):
                    break
            log(f"  repair  {r['weight']:8.1f} kg  max D/C {r['max_ratio']:.3f}  violation {r['viol']:.3f}")
        return x, r

    def trim(x):
        r = f(x)
        while not stop():
            cand = [i for i in range(len(x)) if x[i] > 0]
            ok = [(r["cost"] - ry["cost"], i) for i, ry in zip(cand, fmany([step(x, i, -1) for i in cand]))
                  if ry["ok"] and ry["cost"] < r["cost"]]
            i = next((i for _, i in sorted(ok, reverse=True)       # largest saving that passes every combination
                      if not screen["on"] or f(step(x, i, -1))["ok"]), None)
            if i is None:
                break
            x, r = step(x, i, -1), f(step(x, i, -1))
            while True:                                                   # line search downwards (parallel)
                ys = ahead(x, i, -1)
                n_ok, last = 0, (x, r)
                for y, ry in zip(ys, fmany(ys)):
                    if not (ry["ok"] and ry["cost"] < r["cost"]):
                        break
                    x, r, n_ok = y, ry, n_ok + 1
                if r.get("screened"):                                     # confirm with every combination;
                    rf = f(x)                                             # if it fails, step back one at a time
                    while not rf["ok"] and tuple(x) != tuple(last[0]):
                        x = step(x, i, 1)
                        rf, n_ok = f(x), 0
                    r = rf
                if not ys or n_ok < len(ys):
                    break
            log(f"  trim    {r['weight']:8.1f} kg  max D/C {r['max_ratio']:.3f}  ({vlabel(var[i])})")
        return x, r

    try:
        log(f"variables: {len(var)} (after linking {sum(len(g) for g in var)} slots) over {', '.join(tpls)}; "
            f"{workers} worker process(es)")
        x, r = repair(x)
        x, r = trim(x)
        for _ in range(shake_passes):                                     # step down + repair, keep if lighter
            improved = False
            for i in range(len(x)):
                if x[i] == 0 or not r["ok"] or stop():
                    continue
                y, ry = repair(step(x, i, -1))
                if ry["ok"] and ry["cost"] < r["cost"] - 0.5:
                    x, r = trim(y)
                    improved = True
                    log(f"  shake   {r['weight']:8.1f} kg  ({vlabel(var[i])} down)")
            if not improved:
                break
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    design = decode(inp, var, x, start)
    r = evaluate(inp, design)                                             # with model, for drawing/export
    scr = f" + {len(smemo)} screened ({len(screen['active'])} of {len(uls_all)} ULS combinations)" if screen["on"] else ""
    log(f"done: {r['weight']:.1f} kg ({r['stiffeners']['n']} stiffeners), max D/C {r['max_ratio']:.3f}, {len(memo)} analyses"
        f"{scr}, {time.time() - t0:.0f} s")
    return dict(design=design, result=r, evals=len(memo), screened=len(smemo), seconds=time.time() - t0)


# ============================================================================= STAAD
def engine() -> Path:
    """STAAD's headless engine: env PEB_STAAD_EXE, the known paths, then any installed STAAD.Pro (newest first)."""
    env = os.environ.get("PEB_STAAD_EXE")
    found = sorted(Path(os.environ.get("ProgramFiles", r"C:\Program Files"), "Bentley").glob(
        "**/STAAD/SProStaad/SProStaad.exe"), reverse=True)
    for p in ([Path(env)] if env else []) + ENGINES + found:
        if p.exists():
            return p
    raise FileNotFoundError("SProStaad.exe not found - install STAAD.Pro or set the environment variable PEB_STAAD_EXE "
                            "to the full path of SProStaad.exe")


def parse_forces(text: str) -> dict:
    """{(member, lc): [start[6], end[6]]} from PRINT MEMBER FORCES (same layout as Bridge Stage1)."""
    forces, section, member, key = {}, None, None, None
    for ln in text.splitlines():
        if "MEMBER END FORCES" in ln:
            section = "F"
            continue
        if "SUPPORT REACTIONS" in ln or "JOINT DISPLACEMENT" in ln:
            section = None
        try:
            nums = [float(t) for t in ln.split()]
        except ValueError:
            continue
        if section == "F" and len(nums) in (7, 8, 9):
            if len(nums) == 9:
                member = int(nums[0])
            if len(nums) >= 8:
                key = (member, int(nums[-8]))
                forces[key] = []
            forces[key].append(nums[-6:])
    return forces


def parse_displacements(text: str) -> dict:
    """{(joint, lc): (ux, uy) m} from PRINT JOINT DISPLACEMENTS (cm output)."""
    out, jt, on = {}, None, False
    for ln in text.splitlines():
        if "JOINT DISPLACEMENT (CM" in ln:
            on = True
            continue
        if on and ("END OF LATEST" in ln or re.match(r"\s*\d+\.\s+[A-Z]", ln)):
            on = False
        if not on:
            continue
        try:
            nums = [float(t) for t in ln.split()]
        except ValueError:
            continue
        if len(nums) == 8:
            jt = int(nums[0])
        if len(nums) in (7, 8):
            out[jt, int(nums[-7])] = (nums[-6] / 100, nums[-5] / 100)
    return out


def parse_code_check(text: str) -> dict:
    """{member: dict(status, ratio, load, condition, slender)} from IS800 CHECK CODE output."""
    out = {}
    pat = re.compile(r"\s+(\d+).*?Status:\s+(\w+)\s+Ratio:\s+([\d.]+|Infinity)\s+Critical Load Case:\s+(\d+)"
                     r".*?Critical Condition:\s+(.+?)\s*\|", re.S)
    for block in text.split("Member Number:")[1:]:
        m = pat.match(block)
        if m:
            out[int(m.group(1))] = dict(status=m.group(2), ratio=float(m.group(3)), load=int(m.group(4)),
                                        condition=m.group(5).strip(),
                                        slender=bool(re.search(r"Section Class:\s*Slender", block)))
    return out


def _run(std: Path) -> str:
    anl = std.with_suffix(".ANL")
    anl.unlink(missing_ok=True)
    subprocess.run([str(engine()), std.name], cwd=std.parent, timeout=1800, check=False)  # engine splits at spaces
    if not anl.exists():
        raise RuntimeError(f"STAAD produced no {anl.name}; open {std.name} in STAAD.Pro to see why")
    text = anl.read_text(errors="ignore")
    errs = [ln.strip() for ln in text.splitlines() if re.search(r"\*+\s*ERROR", ln)]
    if errs:
        raise RuntimeError(f"{std.name}: " + " | ".join(errs[:5]))
    return text


def _cmd(head: str, ids: list) -> list:
    """STAAD command + member / load list, wrapped with '-' continuation (72-char lines)."""
    toks, lines, cur = _ranges(ids), [], head
    for t in toks:
        if len(cur) + len(t) + 3 > 72:
            lines.append(cur + " -")
            cur = ""
        cur = (cur + " " + t).strip()
    return lines + [cur]


def _sls_block(mdl: FrameModel, inp: dict, prints: bool) -> list:
    """SLS output for the IS 800 Table 6 checks the optimiser makes: the limits as comments and the joint
    displacements of the column tops, rafter stations and crane rails under every SLS combination.
    STAAD 2026's IS800 LSD code check ignores DFF / DJ1 / DJ2 on TAPERED members (tried: DFF 20000 changed
    nothing), so no deflection code check is written."""
    lim = inp["limits"]
    L = ["* SLS limits (IS 800 Table 6) checked by the optimiser on these displacements:",
         f"*   rafter deflection relative to the eaves <= span/{lim['rafter_defl']:g} (live-load combinations)",
         f"*   eave drift <= H/{lim['eave_drift']:g} (wind-only combinations)"]
    if mdl.crane_info:
        L += [f"*   crane rail drift <= H_rail/{lim.get('crane_drift', 200):g} and rail gauge change <= "
              f"{lim.get('rail_spread_mm', 10):g} mm (CL, 0.8(CL+W))"]
    nodes = {mdl.col_top(c) for c in range(len(mdl.fi.spans) + 1)}
    nodes |= {st["node"] for ln in mdl.lines if ln["type"] == "raf" for st in ln["stations"]}
    nodes |= {n for ci in mdl.crane_info for n in ci["tips"]}
    return (L + _cmd("LOAD LIST", [c["lc"] for c in mdl.sls])
            + (["PRINT JOINT DISPLACEMENTS ALL"] if prints else _cmd("PRINT JOINT DISPLACEMENTS LIST", sorted(nodes))))


def export_std(inp: dict, design: dict, folder, r0: dict | None = None, prints: bool = False, log=print) -> dict:
    """STAAD files of a design WITH its design parameters:
      <name>_ULS_PDELTA.std  P-Delta + PARAMETER block: CODE IS800 LSD, FYLD, FU, STP 2 (welded), LZ / LY / LX from
                             the optimiser's restraint layout, TST 1 + TSP (web stiffener spacing), TRACK 2, CHECK CODE
      <name>_SLS.std         linear; Table 6 limits + joint displacements at the check nodes (see _sls_block)
      <name>_BUCKLING.std    buckling analysis of the gravity combinations
      station_map.json, <name>_design.json
    prints=True adds PRINT MEMBER FORCES / JOINT DISPLACEMENTS (for staad_verify)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mdl = FrameModel(frame_input(inp, design))
    r0 = r0 if r0 and "ends" in r0 else evaluate(inp, design)
    if r0.get("error"):
        raise ValueError(f"design cannot be analysed: {r0['error']}")
    name, paths = inp["name"], {}
    for kind, tag in (("ULS", "ULS_PDELTA"), ("SLS", "SLS"), ("BUCKLING", "BUCKLING")):
        p = folder / f"{name}_{tag}.std"
        mdl.write_std(str(p), kind)
        body = p.read_text().rstrip().splitlines()[:-1]                     # drop FINISH
        if kind == "ULS":
            body += (_cmd("LOAD LIST", [c["lc"] for c in mdl.uls]) + (["PRINT MEMBER FORCES ALL"] if prints else [])
                     + _design_block(mdl, inp, r0))
        elif kind == "SLS":
            body += _sls_block(mdl, inp, prints)
        p.write_text("\n".join(body + ["FINISH"]) + "\n")
        paths[kind] = p
    (folder / "station_map.json").write_text(json.dumps(mdl.station_map(), indent=1, default=float))
    save_json(folder / f"{name}_design.json", dict(input=inp, design=design))
    log(f"STAAD files with design parameters written to {folder}")
    return paths


def _design_block(mdl: FrameModel, inp: dict, r: dict) -> list:
    """STAAD IS800 LSD code check with the optimiser's restraint lengths. STAAD takes ONE LTB length (LX)
    per member, so LX follows the flange compressed by the member's larger ULS moment (fast engine):
    inner flange -> fly-brace spacing, outer flange -> purlin/girt spacing.
    ponytail: in moment-reversal members STAAD then checks the smaller-sign moment with the other flange's
    length (can be un- or over-conservative); our station checks use the correct length for each sign."""
    Ls = restraint_lengths(mdl, int(inp["restraint"]["fly_every"]), *_truss_lengths(inp))
    sign = {m["id"]: m["sign_ic"] for m in mdl.members}
    lx = {}
    for i, mid in enumerate(r["mids"]):
        m_ic = [sign[mid] * v for E in r["ends"].values() for v in (-E[i, 0, 5], E[i, 1, 5])]
        lx[mid] = Ls[mid][0] if max(m_ic) >= -min(m_ic) else Ls[mid][1]
    L = ["PARAMETER 1", "CODE IS800 LSD", f"FYLD {inp['fy'] * 1000:g} ALL",
         f"FU {inp.get('fu', 490.0) * 1000:g} ALL", "STP 2 ALL"]
    if inp["limits"]["dc_target"] != 1.0:
        L.append(f"RATIO {inp['limits']['dc_target']:g} ALL")
    tubes = sorted(m["id"] for m in mdl.members if m.get("sec") and m["sec"]["shape"] != "DA")
    if tubes:                                  # STAAD has no cold-formed choice: it uses curve a for tubes (ours: b)
        L += _cmd("STP 1 MEMB", tubes)
    for name, vals in (("LZ", {k: v[3] for k, v in Ls.items()}), ("LY", {k: v[2] for k, v in Ls.items()}),
                       ("LX", lx)):
        groups = {}
        for mid, v in vals.items():
            groups.setdefault(round(v, 3), []).append(mid)
        for val, mids in sorted(groups.items()):
            L += [f"{name} {val:g} MEMB {x}" for x in _ranges(mids)]
    lay = r.get("stiffeners", {}).get("layout", {})
    stiff = {}
    for mid, k in lay.items():
        if k:
            stiff.setdefault(round(mdl.mem_index[mid]["L"] / k, 3), []).append(mid)
    if stiff:                                              # IS800 LSD: TST 1 = transverse stiffeners provided,
        L += [f"TST 1 MEMB {x}" for x in _ranges(sorted(m for ms in stiff.values() for m in ms))]   # TSP = spacing
        for val, mids in sorted(stiff.items()):
            L += [f"TSP {val:g} MEMB {x}" for x in _ranges(mids)]
    chk = [m["id"] for m in mdl.members if not (m.get("sec") and m["sec"]["shape"] == "DA")]  # PRISMATIC: ours only
    return L + ["TRACK 2 ALL"] + (["CHECK CODE ALL"] if len(chk) == len(mdl.members) else _cmd("CHECK CODE MEMB", chk))


def staad_verify(inp: dict, design: dict, folder: Path, log=print) -> dict:
    """Run ULS P-Delta + SLS in STAAD, re-check with STAAD forces, and read STAAD's own IS800 check."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mdl = FrameModel(frame_input(inp, design))
    r0 = evaluate(inp, design)                     # fast engine: LX choice, stiffener layout, force cross-check
    uls_lcs = [c["lc"] for c in mdl.uls]
    paths = export_std(inp, design, folder, r0=r0, prints=True, log=log)
    log(f"STAAD: running {paths['ULS'].name} (P-Delta + IS800 check) ...")
    uls = _run(paths["ULS"])
    log(f"STAAD: running {paths['SLS'].name} ...")
    sls = _run(paths["SLS"])
    F, U = parse_forces(uls), parse_displacements(sls)
    missing = [k for k in ((m["id"], lc) for m in mdl.members for lc in uls_lcs) if k not in F]
    if missing:
        raise RuntimeError(f"{len(missing)} member/load results missing from the .ANL, e.g. {missing[0]}")

    def ends(lc):
        return {m["id"]: (F[m["id"], lc][0], F[m["id"], lc][1]) for m in mdl.members}

    def disp(lc):
        return {nd: U.get((nd, lc), (0.0, 0.0)) for nd in mdl.nodes}
    layout = r0["stiffeners"]["layout"]
    res = check_design(mdl, inp, ends, disp, stiff=layout,          # same stiffeners + end posts, STAAD forces
                       posts=r0["stiffeners"].get("end_posts", []))
    code = parse_code_check(uls)
    # STAAD's IS800 check is not a valid 2nd opinion where it (a) designs slender sections on gross properties
    # or (b) the panel relies on tension field 8.4.2.2(b). Stiffened (a) panels are judged: TST 1 + TSP c give
    # STAAD's shear capacity with the stiffeners (147.0 vs our 148.1 kN on the demo, 79.6 kN unstiffened).
    sw = slender_web(inp)
    stiffened = sorted(r0["stiffeners"].get("tension_field", []))
    judged = {m: v for m, v in code.items() if not (sw and v["slender"]) and m not in stiffened}
    # P-Delta cross-check: STAAD warns "divergent" on some near-zero-sway cases; compare its moments with ours
    diff = {}
    for i, mid in enumerate(r0["mids"]):
        for lc in uls_lcs:
            for e in (0, 1):
                diff.setdefault(lc, []).append((abs(r0["ends"][lc][i, e, 5] - F[mid, lc][e][5]), abs(F[mid, lc][e][5])))
    force_diff = max(max(a for a, _ in v) / max(max(b for _, b in v), 1e-9) for v in diff.values())
    res.update(staad_code=code, staad_slender=[m for m, v in code.items() if v["slender"]], staad_stiffened=stiffened,
               staad_fail=[m for m, v in judged.items() if v["status"] != "PASS" or v["slender"]],
               staad_max=max((v["ratio"] for v in judged.values()), default=float("nan")), files=paths, model=mdl,
               weight=r0["weight"], stiffeners=r0["stiffeners"], force_diff=force_diff,
               pdelta_warnings=sorted({int(x) for x in re.findall(r"Load case\s+(\d+) is divergent", uls)}))
    res["ok"] = res["viol"] <= 0
    return res


def staad_max_text(v: dict) -> str:
    return (f"{v['staad_max']:.3f}" if v["staad_max"] == v["staad_max"]
            else "n/a (every member is slender or stiffened - STAAD's check not valid there)")


def optimise_verified(inp: dict, folder: Path, log=print, stop=lambda: False, rounds=3, start=None) -> dict:
    """Search with the fast engine, verify in STAAD, tighten and repeat until STAAD forces pass too."""
    work = copy.deepcopy(inp)
    target = inp["limits"]["dc_target"]
    out = None
    for k in range(rounds):
        log(f"=== round {k + 1}: internal D/C target {work['limits']['dc_target']:.3f}")
        out = optimise(work, start=start, log=log, stop=stop)
        if stop():
            break
        Path(folder).mkdir(parents=True, exist_ok=True)                  # keep the search if STAAD/PC dies
        save_json(Path(folder) / f"{inp['name']}_design_round{k + 1}.json", dict(input=inp, design=out["design"]))
        v = staad_verify(inp, out["design"], folder, log=log)
        out["verify"] = v
        log(f"STAAD forces: max D/C {v['max_ratio']:.3f} (target {target}); STAAD IS800 max ratio "
            f"{staad_max_text(v)}, {len(v['staad_fail'])} member(s) FAIL"
            + (f", {len(v['staad_slender'])} slender (checked by our 8.2.1.1(a) only)" if slender_web(inp) else "/slender"))
        if v["ok"]:
            break
        work["limits"]["dc_target"] *= target / v["max_ratio"] * 0.995
        start = out["design"]
    return out


# ============================================================================= report / export
def report(inp: dict, out: dict) -> str:
    r, d = out["result"], out["design"]
    area = sum(inp["spans"]) * inp["bay"]
    L = [f"PEB GABLE OPTIMISER - IS 800:2007 LSM    {inp['name']}",
         "=" * 78,
         f"Spans {inp['spans']} m, eave {inp['eave_height']} m, slope 1:{1 / inp['slope']:g}, bay {inp['bay']} m, "
         f"fy {inp['fy']:g} MPa",
         f"Steel (frame members + web stiffeners) {r['weight']:.0f} kg = {r['weight'] / area:.2f} kg/m2 plan"
         + (f"  [stiffeners {r['stiffeners']['n']} no., {r['stiffeners']['mass']:.0f} kg]" if r["stiffeners"]["n"] else ""),
         f"lambda_cr (gravity combos, elastic eigenvalue) {r['lam_cr_gravity']:.2f}; "
         f"member forces from P-Delta analysis", ""]
    if not inp["wind"]["cpe_verified"]:
        L += ["!!! Wind Cpe are placeholders - replace with IS 875-3 tables before issue !!!", ""]
    L.append("SECTIONS  (D at knots; plates tw x bf x tf per segment, mm)")
    for t, v in d.items():
        L.append(f"  {TEMPLATE_NAMES[t]}")
        if t == "truss":
            L.append(f"    {'double angles IS 808 on ' + format(1000 * gusset(inp), 'g') + ' mm gussets' if truss_family(inp) == 'angle' else 'SHS/RHS IS 4923'}"
                     f", depth at the eave columns h0 {v['h0'][0]} mm, bottom chord rises {v['bs'][0]:g} x the roof, "
                     f"{v['web'][0]} web (verticals at every purlin + ridge)")
            L += [f"    {sd} half: top {v[sd][0]} | bottom {v[sd][1]} | verticals {v[sd][2]} | diagonals {v[sd][3]}"
                  for sd in truss_sides(inp)]
            continue
        L.append("    knots s/L " + "  ".join(f"{s:g}" for s in v["knots"]) + "   D " + " / ".join(map(str, v["D"])))
        L.append("    plates    " + " | ".join(f"{a}x{b}x{c}" for a, b, c in v["plates"]))
        if t in _six(inp, d):
            L.append("    six sections from the ULS |M| envelope: D0>=D1>=D2>=D3=D4<=D5<=D6<=D0, D5 >= line D4-D6 "
                     "(no apex spike); plates change only at knots 2 and 4")
    if slender_web(inp):
        L.append(f"Slender-web option ON (IS 800 8.2.1.1(a)); unstiffened web d/tw limit min(200 eps_w, 345 eps_f^2) = "
                 f"{min(200 * math.sqrt(250 / inp['fy']), 345 * 250 / inp['fy']):.0f}"
                 + (", with stiffeners up to 270 eps_w (8.6.1.1(b))" if stiffeners_on(inp) else ""))
    L += ["", "CONSTRAINTS (equalities eliminated by linking design variables)"]
    for (ta, ka), (tb, kb) in dict.fromkeys(tuple(sorted(j)) for j in rafter_joints(inp)):
        if ta in d and tb in d and (ta, ka) != (tb, kb):
            ia, ib = ka % len(d[ta]["D"]), kb % len(d[tb]["D"])
            sa, sb = min(ia, len(d[ta]["plates"]) - 1), min(ib, len(d[tb]["plates"]) - 1)
            L.append(f"  rafter joint {ta}[{ia}] = {tb}[{ib}]: D {d[ta]['D'][ia]} / {d[tb]['D'][ib]} mm, "
                     f"plates {'x'.join(map(str, d[ta]['plates'][sa]))} / {'x'.join(map(str, d[tb]['plates'][sb]))}")
    L.append(f"  constant flange width per member: {'ON' if inp.get('options', {}).get('constant_bf', True) else 'off'}")
    bad = link_mismatches(inp, d)
    if bad:
        L.append("  !!! design violates linked equalities: " + ", ".join(vlabel(g) for g in bad))
    L += ["", "LINE MASSES (members)"] + [f"  {k:<4} {m:8.1f} kg" for k, m in r["line_mass"].items()]
    mdl, cr = r.get("model"), inp.get("crane", {})
    if mdl is not None and mdl.crane_info:
        L += ["", "CRANE LOADS (IS 875-2 6.3 / 6.4; IS 800 Table 4 crane combinations, Table 6 limits)",
              f"  SWL {cr['capacity_kN']:g} kN, bridge {cr['bridge_kN']:g} kN, crab {cr['crab_kN']:g} kN, a_min "
              f"{cr['a_min']:g} m, wheel base {cr['wheel_base']:g} m x {cr['wheels_per_rail']} wheels/rail, impact "
              f"{cr['impact']:g}, surge {cr['surge']:g}; bracket at {cr['bracket_level']:g} m, e = {cr['e']:g} m, rail "
              f"{cr['gantry_depth']:g} m above the seat"]
        for ci in mdl.crane_info:
            R = ci["R"]
            L.append(f"  crane {ci['crane'] + 1} (span {ci['span'] + 1}, gauge {R['Lc']:.2f} m): wheel Pmax {R['Pmax']:.1f} / "
                     f"Pmin {R['Pmin']:.1f} kN static; per bracket (gantry influence {R['il']:.2f}): Rmax {R['Rmax']:.1f} kN "
                     f"(with impact), Rmin {R['Rmin']:.1f} kN, surge {R['H']:.2f} kN at rail level, gantry DL {R['G']:.1f} kN")
        L.append(f"  load cases {', '.join(map(str, mdl.lc_cl))}; {len(mdl.crane_variants())} CL arrangements; "
                 f"{sum(1 for c in mdl.uls if any(lc in mdl.lc_cl for lc, _ in c['pairs']))} ULS crane combinations")
        L.append("  not in this 2D frame: tractive force 5% of static wheel loads along the rails (6.3(d)) -> "
                 "longitudinal bracing; bearing stiffener under the gantry seat at the bracket tip and column web "
                 "stiffeners opposite the bracket flanges (connection design); gantry girder design")
    if stiffeners_on(inp):
        L += ["", "TRANSVERSE WEB STIFFENERS (single-sided flat, outstand to flange tip; 8.4.2.2(a), 8.6.1(b), 8.7.2)"]
        items = r["stiffeners"]["items"]
        if not items:
            L.append("  none needed")
        by_line = {}
        for it in items:
            by_line.setdefault(it["line"], []).append(it)
        for ln, its in by_line.items():
            L.append(f"  {ln}: {len(its)} no. at s = " + ", ".join(f"{it['s']:.2f}" for it in its) + " m")
            sizes = sorted({(round(it["ts"] * 1e3), round(it["bs"] * 1e3)) for it in its})
            L.append("      plates " + ", ".join(f"{b} x {t}" for t, b in sizes) + " mm (bs x ts); spacing c = "
                     + ", ".join(sorted({f"{it['c']:.2f}" for it in its})) + " m; weld shear >= "
                     f"{max(it['weld_kN_per_mm'] for it in its):.3f} kN/mm (8.7.2.6)")
        tfm = r["stiffeners"].get("tension_field", [])
        if tension_field_on(inp):
            L.append(f"  Tension field 8.4.2.2(b) in members: {tfm or 'none'}")
            if tfm:
                an = r["stiffeners"]["anchors"]
                L.append(f"  anchorage 8.5.1/8.5.3: {len(an)} chain ends, max ratio "
                         f"{max((a['ratio'] for a in an), default=0):.3f}; R_tf <= "
                         f"{max((a['Rtf'] for a in an), default=0):.1f} kN, M_tf <= {max((a['Mtf'] for a in an), default=0):.1f} kNm")
                L.append(f"  web-flange welds in tension-field panels: transfer fv tw = {r['stiffeners']['weld_tf']:.0f} kN/m (8.6.3.3)")
                for x in an:
                    if x.get("end_post"):
                        ep = x["end_post"]
                        L.append(f"  END POST 8.5.2(b) {x['line']} {'start' if x['side'] < 0 else 'end'} (m{x['member']}): "
                                 f"stiffener {ep['bs'] * 1e3:.0f} x {ep['ts'] * 1e3:.0f} at e = {ep['e']:.2f} m from the end plate; "
                                 f"R_tf {ep['Rtf']:.0f} <= {ep['Vd']:.0f} kN, M_tf {ep['Mtf']:.1f} <= {ep['Md']:.1f} kNm, "
                                 f"strut {ep['F']:.0f} <= {ep['Fqd']:.0f} kN (ratio {ep['ratio']:.2f})")
                if any(x.get("end_post") for x in an):
                    L.append("  end posts: the member-end plate / cap plate is the outer stiffener - its bearing and bolt "
                             "design is connection design (not included); 8.5.2(a) single-stiffener form not used")
                at_sup = [x for x in an if x["far"] is None and not x.get("end_post")]
                if at_sup:
                    L.append("  anchor end force carried by the support connection (end plate / cap plate): "
                             + ", ".join(f"{x['line']} m{x['member']} {x['F_end']:.1f} kN" for x in at_sup))
                L.append("  w_tf taken as the lower of IS 800's printed '+' form and the Porter-Rockey-Evans '-' form")
            Fmax = max((it["F"] for it in r["stiffeners"]["items"]), default=0.0)
            over = sum(it["F"] > it["Fqd"] + 1e-9 for it in r["stiffeners"]["items"])
            L.append(f"  stiffener axial force (8.7.2.5 Fq, 8.5.1 end stiffener): max {Fmax:.1f} kN, "
                     + ("all <= Fqd" if not over else f"{over} EXCEED Fqd (see 'stiffener 8.7.2.5' rows)"))
        else:
            L.append("  8.7.2.5: Fq = V - Vcr/gm0 <= 0 wherever the shear check passes (no tension field).")
    pu = r["stiffeners"].get("purlin")
    if pu and inp.get("scheme") != "truss":
        L += ["", f"PURLIN LOADS INTO THE RAFTER WEB (8.7.4 bearing, 8.7.3.1 web buckling, 8.7.8 pull; b1 = "
                  f"{pu['b1'] * 1e3:.0f} mm, web strut KL = {pu['web_KL']:g} d)",
              f"  bare-web purlins: max ratio {pu['web_max']:.3f}; factored purlin load up to {pu['P_push']:.1f} kN push, "
              f"{pu['P_pull']:.1f} kN pull",
              f"  purlins on stiffeners (load-carrying, 8.7.5): {pu['n_carry']}"
              + (f", of which {pu['n_extra']} stiffeners were added only for the purlin load" if pu["n_extra"] else "")]
    L += ["", "GOVERNING CHECKS (worst 15)", f"  {'line':<7}{'mem':>6}{'st':>4}{'D':>6}  {'check':<16}{'D/C':>6}"
          f"{'LC':>6}{'N kN':>9}{'V kN':>8}{'M kNm':>9}"]
    for x in sorted(r["rows"], key=lambda x: -x["ratio"])[:15]:
        L.append(f"  {x['line']:<7}{x['member']:>6}{x['j']:>4}{x['D_mm']:>6}  {x['check']:<16}{x['ratio']:6.3f}"
                 f"{x['lc']:>6}{x['N']:9.1f}{x['V']:8.1f}{x['M']:9.1f}")
    v = out.get("verify")
    if v:
        L += ["", f"STAAD VERIFICATION ({v['files']['ULS'].name}, P-Delta)",
              f"  our IS 800 checks on STAAD forces: max D/C {v['max_ratio']:.3f} -> {'PASS' if v['ok'] else 'FAIL'}",
              f"  STAAD IS800 LSD CHECK CODE: max ratio {staad_max_text(v)}, "
              f"FAIL members: {v['staad_fail'] or 'none'}"]
        if slender_web(inp):
            L.append(f"  slender-web members (STAAD uses gross properties; our 8.2.1.1(a) check governs): "
                     f"{v['staad_slender'] or 'none'}")
        if v.get("staad_stiffened"):
            L.append(f"  tension-field members (STAAD judged on TST/TSP only; our 8.4.2.2(b) check governs): "
                     f"{v['staad_stiffened']}")
        L.append(f"  P-Delta cross-check: max |M_ours - M_STAAD| = {100 * v['force_diff']:.2f} % of max |M|"
                 + (f"; STAAD 'divergent' warnings on LC {v['pdelta_warnings']}" if v["pdelta_warnings"] else ""))
    L += ["", "Notes: SLS = LL-only rafter deflection and wind-only drift. " +
          ("Slender webs: IS 800 8.2.1.1(a) flanges-only + 8.6.1 d/tw limits; slender flanges rejected."
           if slender_web(inp) else "Class 4 sections rejected.")
          + (" Web stiffeners where needed (see schedule); 8.4.2.2(a) assumes stiffeners at supports."
             if stiffeners_on(inp) else ""),
          "Load factors: peb_frame_model.ULS_RULES (verify IS 800 Table 4). Centre-line model, no haunch rigid zones."]
    L.append(f"Steel grade (all schemes): fy {inp['fy']:g} MPa, fu {inp.get('fu', 490):g} MPa.")
    if inp.get("scheme") == "truss" and truss_family(inp) == "tube":
        L.append("Truss: IS 4923 cold-formed tubes, Table 10 curve b (STAAD uses a); web members pin-ended; chord loads as "
                 "UDL between panel points; KL/r per Table 3; web/chord width 0.35-1.0; RHS joint resistance (chord face, "
                 "punching shear) and weld design are NOT checked - use CIDECT DG3 / EN 1993-1-8 7.5.")
    elif inp.get("scheme") == "truss":
        L.append("Truss: double equal angles back to back on gussets, welded ends; 7.5.2.1 webs KL in plane min(K, 0.85) L, "
                 "7.5.3 chords per 7.2.4; Table 10 curve c; Table 2 (b+d)/t <= 25 eps; 6.3.3 rupture with beta 0.7; KL/r per "
                 "Table 3. STAAD 2026 gives TABLE D double angles single-angle stiffness (tested), so the STAAD file uses "
                 "PRISMATIC double-angle properties and STAAD code-checks only the other members. Not checked: gusset "
                 "plates, welds, block shear (6.4), tack connections (7.8, 10.2.5).")
    return "\n".join(L)


# ============================================================================= PEB vs portal truss
def scheme_input(inp: dict, key: str) -> dict:
    """Input of one scheme: 'peb', 'truss' (SHS/RHS tubes) or 'angle' (double-angle truss)."""
    s = copy.deepcopy(inp)
    s["scheme"] = "peb" if key == "peb" else "truss"
    if key != "peb":
        s["truss"]["family"] = "angle" if key == "angle" else "tube"
        s["name"] = inp["name"] + ("_TA" if key == "angle" else "_TR")
    return s


def truss_keys(inp: dict) -> list:
    fam = inp.get("truss", {}).get("family", "both")
    return {"tube": ["truss"], "angle": ["angle"]}.get(fam, ["truss", "angle"])


def installed_cost(inp: dict, r: dict) -> dict:
    """Installed cost of one frame from the Cost rates: material (plate / tube per kg), fabrication (per kg by
    type + per stiffener + per truss joint), erection per kg, painting per m2 of member surface."""
    c, mdl = {**DEFAULT_INPUT["cost"], **inp.get("cost", {})}, r["model"]
    plate = tube = angle = area = 0.0
    for m in mdl.members:
        kg = mdl.member_props(m)["A"] * m["L"] * RHO_STEEL
        sc = m.get("sec")
        if sc and sc["shape"] == "DA":                                   # 2 angles x 2 legs x both faces
            angle, area = angle + kg, area + 8 * sc["b"] * m["L"]
        elif sc:
            tube, area = tube + kg, area + 2 * (sc["B"] + sc["H"]) * m["L"]
        else:
            pl, D = m["plates"], 0.5 * (m["D_a"] + m["D_b"])
            plate, area = plate + kg, area + (2 * D + 2 * pl["bf_in"] + 2 * pl["bf_out"] - 2 * pl["tw"]) * m["L"]
    st = r["stiffeners"]
    joints = len({n for m in mdl.members if mdl.lines[m["line"]]["type"] == "web" for n in (m["start"], m["end"])})
    gus = joints * c["gusset_kg"] if angle else 0.0                      # angle trusses: a gusset at every joint
    plate += st["mass"] + gus
    kg = plate + tube + angle
    parts = dict(material=plate * c["plate_rs_kg"] + tube * c["tube_rs_kg"] + angle * c["angle_rs_kg"],
                 fabrication=plate * c["fab_builtup_rs_kg"] + (tube + angle) * c["fab_truss_rs_kg"]
                 + st["n"] * c["stiffener_rs"] + joints * c["joint_rs"],
                 erection=kg * c["erect_rs_kg"], painting=area * c["paint_rs_m2"])
    plan = sum(inp["spans"]) * inp["bay"]
    total = sum(parts.values())
    return dict(parts, total=total, per_m2=total / plan, kg=kg, kg_m2=kg / plan, plate_kg=plate, tube_kg=tube,
                angle_kg=angle, gusset_kg=gus, area_m2=area, joints=joints, stiffeners=st["n"])


def peb_triggers(inp: dict, design: dict, r: dict) -> list:
    """PEB economic / fabrication limits (inp.peb_limits) crossed by an optimised PEB design."""
    lim = {**DEFAULT_INPUT["peb_limits"], **inp.get("peb_limits", {})}
    I_ = [v for t, v in design.items() if t != "truss"]
    D = max(max(v["D"]) for v in I_)
    pl = max(max(max(q[0], q[2]) for q in v["plates"]) for v in I_)
    kgm2 = r["weight"] / (sum(inp["spans"]) * inp["bay"])
    out = [msg for bad, msg in (
        (D > lim["D_max_mm"], f"depth {D} mm > {lim['D_max_mm']:g}"),
        (pl > lim["plate_max_mm"], f"plate {pl} mm > {lim['plate_max_mm']:g}"),
        (kgm2 > lim["kg_m2_max"], f"{kgm2:.1f} kg/m2 > {lim['kg_m2_max']:g}"),
        (r["stiffeners"]["n"] > lim["stiffeners_max"], f"{r['stiffeners']['n']} stiffeners > {lim['stiffeners_max']:g}"),
        (r["lam_cr_gravity"] < lim["lam_cr_min"], f"lambda_cr {r['lam_cr_gravity']:.2f} < {lim['lam_cr_min']:g}"),
        (not r["ok"], "no feasible PEB design in the catalogue")) if bad]
    return out


def optimise_both(inp: dict, folder=None, log=print, stop=lambda: False, workers=None, starts=None) -> dict:
    """Tapered welded I-section PEB first; the portal truss always (truss.compare = 'always') or only when the
    PEB crosses a peb_limits entry; installed cost of each. folder -> STAAD-verified (folder/PEB, folder/TRUSS)."""
    ins = {k: scheme_input(inp, k) for k in ["peb"] + truss_keys(inp)}

    def run(k):
        st = (starts or {}).get(k)
        return (optimise_verified(ins[k], Path(folder) / k.upper(), log, stop, start=st) if folder
                else optimise(ins[k], start=st, log=log, stop=stop, workers=workers))
    log("=== SCHEME 1: tapered welded I-section PEB frame")
    out = dict(inputs=dict(peb=ins["peb"]), peb=run("peb"))
    out["triggers"] = peb_triggers(ins["peb"], out["peb"]["design"], out["peb"]["result"])
    log("PEB limits crossed: " + ("; ".join(out["triggers"]) or "none"))
    if inp.get("truss", {}).get("compare", "always") == "always" or out["triggers"]:
        for i, k in enumerate(truss_keys(inp)):
            if stop():
                break
            log(f"=== SCHEME {i + 2}: portal truss, {SCHEME_NAMES[k]}")
            out["inputs"][k], out[k] = ins[k], run(k)
    out["cost"] = {k: installed_cost(out["inputs"][k], out[k]["result"]) for k in out["inputs"]}
    return out


def compare_report(both: dict) -> str:
    C, R = both["cost"], {k: both[k]["result"] for k in both["cost"]}
    ks = list(C)
    name = SCHEME_NAMES
    L = ["SCHEME COMPARISON - installed cost of one frame (INR)",
         "Rates = Cost tab values: PLACEHOLDERS until replaced with your fabricator's / erector's rates", "=" * 78,
         f"  {'':<28}" + "".join(f"{name[k]:>20}" for k in ks)]

    def row(label, fn):
        L.append(f"  {label:<28}" + "".join(f"{fn(k):>20}" for k in ks))
    row("steel kg (kg/m2 plan)", lambda k: f"{C[k]['kg']:.0f} ({C[k]['kg_m2']:.1f})")
    row("plate / tube / angle kg", lambda k: f"{C[k]['plate_kg']:.0f}/{C[k]['tube_kg']:.0f}/{C[k]['angle_kg']:.0f}")
    row("stiffeners / truss joints", lambda k: f"{C[k]['stiffeners']} / {C[k]['joints']}")
    row("max D/C (feasible)", lambda k: f"{R[k]['max_ratio']:.3f} ({'yes' if R[k]['ok'] else 'NO'})")
    row("lambda_cr gravity", lambda k: f"{R[k]['lam_cr_gravity']:.2f}")
    row("paint area m2", lambda k: f"{C[k]['area_m2']:.0f}")
    for part in ("material", "fabrication", "erection", "painting"):
        row(f"{part} Rs", lambda k, part=part: f"{C[k][part]:,.0f}")
    row("TOTAL Rs", lambda k: f"{C[k]['total']:,.0f}")
    row("Rs per m2 plan", lambda k: f"{C[k]['per_m2']:,.0f}")
    L.append(f"  PEB limits crossed: {'; '.join(both['triggers']) or 'none'}")
    ok = sorted((k for k in ks if R[k]["ok"]), key=lambda k: C[k]["total"])
    if len(ok) >= 2:
        a, b = ok[0], ok[1]
        L.append(f"  -> {name[a]} is cheapest: Rs {C[b]['total'] - C[a]['total']:,.0f} per frame "
                 f"({100 * (C[b]['total'] / C[a]['total'] - 1):.1f} %) below {name[b]} at these rates")
    elif ok:
        L.append(f"  -> only {name[ok[0]]} found a feasible design")
    if any(C[k]["gusset_kg"] for k in ks):
        L.append(f"  angle truss steel includes gussets: {', '.join(f'{C[k]['gusset_kg']:.0f} kg' for k in ks if C[k]['gusset_kg'])}")
    return "\n".join(L)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))


if __name__ == "__main__":
    import sys
    inp = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else DEFAULT_INPUT
    out = optimise(inp)
    print(report(inp, out))
