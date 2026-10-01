"""Self-checks: section properties, sign conventions, orientation mapping, statics."""
import math, numpy as np
from peb_frame_model import *

def P(tw, bf, tf, bf2=None, tf2=None):
    return Plates(tw, bf, tf, bf2 or bf, tf2 or tf)

# ---------- 1. I-section properties vs closed form (symmetric)
D, p = 0.5, P(0.006, 0.2, 0.012)
s = i_section(D, p)
hw = D - 0.024
A_cf  = 2*0.2*0.012 + hw*0.006
Iz_cf = (0.2*D**3 - (0.2-0.006)*hw**3)/12
Zp_cf = 0.2*0.012*(D-0.012) + 0.006*hw**2/4
Iw_cf = (0.012*0.2**3/12)*(D-0.012)**2/2
for k, a, b in [("A",s["A"],A_cf),("Iz",s["Iz"],Iz_cf),("Zp",s["Zp"],Zp_cf),("Iw",s["Iw"],Iw_cf)]:
    assert abs(a-b)/b < 1e-9, (k, a, b)
print("1. I-section A, Iz, Zp, Iw match closed form")

# ---------- 2. single-span near-flat portal, prismatic, pinned: knee M vs force method
L, h = 20.0, 6.0
pr = Profile([(0, 0.6), (1, 0.6)], [P(0.008, 0.25, 0.012)])
fi = FrameInput(name="T2", spans=[L], eave_height=h, slope=1e-6, bay=1.0, purlin_spacing=1.0,
                girt_spacing=1.0, rafter_SR_ext=pr, rafter_SR_int=pr, col_ext=pr, col_int=pr,
                loads=Loads(q_sdl_slope=0, q_coll_plan=0, q_ll_plan=10.0, q_wall=0))
mdl = FrameModel(fi); fe = LinearFrame2D(mdl)
res = fe.solve({101: 1.0})
st = fe.stations(res)
w = 10.0
k = 1.0 * h / L                    # Ib = Ic
M_formula = w*L**2/(4*(3+2*k))     # pinned-base portal, UDL on beam (force method)
M_knee_raf = st["R0"][0]["M"][0]
M_knee_col = st["C0"][-1]["M"][0]
print(f"2. knee M: FE rafter {M_knee_raf:.3f}, FE column {M_knee_col:.3f}, formula {M_formula:.3f} kNm")
assert M_knee_raf > 0 and M_knee_col > 0, "hogging knee must put INNER flange in compression (+)"
assert abs(M_knee_raf - M_formula)/M_formula < 0.01
Rv = sum(r[1] for r in res["R"].values())
assert abs(Rv - w*L) < 1e-6 * w * L
ridge = st["R0"][-1]["M"][0]
assert ridge < 0, "mid-span sagging must put inner flange in TENSION (-)"
print(f"   mid-span M_ic = {ridge:.3f} (sagging, formula {w*L**2/8 - M_formula:.3f}); statics OK")

# ---------- 3. tapered two-span leaning column: continuity at every station, statics, symmetry
SR = Profile([(0, 0.80), (0.30, 0.45), (0.80, 0.55), (1.0, 0.55)],
             [P(0.006, 0.20, 0.012), P(0.005, 0.18, 0.008), P(0.005, 0.18, 0.008)])
SRi = Profile([(0, 0.90), (0.30, 0.45), (0.80, 0.55), (1.0, 0.55)], SR.plates)
CE = Profile([(0, 0.30), (1, 0.80)], [P(0.006, 0.22, 0.012, 0.25, 0.010)])   # unequal flanges
CI = Profile([(0, 0.25), (1, 0.25)], [P(0.006, 0.20, 0.010)])
fi = FrameInput(name="T3", spans=[24, 24], eave_height=8, slope=0.1, rafter_SR_ext=SR,
                rafter_SR_int=SRi, col_ext=CE, col_int=CI,
                loads=Loads(q_ll_plan=0.75))
mdl = FrameModel(fi); fe = LinearFrame2D(mdl)
nrev = sum(m["reversed"] for m in mdl.members)
print(f"3. {len(mdl.nodes)} nodes, {len(mdl.members)} members, {nrev} reversed for f1>=f3")
assert all(m["f1"] >= m["f3"] - 1e-12 for m in mdl.members)
for lc in (1, 101, 102):
    res = fe.solve({lc: 1.0})
    tot = mdl.case_totals(lc)
    Rx = sum(r[0] for r in res["R"].values()); Ry = sum(r[1] for r in res["R"].values())
    assert abs(Rx + tot["FX"]) < 1e-6 and abs(Ry + tot["FY"]) < 1e-6*abs(tot["FY"]), (lc, Rx, Ry, tot)
    st = fe.stations(res)
    for name, sts in st.items():
        for s_ in sts:
            if len(s_["M"]) == 2:
                assert abs(s_["M"][0] - s_["M"][1]) < 1e-6 * (1 + abs(s_["M"][0])), (name, s_)
                assert abs(s_["N"][0] - s_["N"][1]) < 1e-6 * (1 + abs(s_["N"][0])), (name, s_)
print("   statics (sum R = -sum loads) and M, N continuity across reversed members: OK")
res = fe.solve({1: 1.0, 101: 1.0, 102: 1.0}); st = fe.stations(res)
L0, L3 = st["R0"], st["R3"]
assert abs(L0[0]["M"][0] - L3[-1]["M"][0]) < 1e-6 * abs(L0[0]["M"][0]), "symmetry"
assert L0[0]["M"][0] > 0 and st["C0"][-1]["M"][0] > 0
# unequal column flanges: left & right columns must have mirrored flange assignment in STAAD
lcol = [m for m in mdl.members if m["line_name"] == "C0"][0]
rcol = [m for m in mdl.members if m["line_name"] == "C2"][0]
assert lcol["reversed"] and rcol["reversed"] and lcol["top_is_outer"] != rcol["top_is_outer"]
print("   symmetric gravity -> symmetric knee moments; knees hogging (inner flange compression)")
print("   exterior columns: start=top (deeper), top-flange side mirrored L/R as required")
print("ALL CHECKS PASSED")
