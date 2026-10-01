"""python test_optimiser.py  - fast-engine checks (no STAAD needed)."""
import copy
import math

import numpy as np

import optimiser as o
from peb_frame_model import FrameModel, LinearFrame2D

I = copy.deepcopy(o.DEFAULT_INPUT)
seed = {t: I["templates"][t] for t in o.active_templates(I)}

# 1. catalogue encode/decode round trip on catalogue values
var = o.variables(I, list(seed))
x = o.encode(I, seed, var)
assert o.encode(I, o.decode(I, var, x), var) == x

# 2. P-Delta moments bracketed by first order and first order / (1 - 1/lambda_cr)
r = o.evaluate(I, seed)
mdl = FrameModel(o.frame_input(I, seed))
fe = LinearFrame2D(mdl, shear=True)                          # same stiffness as the P-Delta engine
u01 = mdl.uls[0]
lin = fe.solve(dict(u01["pairs"]))
lam = r["lam_cr"][u01["lc"]]
assert 1.5 < lam < 3.5, lam
knee = next(m for m in mdl.members if m["line_name"] == "R0" and m["j"] == 0)["id"]
m1 = abs(lin["ends"][knee][0][5])
m2 = abs(r["ends"][u01["lc"]][r["mids"].index(knee), 0, 5])
assert 0.98 * m1 < m2 < m1 / (1 - 1 / lam), (m1, m2, lam)       # P-Delta between 1st order and amplified bound
assert r["weight"] > 3000 and r["lam_cr_gravity"] == min(r["lam_cr"].values())
print(f"seed: {r['weight']:.0f} kg, max D/C {r['max_ratio']:.3f}, lambda_cr {r['lam_cr_gravity']:.2f}, "
      f"knee M 1st order {m1:.1f} / P-Delta {m2:.1f} kNm")

# 3. a stiff single-span frame optimises to a feasible, lighter design
S = copy.deepcopy(I)
S.update(name="T1", spans=[15.0], eave_height=6.0)
S["wind"]["cases"] = [dict(label="+X", walls=[0.7, -0.3], roof=[-0.9, -0.4])]
S["wind"]["cpi"] = [0.2]
S["templates"]["raf_ext"] = dict(knots=[0, 0.5, 1], D=[700, 450, 450], plates=[[8, 250, 14], [6, 200, 10]])
S["templates"]["col_ext"] = dict(knots=[0, 1], D=[400, 700], plates=[[8, 250, 14]])
start = o.evaluate(S, {t: S["templates"][t] for t in o.active_templates(S)})
out = o.optimise(S, log=lambda s: None, shake_passes=0, workers=1)
res = out["result"]
assert res["ok"] and res["max_ratio"] <= 1.0 + 1e-9, res["max_ratio"]
assert res["weight"] < start["weight"], (res["weight"], start["weight"])
assert all(math.isfinite(x["ratio"]) for x in res["rows"])
print(f"single span: {start['weight']:.0f} -> {res['weight']:.0f} kg, max D/C {res['max_ratio']:.3f}, "
      f"{out['evals']} analyses")
# 4. slender-web option (IS 800 8.2.1.1(a)) can only widen the feasible set -> never heavier
S["options"] = dict(slender_web=True)
out_sw = o.optimise(S, start=out["design"], log=lambda s: None, shake_passes=0, workers=1)
assert out_sw["result"]["ok"] and out_sw["result"]["weight"] <= res["weight"] + 1e-6
print(f"single span, slender webs allowed: {out_sw['result']['weight']:.0f} kg")
# 5. transverse stiffeners (penalty 0) only add options -> cost never above the unstiffened optimum
S["options"] = dict(slender_web=True, stiffeners=True, stiffener_penalty_kg=0.0)
out_st = o.optimise(S, start=out_sw["design"], log=lambda s: None, shake_passes=0, workers=1)
rs, st = out_st["result"], out_st["result"]["stiffeners"]
assert rs["ok"] and rs["cost"] <= out_sw["result"]["weight"] + 1e-6, (rs["cost"], out_sw["result"]["weight"])
assert st["n"] == len(st["items"]) and (st["n"] == 0) == (st["mass"] == 0)
assert all(x["stiff_k"] == st["layout"][x["member"]] for x in rs["rows"] if x["member"])
print(f"single span, slender webs + stiffeners: {rs['weight']:.0f} kg, {st['n']} stiffeners ({st['mass']:.1f} kg)")
# 6. tension field (8.4.2.2(b)): shear-dominated 14 m span, 6 mm web -> used in interior panels, fewer stiffeners,
#    every chain anchored (8.5.1), stiffener forces within Fqd, never in the panel next to a support
T = copy.deepcopy(o.DEFAULT_INPUT)
T.update(name="TFX", spans=[14.0], eave_height=6.0)
T["wind"]["cases"], T["wind"]["cpi"] = [dict(label="+X", walls=[0.7, -0.3], roof=[-0.9, -0.4])], [0.2]
T["loads"]["q_coll_plan"] = 6.0
T["templates"]["raf_ext"] = dict(knots=[0, 1], D=[900, 900], plates=[[6, 300, 20]])
T["templates"]["col_ext"] = dict(knots=[0, 1], D=[900, 900], plates=[[10, 300, 20]])
res = {}
for tf in (False, True):
    T["options"].update(slender_web=True, stiffeners=True, tension_field=tf)
    res[tf] = o.evaluate(T, o.compatible(T, T["templates"]))
sa, sb = res[False]["stiffeners"], res[True]["stiffeners"]
assert sb["tension_field"] and not sa["tension_field"] and sb["n"] < sa["n"], (sa["n"], sb["n"])
assert res[True]["max_ratio"] <= res[False]["max_ratio"] + 1e-9
assert all(an["ratio"] <= 1.0 for an in sb["anchors"]) and all(i["F"] <= i["Fqd"] for i in sb["items"])
mdl = res[True]["model"]
ends = {ln["members"][0] for ln in mdl.lines} | {ln["members"][-1] for ln in mdl.lines}
assert not ends & set(sb["tension_field"])
print(f"tension field: stiffeners {sa['n']} -> {sb['n']}, TF members {sb['tension_field']}, "
      f"max anchor ratio {max(an['ratio'] for an in sb['anchors']):.2f}")
# 7. end posts (8.5.2(b)). With only one stiffener spacing (k = 1) the knee panel can pass only by tension field
#    + end post; with all spacings the cheaper (2, a) wins, since an end post costs one stiffener.
T["loads"]["q_coll_plan"] = 3.5
T["options"].update(slender_web=True, stiffeners=True, tension_field=True, end_posts=True)
K0, o.K_STIFF = o.K_STIFF, (1,)
try:
    r7 = o.evaluate(T, o.compatible(T, T["templates"]))
finally:
    o.K_STIFF = K0
st7 = r7["stiffeners"]
eps = [an for an in st7["anchors"] if an.get("end_post")]
assert st7["end_posts"] and eps and all(an["end_post"]["ratio"] <= 1 for an in eps), st7["end_posts"]
assert all(an["member"] in st7["tension_field"] for an in eps)
assert {an["member"] for an in eps} == {i["member"] for i in st7["items"] if i.get("end_post")}   # in the schedule
r7b = o.evaluate(T, o.compatible(T, T["templates"]))
assert not r7b["stiffeners"]["end_posts"] and r7b["stiffeners"]["n"] <= st7["n"]
print(f"end posts: {len(eps)} designed (k = 1 only), e = {sorted({an['end_post']['e'] for an in eps})} m; "
      f"all spacings -> none needed ({r7b['stiffeners']['n']} stiffeners vs {st7['n']})")
# 8. purlin loads into the web (8.7.4 / 8.7.3.1). Statics: an interior purlin takes SDL x bay x sub-member length
#    (normal component). A 4 mm web 700 deep fails web buckling bare (KL = d) and passes with KL = 0.7 d; with the
#    stiffener option, load-carrying stiffeners are added at exactly those purlins (8.7.5).
W = copy.deepcopy(o.DEFAULT_INPUT)
mw = o.FrameModel(o.frame_input(W, W["templates"]))
m3 = mw.mem_index[mw.lines[3]["members"][3]]
assert abs(o.purlin_reactions(mw)[("R0", 3)][1] + 0.15 * 7.5 * m3["L"] / (1 + 0.1 ** 2) ** 0.5) < 1e-9
for t in ("raf_ext", "raf_int"):
    W["templates"][t] = dict(knots=[0, 0.3, 0.8, 1], D=[900, 700, 700, 600], plates=[[6, 200, 12], [4, 200, 12], [5, 200, 10]])
W["options"].update(slender_web=True)
pw = {}
for keff, st_on in ((1.0, False), (0.7, False), (1.0, True)):
    W["purlin"]["web_KL"], W["options"]["stiffeners"] = keff, st_on
    pw[keff, st_on] = o.evaluate(W, o.compatible(W, W["templates"]))["stiffeners"]["purlin"]
assert pw[1.0, False]["web_max"] > 1.0 > pw[0.7, False]["web_max"], pw
assert pw[1.0, True]["n_extra"] > 0 and pw[1.0, True]["web_max"] <= 1.0
print(f"purlins: bare 4 mm web {pw[1.0, False]['web_max']:.2f} (KL d) / {pw[0.7, False]['web_max']:.2f} (KL 0.7d); "
      f"{pw[1.0, True]['n_extra']} load-carrying stiffeners added")
# 9. six-section rafters: every decoded design obeys the shape rules (any catalogue indices), the projection is
#    idempotent, knots follow the moment envelope, and optimise() returns six sections
import random
import tempfile

from peb_frame_model import end_forces_to_stations

S6 = copy.deepcopy(o.DEFAULT_INPUT)
var6, rng = o.variables(S6, o.active_templates(S6)), random.Random(1)
for _ in range(200):
    d6 = o.decode(S6, var6, [rng.randrange(len(o.CAT[g[0][1]])) for g in var6])
    top = min(d6[t]["D"][0] for t in o.SIX)
    for t in o.SIX:
        D, s, pl = d6[t]["D"], d6[t]["knots"], d6[t]["plates"]
        assert D[0] >= D[1] >= D[2] >= D[3] == D[4] <= D[5] <= D[6] <= top, D
        assert D[5] >= D[4] + (D[6] - D[4]) * (s[5] - s[4]) / (s[6] - s[4]) - 1e-9, D        # no apex spike
        assert pl[0] == pl[1] and pl[2] == pl[3] and pl[4] == pl[5]
    assert d6["raf_ext"]["D"][6] == d6["raf_int"]["D"][6] and d6["raf_ext"]["plates"][5] == d6["raf_int"]["plates"][5]
    x6 = o.encode(S6, d6, var6)
    assert o.encode(S6, o.decode(S6, var6, x6), var6) == x6
env = [(np.linspace(0, 1, 10), np.array([500, 380, 270, 170, 90, 60, 110, 170, 210, 230.0]))]
assert o.six_knots(env, 9) == [0, 2, 3, 4, 6, 7, 9], o.six_knots(env, 9)
assert len(out["design"]["raf_ext"]["knots"]) == 7
tr = [x for x in o.evaluate(S6, o.compatible(S6, S6["templates"]))["rows"] if x["check"].startswith("taper")]
assert len(tr) == 4 * 5, len(tr)                                  # 4 rafter pieces x 5 non-haunch segments
r01 = next(x for x in tr if x["line"] == "R0" and x["j"] == 1)    # 700 -> 600 mm over one 12.06/9 m station
assert abs(r01["M"] - 100 / (math.hypot(12, 1.2) / 9)) < 1e-6 and abs(r01["ratio"] - r01["M"] / 150) < 1e-12
print(f"six sections: shape rules hold on 200 random designs; test-3 knots {out['design']['raf_ext']['knots']}")
# 10. crane on brackets (IS 875-2 6.3): wheel / bracket loads by hand, statics of the crane cases, bracket root
#     moment = R e, 8 arrangements for one crane, SLS crane rows present
C = copy.deepcopy(o.DEFAULT_INPUT)
C["crane"]["spans"] = [1]
cr, dC = C["crane"], o.compatible(C, C["templates"])
mc = FrameModel(o.frame_input(C, dC))
R = mc.crane_info[0]["R"]
Lc, lift = 24 - 2 * cr["e"], cr["crab_kN"] + cr["capacity_kN"]
il = 1 + (1 - cr["wheel_base"] / 7.5)
assert abs(R["Rmax"] - (cr["bridge_kN"] / 4 + lift * (Lc - cr["a_min"]) / Lc / 2) * 1.25 * il) < 1e-9
assert abs(R["Rmin"] - (cr["bridge_kN"] / 4 + lift * cr["a_min"] / Lc / 2) * il) < 1e-9
assert abs(R["H"] - 0.05 * lift / 2 * il) < 1e-9
fe = LinearFrame2D(mc)
sv = fe.solve({501: 1.0})
assert abs(sum(v[1] for v in sv["R"].values()) - (R["Rmax"] + R["Rmin"])) < 1e-6
mb = mc.mem_index[mc.brackets[0]["members"][0]]
m_root = next(M for j, N, V, M in end_forces_to_stations(mb, *sv["ends"][mb["id"]]) if j == 0)
assert abs(m_root - R["Rmax"] * cr["e"]) < 1e-6, (m_root, R["Rmax"] * cr["e"])     # hogging: bottom flange compressed
sh = fe.solve({503: 1.0})
assert abs(sum(v[0] for v in sh["R"].values()) + R["H"]) < 1e-6
assert len(mc.crane_variants()) == 8
assert sum(1 for c in mc.uls if any(lc in mc.lc_cl for lc, _ in c["pairs"])) == 8 * (2 + 4 * len(mc.lc_wl))
rc = o.evaluate(C, dC)
assert {x["check"] for x in rc["rows"] if x["line"].startswith("K1")} == {"crane drift H/200", "rail spread 10mm"}
# combination screening (search): every combination = the full check; a subset can only report less
gov = {x["lc"] for x in rc["rows"] if x["lc"] in {c["lc"] for c in mc.uls}}
r_all = o.evaluate(C, dC, only=frozenset(c["lc"] for c in mc.uls))
r_gov = o.evaluate(C, dC, only=frozenset(gov))
assert r_all["screened"] and not rc["screened"]
assert abs(r_all["max_ratio"] - rc["max_ratio"]) < 1e-12 and abs(r_all["viol"] - rc["viol"]) < 1e-12
assert abs(r_gov["max_ratio"] - rc["max_ratio"]) < 1e-9          # the governing combinations reproduce it
assert r_gov["viol"] <= rc["viol"] + 1e-9
print(f"crane: Rmax {R['Rmax']:.1f} kN, Rmin {R['Rmin']:.1f} kN, surge {R['H']:.2f} kN per bracket; "
      f"{len(mc.uls)} ULS combinations, {len(gov)} govern (screening exact on them)")
# 11. STAAD export carries the design parameters (IS800 LSD block, deflection check), STAAD line length
with tempfile.TemporaryDirectory() as td:
    ps = o.export_std(C, dC, td, log=lambda s: None)
    u, sl = ps["ULS"].read_text(), ps["SLS"].read_text()
    for key in ("PDELTA", "CODE IS800 LSD", "FYLD 345000", "FU 490000", "STP 2", "LZ ", "LY ", "LX ", "TRACK 2",
                "CHECK CODE ALL", " MZ "):
        assert key in u, key
    for key in ("span/180", "H/150", "rail gauge", "LOAD LIST", "PRINT JOINT DISPLACEMENTS LIST"):
        assert key in sl, key
    assert all(len(ln) <= 72 for f in ps.values() for ln in f.read_text().splitlines() if not ln.startswith("*"))
    uT = o.export_std(T, o.compatible(T, T["templates"]), td + "/t", log=lambda s: None)["ULS"].read_text()
    assert "TST 1 MEMB" in uT and "\nTSP " in uT and "STIFF" not in uT          # IS800 LSD stiffener parameters
print("export: ULS file carries the IS800 LSD parameters; SLS file the Table 6 limits + check-node displacements")
# 12. portal truss scheme: geometry (bottom chord h0 below the eaves at the columns, bs x rise at the ridge), web
#     members pin-ended (no end moment), statics, leaning interior column axial-only, catalogue chain, cost, STAAD file
X = o.scheme_input(o.DEFAULT_INPUT, "truss")
dX = o.compatible(X, X["templates"])
dX["truss"].update(h0=[1500], bs=[0.5])
mX = FrameModel(o.frame_input(X, dX))
yb = [st["y"] for ln in mX.lines if ln["type"] == "bot" for st in (ln["stations"][0], ln["stations"][-1])]
assert min(yb) == 8.0 - 1.5 and abs(max(yb) - (8.0 - 1.5 + 0.5 * 1.2)) < 1e-9, yb
assert {ln["type"] for ln in mX.lines} == {"col", "raf", "bot", "web"}
assert sum(ln["kind"] == "vert" for ln in mX.lines) == 4 * 8 + 2                   # 8 interior per piece + 2 ridges
assert sum(ln["kind"] == "diag" for ln in mX.lines) == 4 * 9
sX = LinearFrame2D(mX).solve({1: 1.0, 101: 1.0})
assert abs(sum(v[1] for v in sX["R"].values()) + mX.case_totals(1)["FY"] + mX.case_totals(101)["FY"]) < 1e-6
assert max(abs(sX["ends"][m["id"]][e][5]) for m in mX.members if m.get("truss") for e in (0, 1)) < 1e-9
c1 = mX.line_by_name("C1")
assert all(abs(sX["ends"][mid][e][5]) < 1e-6 for mid in c1["members"] for e in (0, 1))    # leaning: no moment
T_ = o.TUBES
assert all(b["w"] > a["w"] and b["A"] > a["A"] for a, b in zip(T_, T_[1:]))             # step up = more steel
rX = o.evaluate(X, dX)
assert not rX.get("error") and all(math.isfinite(x["ratio"]) for x in rX["rows"])
assert {x["check"] for x in rX["rows"]} >= {"joint b1/b0 <= 1", "joint b1/b0 >= 0.35"}
cX = o.installed_cost(X, rX)
assert cX["joints"] > 0 and cX["tube_kg"] > 0 and abs(cX["total"] - sum(cX[k] for k in ("material", "fabrication",
                                                                                         "erection", "painting"))) < 1e-6
with tempfile.TemporaryDirectory() as td:
    uX = o.export_std(X, dX, td, r0=rX, log=lambda s: None)["ULS"].read_text()
    assert "MEMBER PROPERTY 'INDIA (IS 4923-2017).DB3'" in uX and "TABLE 'SHS' ST 'SHS" in uX and "MEMBER TRUSS" in uX
    assert "FYLD 345000 ALL" in uX and "FYLD 310000" not in uX and "STP 1 MEMB" in uX       # one grade, all schemes
print(f"truss: {len(mX.members)} members, seed {rX['weight']:.0f} kg, max D/C {rX['max_ratio']:.2f}, "
      f"lambda_cr {rX['lam_cr_gravity']:.1f}, installed Rs {cX['total']:,.0f}")
# 13. double-angle truss and single-ridge roof: angle catalogue holds only Table 2 class-3 angles at the frame grade,
#     a tube seed maps to angles of similar kg/m, PRISMATIC double angles in STAAD (STAAD's TABLE D is single-angle
#     stiffness) and left out of CHECK CODE, gussets costed; single ridge: sides, diagonals toward the centre, statics
G = o.scheme_input(o.DEFAULT_INPUT, "angle")
dG = o.compatible(G, G["templates"])
eps = math.sqrt(250 / G["fy"])
assert all(n.startswith("2-ISEA") for sd in ("ext", "int") for n in dG["truss"][sd])
assert all(2 * d["b"] / d["t"] <= 25 * eps + 1e-9 for d in o.angle_chain(345.0, 0.008))
rG = o.evaluate(G, dG)
assert not rG.get("error") and any(x["check"].startswith("KL/r") for x in rG["rows"])
assert not any(x["check"].startswith("joint") for x in rG["rows"])                       # tube joint rules only
cG = o.installed_cost(G, rG)
assert cG["angle_kg"] > 0 and cG["tube_kg"] == 0 and abs(cG["gusset_kg"] - cG["joints"] * 6.0) < 1e-9
with tempfile.TemporaryDirectory() as td:
    uG = o.export_std(G, dG, td, r0=rG, log=lambda s: None)["ULS"].read_text()
    assert " PRIS AX " in uG and "CHECK CODE MEMB" in uG and "CHECK CODE ALL" not in uG and "TABLE D" not in uG
R_ = copy.deepcopy(o.DEFAULT_INPUT)
R_.update(roof="single_ridge", spans=[20.0, 20.0, 20.0])
R_["wind"]["cases"] = [dict(label="+X", walls=[0.7, -0.3], roof=[-0.9, -0.5, -0.4, -0.3])]
RS = o.scheme_input(R_, "truss")
mR = FrameModel(o.frame_input(RS, o.compatible(RS, RS["templates"])))
assert [(p["kind"], mR.side(p)) for p in mR.pieces] == [("S-S", "ext"), ("S-R", "int"), ("R-S", "int"), ("S-S", "ext")]
assert o.truss_sides(RS) == ["ext", "int"] and o.truss_sides(o.scheme_input(o.DEFAULT_INPUT | dict(spans=[24.0]), "truss")) == ["ext"]
p0 = mR.pieces[0]
top0 = mR.line_by_name("R0")["stations"]
d0 = [ln for ln in mR.lines if ln["kind"] == "diag" and p0["x0"] < sum(st["x"] for st in ln["stations"]) / 2 < p0["x1"]]
first = min(d0, key=lambda ln: min(st["x"] for st in ln["stations"]))                  # Pratt: top at the column side
assert first["stations"][0]["y"] > first["stations"][1]["y"] and first["stations"][0]["x"] < first["stations"][1]["x"]
last = max(d0, key=lambda ln: max(st["x"] for st in ln["stations"]))                   # ... and mirrored at the far column
assert last["stations"][0]["y"] > last["stations"][1]["y"] and last["stations"][0]["x"] > last["stations"][1]["x"]
sR = LinearFrame2D(mR).solve({1: 1.0})
assert abs(sum(v[1] for v in sR["R"].values()) + mR.case_totals(1)["FY"]) < 1e-6
print(f"angle truss: {len(o.CAT['angle'])} class-3 double angles at fy {G['fy']:g}, seed {rG['weight']:.0f} kg + "
      f"{cG['gusset_kg']:.0f} kg gussets; single-ridge truss {len(mR.members)} members")
print("OPTIMISER CHECKS PASSED")
