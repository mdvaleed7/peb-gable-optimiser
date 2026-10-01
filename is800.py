"""
is800.py - IS 800:2007 (LSM) strength checks of a welded I-section at one station.

Split in two so the optimiser can reuse the force-independent part:
    cap = capacities(D, plates, fy, L_in, L_out, L_y, L_z, slender_web, c)   # section + member resistances
    r   = ratios(cap, N, V, M_ic)                            # demand/capacity, <= 1.0 passes

Units: m, kN, kNm; fy in MPa. N > 0 = compression, M_ic > 0 = inner flange in compression.
Clauses (IS 800:2007 - check against your copy): Table 2 (classification), Table 5 (gamma_m),
7.1.2 (compression, Table 7/10 buckling curves), 8.2.1.2 (Md), 8.2.2 (LTB), 8.4 (shear incl. 8.4.2.2(a)
simple post-critical web buckling), 9.2.2 (high shear), 9.3.1 (section, linear form),
9.3.2.2 (member buckling, My = 0).
Slender-web option (slender_web=True), used only where the web exceeds the Table 2 semi-compact limit:
    3.7.2   slender webs designed per 8.2.1.1 (flexure) and 8.4.2.2 (shear)
    8.2.1.1(a) moment and axial force resisted by the flanges only; web resists shear only (8.4)
    8.6.1.1(a) d/tw <= 200 eps_w (no transverse stiffeners, web welded to both flanges)
    8.6.1.2(a) d/tw <= 345 eps_f^2 (compression flange buckling into the web)
Transverse stiffeners (c = spacing, None = stiffeners at supports only):
    8.4.2.2(a) kv = 4 + 5.35/(c/d)^2 (c/d < 1), 5.35 + 4/(c/d)^2 (c/d >= 1); threshold 67 eps sqrt(kv/5.35) (8.4.2.1)
    8.6.1.1(b) d/tw <= 200 eps_w (d <= c <= 3d), c/tw <= 200 eps_w (0.74d <= c < d), 270 eps_w (c < 0.74d);
               c > 3d -> unstiffened;  8.6.1.2(b) d/tw <= 345 eps_f^2 (c >= 1.5d), 345 eps_f (c < 1.5d)
    stiffener(): single-sided flat, 8.7.1.2 outstand <= 20 t eps (core 14 t eps), 8.7.2.4 Is about the web face.
    8.7.2.5 Fq = V - Vcr/gm0 <= 0 automatically while V <= Vd (no tension field), so Fqd is not needed.
    8.4.2.2(a) assumes transverse stiffeners at the supports (knee / ridge / base connections).
Tension field (tension_field=True, stiffened panels with c/d >= 1): 8.4.2.2(b)
    Vtf = Av tau_b + 0.9 w_tf tw fv sin(phi) <= Vp,  phi = atan(d/c),  psi = 1.5 tau_b sin(2 phi),
    fv = sqrt(fyw^2 - 3 tau_b^2 + psi^2) - psi,  s = (2/sin phi) sqrt(Mfr / (fyw tw)) <= c,
    Mfr = 0.25 bf tf^2 fyf [1 - (Nf / (bf tf fyf / gm0))^2]
    w_tf: printed as d cos(phi) + (c - sc - st) sin(phi); the Porter-Rockey-Evans basis has a minus sign
    (wider band for stiffer flanges). Used here: w_tf = max(0, d cos(phi) - |c - sc - st| sin(phi)), the lower
    of the two readings.  Anchorage (8.5.1 / 8.5.3) and the stiffener force (8.7.2.5) are checked by the
    optimiser at panel level. A panel next to a support needs an end post (8.5.2): end_post() designs the
    double-stiffener form (b); the single-stiffener form (a) (width / thickness <= flange) is not implemented.
Shear area of WELDED I-sections: Av = d tw (8.4.1.1).
Concentrated loads through a flange (purlins): web_point_load() - 8.7.4 bearing and 8.7.3.1 web buckling.
  ponytail: flange-only member checks reuse 7.1.2 / 8.2.2 with flange area and flange plastic moment
  (full-section r, Mcr, Iw); interpretation of 8.2.1.1(a) for members - verify against your practice.

ponytail: deliberate conservative simplifications, each with its upgrade path
  - Without slender_web, Class 4 sections are rejected via the 'cls' ratio. Slender FLANGES are always
    rejected (8.2.1.1 needs plastic/compact/semi-compact flanges).
  - Flange outstand b = bf/2 (not (bf - tw)/2).  Class 3 uses min(Ze_in, Ze_out).
  - Cmz = 1.0 and KLT = 1.0 in 9.3.2.2 (both upper bounds).  Upgrade: Table 18 from the M diagram.
  - Mcr uses the doubly-symmetric formula with the station's own Iy, It, Iw (mono-symmetry and taper ignored).
"""
from __future__ import annotations

import math

from peb_frame_model import E_STEEL, POISSON, RHO_STEEL, Plates, i_section

G_STEEL = E_STEEL / (2 * (1 + POISSON))
GM0 = 1.10                                      # Table 5, yielding
ALPHA = dict(a=0.21, b=0.34, c=0.49, d=0.76)    # Table 7 imperfection factors
CURVE_Z, CURVE_Y = "b", "c"                     # Table 10, welded I, tf <= 40 mm: z-z b, y-y c
ALPHA_LT = 0.49                                 # 8.2.2, welded sections


def chi(lam: float, alpha: float) -> float:
    """Reduction factor, 7.1.2.1 / 8.2.2 form."""
    phi = 0.5 * (1 + alpha * (lam - 0.2) + lam * lam)
    return min(1.0, 1.0 / (phi + math.sqrt(max(phi * phi - lam * lam, 0.0))))


def kv(c, d: float) -> float:
    """8.4.2.2(a) shear buckling coefficient; c = None -> transverse stiffeners at supports only."""
    if c is None:
        return 5.35
    r = c / d
    return 4.0 + 5.35 / r ** 2 if r < 1.0 else 5.35 + 4.0 / r ** 2


def web_limit_86(d: float, c, eps_w: float, eps_f: float) -> float:
    """Largest d/tw allowed by 8.6.1.1 (serviceability) and 8.6.1.2 (flange buckling into the web)."""
    if c is None or c > 3 * d:                                  # 8.6.1.1(b)(4): c > 3d -> unstiffened
        return min(200 * eps_w, 345 * eps_f ** 2)
    serv = 200 * eps_w if c >= d else 200 * eps_w * d / c if c >= 0.74 * d else 270 * eps_w
    return min(serv, 345 * eps_f ** 2 if c >= 1.5 * d else 345 * eps_f)


STIFF_T = [0.006, 0.008, 0.010, 0.012, 0.016]                   # stiffener plate thicknesses (m)


def stiffener_fqd(d: float, tw: float, ts: float, bs: float, fy: float) -> float:
    """8.7.1.5 buckling resistance of a single-sided flat stiffener: core outstand (14 t eps) plus 20 tw of web
    each side, curve c, KL = 0.7 d, r about the axis parallel to the web."""
    f, eps = fy * 1e3, math.sqrt(250.0 / fy)
    bc = min(bs, 14 * ts * eps)
    As, Aw = bc * ts, 40 * tw * tw
    zs = tw / 2 + bc / 2                                        # stiffener centroid from the web mid-plane
    zb = As * zs / (As + Aw)
    I = Aw * zb ** 2 + 40 * tw * tw ** 3 / 12 + ts * bc ** 3 / 12 + As * (zs - zb) ** 2
    lam = math.sqrt(f / (math.pi ** 2 * E_STEEL)) * 0.7 * d / math.sqrt(I / (As + Aw))
    return chi(lam, ALPHA["c"]) * (As + Aw) * f / GM0


def stiffener(d: float, tw: float, bf: float, fy: float, c: float, F: float = 0.0):
    """Lightest single-sided flat intermediate stiffener (outstand to the flange tip) with Is >= 8.7.2.4 and
    axial force F <= Fqd (8.7.2.5 Fq plus any anchor force), or None."""
    eps = math.sqrt(250.0 / fy)
    Is_min = 0.75 * d * tw ** 3 if c / d >= math.sqrt(2) else 1.5 * d ** 3 * tw ** 3 / c ** 2   # 8.7.2.4
    for ts in STIFF_T:
        bs = min((bf - tw) / 2, 20 * ts * eps)                  # 8.7.1.2 outstand limit
        Is = ts * min(bs, 14 * ts * eps) ** 3 / 3               # core section, about the web face
        Fqd = stiffener_fqd(d, tw, ts, bs, fy)
        if Is >= Is_min and F <= Fqd:
            return dict(ts=ts, bs=bs, d=d, Is=Is, Is_min=Is_min, F=F, Fqd=Fqd, mass=ts * bs * d * RHO_STEEL,
                        weld_kN_per_mm=(tw * 1e3) ** 2 / (5 * bs * 1e3))            # 8.7.2.6
    return None


def web_point_load(D: float, d: float, tw: float, tf: float, fy: float, b1: float, k_eff: float = 1.0):
    """Local web capacity under a load applied through a flange away from a member end (m, kN):
      8.7.4   bearing   Fw   = (b1 + n2) tw fyw / gm0,  n2 = 2 x 2.5 tf (1:2.5 through the flange, both sides)
      8.7.3.1 buckling  Fcdw = (b1 + n1) tw fcd,        n1 = D (45 deg to mid-depth, both sides), web strip as
              a strut about the axis parallel to the web, r = tw / sqrt(12), KL = k_eff d (8.7.1.5: 0.7 when the
              loaded flange is restrained against rotation, 1.0 when not), curve c (7.1.2.1)
    b1 = stiff bearing length on the flange (8.7.1.3). Fw also limits a pull (uplift) through the flange (8.7.8)."""
    f = fy * 1e3
    lam = math.sqrt(f / (math.pi ** 2 * E_STEEL)) * k_eff * d / (tw / math.sqrt(12))
    return (b1 + 5 * tf) * tw * f / GM0, chi(lam, ALPHA["c"]) * (b1 + D) * tw * f / GM0


END_POST_E = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)     # m, distance of the end-post stiffener from the member end


def end_post(d: float, tw: float, bf: float, fy: float, c: float, Rtf: float, Mtf: float):
    """8.5.2(b) double-stiffener end post at a knee / haunch: the member-end plate and a flat stiffener at
    distance e enclose a web strip e x tw, checked as a beam spanning between the flanges (8.5.3 forces):
        Rtf <= e tw fy / (sqrt3 gm0)                    shear
        Mtf <= (As e + tw e^2 / 4) fy / gm0             moment (As = core stiffener area, 8.7.1.2)
        Mtf / e <= Fqd                                  stiffener as a strut (8.7.1.5)
        Is >= 8.7.2.4 for the adjacent tension-field panel of width c - e, and c - e >= d (8.4.2.2(b))
    The end plate is assumed at least as strong as the stiffener. Lightest plate, then smallest e; or None."""
    f, eps = fy * 1e3, math.sqrt(250.0 / fy)
    for ts in STIFF_T:
        bs = min((bf - tw) / 2, 20 * ts * eps)
        bc = min(bs, 14 * ts * eps)
        As, Is, Fqd = ts * bc, ts * bc ** 3 / 3, stiffener_fqd(d, tw, ts, bs, fy)
        for e in END_POST_E:
            cp = c - e
            if cp < d:
                break
            Is_min = 0.75 * d * tw ** 3 if cp / d >= math.sqrt(2) else 1.5 * d ** 3 * tw ** 3 / cp ** 2
            Vd, Md = e * tw * f / (math.sqrt(3) * GM0), (As * e + tw * e * e / 4) * f / GM0
            if Rtf <= Vd and Mtf <= Md and Mtf / e <= Fqd and Is >= Is_min:
                return dict(ts=ts, bs=bs, e=e, d=d, Vd=Vd, Md=Md, Rtf=Rtf, Mtf=Mtf, F=Mtf / e, Fqd=Fqd, Is=Is,
                            Is_min=Is_min, ratio=max(Rtf / Vd, Mtf / Md, Mtf / e / Fqd),
                            mass=ts * bs * d * RHO_STEEL, weld_kN_per_mm=(tw * 1e3) ** 2 / (5 * bs * 1e3))
    return None


def capacities(D: float, p: Plates, fy: float, L_in: float, L_out: float, L_y: float, L_z: float,
               slender_web: bool = False, c=None, tension_field: bool = False) -> dict:
    """L_in / L_out: LTB lengths when the inner / outer flange is in compression (restraint spacing).
    L_y: out-of-plane flexural buckling length; L_z: in-plane (non-sway system) length.
    c: transverse stiffener spacing (m), None = stiffeners at supports only.
    tension_field: use 8.4.2.2(b) where c/d >= 1 (the caller guarantees anchorage, 8.5.1)."""
    f = fy * 1e3                                 # kN/m2
    eps = math.sqrt(250.0 / fy)
    s = i_section(D, p)
    d, A = D - p.tf_in - p.tf_out, s["A"]
    Iy = s["Iy"]
    rz, ry = math.sqrt(s["Iz"] / A), math.sqrt(Iy / A)

    def pd(L, r, curve, area=A):                 # 7.1.2 design compressive strength
        lam = math.sqrt(f / (math.pi ** 2 * E_STEEL)) * L / r
        return chi(lam, ALPHA[curve]) * area * f / GM0, lam

    Pdz, lam_z = pd(L_z, rz, CURVE_Z)
    Pdy, _ = pd(L_y, ry, CURVE_Y)
    Af_in, Af_out = p.bf_in * p.tf_in, p.bf_out * p.tf_out
    hf = D - (p.tf_in + p.tf_out) / 2                # lever arm between flange centroids

    # shear, 8.4.1 / 8.4.2 simple post-critical method
    k_v = kv(c, d)
    Vpl = d * p.tw * f / math.sqrt(3)                           # 8.4.1, Av = d tw (welded, 8.4.1.1)
    tb = f / math.sqrt(3)
    if d / p.tw > 67 * eps * math.sqrt(k_v / 5.35):             # 8.4.2.1
        tcr = k_v * math.pi ** 2 * E_STEEL / (12 * (1 - POISSON ** 2) * (d / p.tw) ** 2)
        lw = math.sqrt(f / (math.sqrt(3) * tcr))
        tb = (f / math.sqrt(3) if lw <= 0.8 else
              (1 - 0.8 * (lw - 0.8)) * f / math.sqrt(3) if lw < 1.2 else f / (math.sqrt(3) * lw ** 2))
    Vd = min(Vpl, d * p.tw * tb) / GM0                          # 8.4.2.2(a) simple post-critical

    def ltb(L, bb_Zp):                           # 8.2.2: Md = beta_b Zp chi_LT fy / gm0
        Mcr = math.sqrt(math.pi ** 2 * E_STEEL * Iy / L ** 2 *
                        (G_STEEL * s["It"] + math.pi ** 2 * E_STEEL * s["Iw"] / L ** 2))
        lam = math.sqrt(bb_Zp * f / Mcr)
        return (1.0 if lam <= 0.4 else chi(lam, ALPHA_LT)) * bb_Zp * f / GM0

    Ze = min(s["Ze_in"], s["Ze_out"])
    Zf = min(Af_in, Af_out) * hf                 # flange-only plastic modulus (8.2.1.1(a))
    Mfd = Zf * f / GM0
    cap = dict(eps=eps, f=f, A=A, d=d, tw=p.tw, Zp=s["Zp"], Ze=Ze, Nd=A * f / GM0, Pdz=Pdz, Pdy=Pdy,
             lam_z=lam_z, Vd=Vd, Mfd=Mfd, slender_web=slender_web, c=c, kv=k_v, tau_b=tb, Vpl=Vpl,
             tf=bool(tension_field and c is not None and c >= d and tb < f / math.sqrt(3)),   # 8.4.2.2(b): c/d >= 1
             bf_in=p.bf_in, tf_in=p.tf_in, bf_out=p.bf_out, tf_out=p.tf_out, Af_in=Af_in, Af_out=Af_out, hf=hf,
             flange_in=p.bf_in / 2 / p.tf_in, flange_out=p.bf_out / 2 / p.tf_out,
             ltb={cls: dict(inner=ltb(L_in, bz), outer=ltb(L_out, bz))
                  for cls, bz in (("plastic", s["Zp"]), ("elastic", Ze))})
    if slender_web:
        cap.update(web86=web_limit_86(d, c, eps, eps),                # 8.6.1.1, 8.6.1.2 (same grade)
                 Pdz_f=pd(L_z, rz, CURVE_Z, Af_in + Af_out)[0], Pdy_f=pd(L_y, ry, CURVE_Y, Af_in + Af_out)[0],
                 ltb_f=dict(inner=ltb(L_in, Zf), outer=ltb(L_out, Zf)))
    return cap


def v_tf(c: dict, N: float, M: float) -> float:
    """8.4.2.2(b) nominal tension field shear strength at a station (flange forces from N and M)."""
    d, tw, cs, f, tb = c["d"], c["tw"], c["c"], c["f"], c["tau_b"]
    phi = math.atan(d / cs)
    psi = 1.5 * tb * math.sin(2 * phi)
    fv = math.sqrt(max(f * f - 3 * tb * tb + psi * psi, 0.0)) - psi
    Fm, At = abs(M) / c["hf"], c["Af_in"] + c["Af_out"]
    s_sum = 0.0
    for Af, bf, tfl, comp in ((c["Af_in"], c["bf_in"], c["tf_in"], M >= 0),
                              (c["Af_out"], c["bf_out"], c["tf_out"], M < 0)):
        Nf = N * Af / At + (Fm if comp else -Fm)               # flange axial force, + = compression
        Mfr = 0.25 * bf * tfl ** 2 * f * max(0.0, 1 - (Nf / (Af * f / GM0)) ** 2)
        s_sum += min(2 / math.sin(phi) * math.sqrt(Mfr / (f * tw)), cs)
    wtf = max(0.0, d * math.cos(phi) - abs(cs - s_sum) * math.sin(phi))   # lower of the two readings
    return min(d * tw * tb + 0.9 * wtf * tw * fv * math.sin(phi), c["Vpl"])


def tf_weld(c: dict) -> float:
    """8.6.3.3: web-to-flange welds of a tension-field panel transfer the tension field stress, fv tw (kN/m)."""
    phi = math.atan(c["d"] / c["c"])
    psi = 1.5 * c["tau_b"] * math.sin(2 * phi)
    return (math.sqrt(max(c["f"] ** 2 - 3 * c["tau_b"] ** 2 + psi ** 2, 0.0)) - psi) * c["tw"]


def shear_capacity(c: dict, N: float, M: float) -> float:
    """Design shear strength: 8.4.2.2(a), or 8.4.2.2(b) where the panel may use tension field action."""
    return max(c["Vd"], v_tf(c, N, M) / GM0) if c["tf"] else c["Vd"]


def ratios(c: dict, N: float, V: float, M: float) -> dict:
    """Demand/capacity ratios at one station for one load combination."""
    eps, f = c["eps"], c["f"]
    flange = c["flange_in"] if M >= 0 else c["flange_out"]
    r2 = max(N, 0.0) / c["A"] / (f / GM0)                   # Table 2 web limits with axial
    web3, web2 = max(126 * eps / (1 + 2 * r2), 42 * eps), max(105 * eps / (1 + 1.5 * r2), 42 * eps)
    web = c["d"] / c["tw"] / web3
    if c["slender_web"] and web > 1:                        # slender web -> 8.2.1.1(a) flanges only
        return _flanges_only(c, N, V, M, flange)
    # with the slender-web option a web past the class 3 limit is designed by 8.2.1.1(a) instead,
    # so here only the flange class is a constraint
    slender = max(0.0 if c["slender_web"] else web, flange / (13.6 * eps))
    plastic = c["d"] / c["tw"] <= web2 and flange <= 9.4 * eps  # class 1/2 -> Zp, class 3 -> Ze
    Md = (c["Zp"] if plastic else c["Ze"]) * f / GM0
    V, Ma, Vd = abs(V), abs(M), shear_capacity(c, N, M)
    if V > 0.6 * Vd:                                        # 9.2.2 high shear
        beta = (2 * V / Vd - 1) ** 2
        Md = min(Md - beta * max(Md - c["Mfd"], 0.0), 1.2 * c["Ze"] * f / GM0)
    Mltb = min(c["ltb"]["plastic" if plastic else "elastic"]["inner" if M >= 0 else "outer"], Md)
    P = max(N, 0.0)
    nz = P / c["Pdz"]
    Kz = min(1 + (c["lam_z"] - 0.2) * nz, 1 + 0.8 * nz)
    return dict(
        cls=slender,                                        # > 1 means Class 4 (rejected)
        shear=V / Vd,
        section=abs(N) / c["Nd"] + Ma / Md,                 # 9.3.1 linear interaction
        buckling_y=P / c["Pdy"] + Ma / Mltb,                # 9.3.2.2 (a), KLT = 1
        buckling_z=nz + Kz * Ma / Mltb)                     # 9.3.2.2 (b), Cmz = 1


GM1 = 1.25                                                      # Table 5, ultimate stress
CURVE_TUBE = "b"                                                # Table 10: hollow section, cold formed (IS 4923)


def tube_capacities(s: dict, fy: float, fu: float, L_z: float, L_y: float) -> dict:
    """RHS / SHS (tubes_is4923.json: H = in-plane depth, B = width, t) with in-plane / out-of-plane buckling
    lengths L_z / L_y (7.2.4 for truss members). 6.2 / 6.3.1 tension (welded: An = Ag), 7.1.2 compression
    (curve b), 8.4.1.1 Av = A H / (B + H), 8.2.1.2 Md. Flat widths taken as B - 2t, H - 2t (corners ignored:
    larger b/t, conservative)."""
    f, eps, A = fy * 1e3, math.sqrt(250.0 / fy), s["A"]
    rz, ry = math.sqrt(s["Iz"] / A), math.sqrt(s["Iy"] / A)

    def pd(L, r):
        lam = math.sqrt(f / (math.pi ** 2 * E_STEEL)) * L / r
        return chi(lam, ALPHA[CURVE_TUBE]) * A * f / GM0, lam
    Pdz, lam_z = pd(L_z, rz)
    Pdy, _ = pd(L_y, ry)
    return dict(eps=eps, f=f, A=A, Nd=A * f / GM0, Td=min(A * f / GM0, 0.9 * A * fu * 1e3 / GM1), Pdz=Pdz, Pdy=Pdy,
                lam_z=lam_z, KLr=max(L_z / rz, L_y / ry), Vd=A * s["H"] / (s["B"] + s["H"]) * f / (math.sqrt(3) * GM0),
                Zp=s["Zpz"], Ze=s["Zez"], b_t=(s["B"] - 2 * s["t"]) / s["t"], d_t=(s["H"] - 2 * s["t"]) / s["t"])


def tube_ratios(c: dict, N: float, V: float, M: float) -> dict:
    """Tube demand/capacity at one station: Table 2 class (flange b/t 42 eps, web 126 eps / (1 + 2 r2) >= 42 eps;
    class 1/2 -> Zp: flange 33.5 eps, web 105 eps / (1 + 1.5 r2)), 9.2.2 high shear, 9.3.1 / 9.3.2.2 interaction
    (no LTB for closed sections: KLT = 1, Mdz = Md), 6.2 / 6.3 tension."""
    eps, f = c["eps"], c["f"]
    r2 = max(N, 0.0) / c["Nd"]
    web3, web2 = max(126 * eps / (1 + 2 * r2), 42 * eps), max(105 * eps / (1 + 1.5 * r2), 42 * eps)
    plastic = c["b_t"] <= 33.5 * eps and c["d_t"] <= web2
    Md = (c["Zp"] if plastic else c["Ze"]) * f / GM0
    V, Ma, P = abs(V), abs(M), max(N, 0.0)
    if V > 0.6 * c["Vd"]:                                  # ponytail: 9.2.2 with Mfd = 0 (conservative for tubes)
        Md *= 1 - (2 * V / c["Vd"] - 1) ** 2
    nz = P / c["Pdz"]
    Kz = min(1 + (c["lam_z"] - 0.2) * nz, 1 + 0.8 * nz)
    return dict(cls=max(c["b_t"] / (42 * eps), c["d_t"] / web3), shear=V / c["Vd"], tension=max(-N, 0.0) / c["Td"],
                section=abs(N) / c["Nd"] + Ma / Md, buckling_y=P / c["Pdy"] + Ma / Md, buckling_z=nz + Kz * Ma / Md)


CURVE_ANGLE = "c"                                               # Table 10: angle / channel / T -> c


def double_angle(a: dict, tg: float) -> dict:
    """Two equal angles back to back on a gusset tg thick (vertical legs on the gusset, gusset in the truss
    plane) from single-angle data (angles_is808.json): in-plane Iz = 2 Izz, out-of-plane
    Iy = 2 [Iyy + A (c + tg/2)^2], Zez = 2 Zz (tip of the vertical leg), H = leg (in plane), B = 2 b + tg."""
    b, t, A = a["b"], a["t"], a["A"]
    return dict(name="2-" + a["name"], shape="DA", b=b, t=t, H=b, B=2 * b + tg, tg=tg, A=2 * A, Iz=2 * a["Izz"],
                Iy=2 * (a["Iyy"] + A * (a["c"] + tg / 2) ** 2), Zez=2 * a["Zz"], Zpz=2 * a["Zpz"], It=2 * a["It"],
                w=2 * a["w"])


def angle_capacities(s: dict, fy: float, fu: float, L_z: float, L_y: float) -> dict:
    """Double equal angles (double_angle()): 7.1.2 compression curve c (7.5.2 / 7.5.3: axial, lengths from the
    caller), 6.2 yielding and 6.3.3 / 6.3.4 rupture per angle Tdn = 0.9 Anc fu/gm1 + beta Ago fy/gm0 with welded
    ends (Anc = gross connected leg) and beta = 0.7, its lower bound (needs no connection length; conservative),
    Md = Ze fy/gm0 (class 3, in-plane bending about the leg-parallel axis: no LTB), Av = 2 b t (legs parallel
    to the shear - 8.4.1.1 gives no angle rule)."""
    f, eps, A, b, t = fy * 1e3, math.sqrt(250.0 / fy), s["A"], s["b"], s["t"]
    rz, ry = math.sqrt(s["Iz"] / A), math.sqrt(s["Iy"] / A)

    def pd(L, r):
        lam = math.sqrt(f / (math.pi ** 2 * E_STEEL)) * L / r
        return chi(lam, ALPHA[CURVE_ANGLE]) * A * f / GM0, lam
    Pdz, lam_z = pd(L_z, rz)
    Pdy, _ = pd(L_y, ry)
    leg = (b - t / 2) * t
    Tdn = 2 * (0.9 * leg * fu * 1e3 / GM1 + 0.7 * leg * f / GM0)
    return dict(eps=eps, f=f, A=A, Nd=A * f / GM0, Td=min(A * f / GM0, Tdn), Pdz=Pdz, Pdy=Pdy, lam_z=lam_z,
                KLr=max(L_z / rz, L_y / ry), Vd=2 * b * t * f / (math.sqrt(3) * GM0), Ze=s["Zez"], b_t=b / t)


def angle_ratios(c: dict, N: float, V: float, M: float) -> dict:
    """Double-angle demand/capacity: Table 2 (angles, components separated, axial: b/t, d/t <= 15.7 eps and
    (b + d)/t <= 25 eps; bending only: 15.7 eps), 9.3.1 / 9.3.2.2 with Mdz = Ze fy / gm0, 6.2 / 6.3 tension."""
    eps, P, Ma, V = c["eps"], max(N, 0.0), abs(M), abs(V)
    Md = c["Ze"] * c["f"] / GM0
    if V > 0.6 * c["Vd"]:                                  # ponytail: 9.2.2 with Mfd = 0
        Md *= 1 - (2 * V / c["Vd"] - 1) ** 2
    nz = P / c["Pdz"]
    Kz = min(1 + (c["lam_z"] - 0.2) * nz, 1 + 0.8 * nz)
    cls = max(c["b_t"] / (15.7 * eps), 2 * c["b_t"] / (25 * eps) if P > 0 else 0.0)
    return dict(cls=cls, shear=V / c["Vd"], tension=max(-N, 0.0) / c["Td"], section=abs(N) / c["Nd"] + Ma / Md,
                buckling_y=P / c["Pdy"] + Ma / Md, buckling_z=nz + Kz * Ma / Md)


def _flanges_only(c: dict, N: float, V: float, M: float, flange: float) -> dict:
    """8.2.1.1(a): M and N taken by the flanges, web carries shear only (no 9.2.2 interaction)."""
    eps, f = c["eps"], c["f"]
    comp_in = M >= 0                                        # inner flange in compression
    Ac, At = (c["Af_in"], c["Af_out"]) if comp_in else (c["Af_out"], c["Af_in"])
    Fm = abs(M) / c["hf"]                                   # flange couple
    Fc, Ft = N * Ac / (Ac + At) + Fm, N * At / (Ac + At) - Fm  # + = compression
    fd = f / GM0
    P, Ma = max(N, 0.0), abs(M)
    Mltb = min(c["ltb_f"]["inner" if comp_in else "outer"], c["Mfd"])
    nz = P / c["Pdz_f"]
    Kz = min(1 + (c["lam_z"] - 0.2) * nz, 1 + 0.8 * nz)
    return dict(
        cls=max(c["d"] / c["tw"] / c["web86"], flange / (13.6 * eps)),   # 8.6.1 web, Table 2 flange
        shear=abs(V) / shear_capacity(c, N, M),
        section=max(abs(Fc) / (Ac * fd), abs(Ft) / (At * fd)),
        buckling_y=P / c["Pdy_f"] + Ma / Mltb,
        buckling_z=nz + Kz * Ma / Mltb)


if __name__ == "__main__":
    # Hand check: 600 x 6 web, 200 x 12 flanges, fy 250, short restraint -> plastic, no LTB
    p = Plates(0.006, 0.2, 0.012, 0.2, 0.012)
    c = capacities(0.6, p, 250.0, 0.5, 0.5, 0.5, 0.5)
    Zp = 2 * 0.2 * 0.012 * (0.6 - 0.012) / 2 + 0.006 * (0.6 - 0.024) ** 2 / 4
    assert abs(c["Zp"] - Zp) / Zp < 1e-9
    Md = Zp * 250e3 / 1.1
    r = ratios(c, 0.0, 0.0, Md)
    assert abs(r["section"] - 1) < 1e-9 and abs(r["buckling_y"] - 1) < 1e-9, r
    # web d/tw = 96 (fy 250, eps 1): 84 < 96 <= 105 -> class 2, so Zp is used and the class ratio passes
    assert r["cls"] < 1
    # shear: d/tw = 96 > 67 -> web buckling governs; tau_b by hand
    tcr = 5.35 * math.pi ** 2 * 2e8 / (12 * 0.91 * 96 ** 2)
    lw = math.sqrt(250e3 / (math.sqrt(3) * tcr))
    tb = (1 - 0.8 * (lw - 0.8)) * 250e3 / math.sqrt(3) if lw < 1.2 else 250e3 / (math.sqrt(3) * lw ** 2)
    assert abs(c["Vd"] - min(0.576 * 0.006 * 250e3 / math.sqrt(3), 0.576 * 0.006 * tb) / 1.1) < 1e-6   # Av = d tw
    # long unrestrained inner flange -> LTB reduces capacity; axial adds
    c2 = capacities(0.6, p, 250.0, 6.0, 1.5, 6.0, 12.0)
    assert c2["ltb"]["plastic"]["inner"] < c2["ltb"]["plastic"]["outer"] < Md + 1e-9
    r2 = ratios(c2, 100.0, 0.0, 0.5 * Md)
    assert r2["buckling_y"] > r2["section"]
    # slender web -> class ratio > 1
    assert ratios(capacities(1.2, Plates(0.005, 0.2, 0.012, 0.2, 0.012), 345, 1, 1, 1, 1), 0, 0, 1)["cls"] > 1
    # slender-web option: 1000 x 5 web (d/tw 195 > 126 eps) with 250 x 14 flanges, fy 250
    ps = Plates(0.005, 0.25, 0.014, 0.25, 0.014)
    cs = capacities(1.0, ps, 250.0, 0.5, 0.5, 0.5, 0.5, slender_web=True)
    assert ratios(capacities(1.0, ps, 250.0, 0.5, 0.5, 0.5, 0.5), 0, 0, 1)["cls"] > 1   # rejected by default
    Mf = 0.25 * 0.014 * (1.0 - 0.014) * 250e3 / 1.1                # flange couple, hand calc
    rs = ratios(cs, 0.0, 0.0, Mf)
    assert rs["cls"] < 1 and abs(rs["section"] - 1) < 1e-9 and abs(rs["buckling_y"] - 1) < 1e-9, rs
    rs2 = ratios(cs, 200.0, 0.0, 0.5 * Mf)                         # axial shared by flanges by area
    assert abs(rs2["section"] - (100 / (0.25 * 0.014 * 250e3 / 1.1) + 0.5)) < 1e-9, rs2
    d_lim = min(200, 345)                                          # 8.6.1: d/tw limit at fy 250
    thin = Plates(0.972 / (d_lim + 5), 0.25, 0.014, 0.25, 0.014)   # just past the 8.6.1.1 limit
    assert ratios(capacities(1.0, thin, 250.0, 0.5, 0.5, 0.5, 0.5, True), 0, 0, 1)["cls"] > 1
    # transverse stiffeners: kv, 8.6.1 limits (fy 345: eps 0.8513), stiffener stiffness
    assert abs(kv(1.0, 1.0) - 9.35) < 1e-12 and abs(kv(0.5, 1.0) - 25.4) < 1e-12 and kv(None, 1.0) == 5.35
    e = math.sqrt(250 / 345)
    for cd, lim in ((None, 200 * e), (4.0, 200 * e), (2.0, 200 * e), (0.8, 200 * e / 0.8), (0.5, 270 * e)):
        assert abs(web_limit_86(1.0, cd, e, e) - lim) < 1e-9, (cd, web_limit_86(1.0, cd, e, e), lim)
    assert abs(web_limit_86(1.0, 1.2, e, e) - min(200 * e, 345 * e)) < 1e-9          # c < 1.5d: 345 eps_f
    thin = Plates(0.005, 0.2, 0.012, 0.2, 0.012)
    v0 = capacities(0.9, thin, 345, 1, 1, 1, 1)["Vd"]
    v1 = capacities(0.9, thin, 345, 1, 1, 1, 1, c=0.6)["Vd"]
    assert v1 > 1.5 * v0, (v0, v1)                                                      # c/d 0.68 -> kv 15.6
    st = stiffener(0.876, 0.005, 0.2, 345, 0.6)
    assert st and st["Is"] >= 1.5 * 0.876 ** 3 * 0.005 ** 3 / 0.6 ** 2 and st["bs"] <= 20 * st["ts"] * e + 1e-12
    near = capacities(0.45, Plates(0.004, 0.2, 0.012, 0.2, 0.012), 345, 1, 1, 1, 1, slender_web=True)
    assert ratios(near, 0, 0, 1.0)["cls"] < 0.9                     # d/tw 106.5 ~ class 3 limit: not a constraint
    # tension field: 900 x 4 web, c = 1.34 (c/d 1.45), 200 x 12 flanges, fy 345
    ptf = Plates(0.004, 0.2, 0.012, 0.2, 0.012)
    ca = capacities(0.924, ptf, 345, 1, 1, 1, 1, True, 1.34)
    ct = capacities(0.924, ptf, 345, 1, 1, 1, 1, True, 1.34, tension_field=True)
    assert ct["tf"] and not ca["tf"] and not capacities(0.924, ptf, 345, 1, 1, 1, 1, True, 0.67, True)["tf"]
    v0, v1 = v_tf(ct, 0.0, 0.0), v_tf(ct, 0.0, 300.0)
    assert ca["Vd"] * 1.1 < v1 < v0 <= ct["Vpl"] + 1e-9, (ca["Vd"] * 1.1, v1, v0)   # flange force lowers Vtf
    assert shear_capacity(ct, 0, 0) > shear_capacity(ca, 0, 0)
    fq6, fq12 = stiffener_fqd(0.9, 0.004, 0.006, 0.098, 345), stiffener_fqd(0.9, 0.004, 0.012, 0.098, 345)
    assert 0 < fq6 < fq12 and stiffener(0.9, 0.004, 0.2, 345, 1.34, F=fq6 * 1.01)["ts"] > 0.006
    # end post 8.5.2(b): hand check of the chosen plate / e, and bigger forces never give a lighter post
    ep = end_post(0.9, 0.006, 0.2, 345, 1.4, 200.0, 40.0)
    f3 = 345e3
    ac = ep["ts"] * min(ep["bs"], 14 * ep["ts"] * math.sqrt(250 / 345))
    assert abs(ep["Md"] - (ac * ep["e"] + 0.006 * ep["e"] ** 2 / 4) * f3 / 1.1) < 1e-9 and ep["ratio"] <= 1
    ep2 = end_post(0.9, 0.006, 0.2, 345, 1.4, 400.0, 80.0)
    assert ep2 and (ep2["ts"], ep2["e"]) >= (ep["ts"], ep["e"]) and end_post(0.9, 0.006, 0.2, 345, 0.95, 200, 40) is None
    # web under a purlin: 900 x 4 web, 10 mm flange, b1 20 mm, fy 345 (hand calc)
    Fw, Fc = web_point_load(0.92, 0.9, 0.004, 0.010, 345, 0.020, 1.0)
    assert abs(Fw - (0.020 + 0.050) * 0.004 * 345e3 / 1.1) < 1e-9
    lam = math.sqrt(345e3 / (math.pi ** 2 * 2e8)) * 0.9 / (0.004 / math.sqrt(12))
    assert abs(Fc - chi(lam, 0.49) * (0.020 + 0.92) * 0.004 * 345e3 / 1.1) < 1e-9 and 8 < Fc < 13, Fc
    assert web_point_load(0.92, 0.9, 0.004, 0.010, 345, 0.020, 0.7)[1] > 1.8 * Fc          # KL 0.7 d vs d
    # tube: SHS 100x100x4 (IS 4923 DB: A 14.95 cm2, I 226.9 cm4 - STAAD printed AXX 14.95), fy 310, fu 450
    import json
    from pathlib import Path
    shs = next(t for t in json.loads((Path(__file__).parent / "tubes_is4923.json").read_text())["tubes"]
               if t["name"] == "SHS 100x100x4")
    ct = tube_capacities(shs, 310, 450, 2.0, 2.0)
    assert abs(ct["Nd"] - 14.95e-4 * 310e3 / 1.1) < 0.5 and abs(ct["Td"] - ct["Nd"]) < 1e-9       # yield < rupture
    lam = math.sqrt(310e3 / (math.pi ** 2 * 2e8)) * 2.0 / math.sqrt(shs["Iz"] / shs["A"])
    assert abs(ct["Pdz"] - chi(lam, 0.34) * shs["A"] * 310e3 / 1.1) < 1e-9                      # curve b
    rt = tube_ratios(ct, -ct["Td"], 0, 0)
    assert abs(rt["tension"] - 1) < 1e-9 and rt["cls"] < 1                                     # b/t 23 < 42 eps
    assert abs(tube_ratios(ct, ct["Pdz"], 0, 0)["buckling_z"] - 1) < 1e-9
    # double angle 2-ISEA 65X65X8 on an 8 mm gusset (IS 808 DB: A 9.85 cm2, I 38.4 cm4, c 19.1 mm, Zz 8.36 cm3)
    a65 = next(a for a in json.loads((Path(__file__).parent / "angles_is808.json").read_text())["angles"]
               if a["name"] == "ISEA 65X65X8")
    da = double_angle(a65, 0.008)
    assert abs(da["A"] - 19.7e-4) < 1e-9 and abs(da["Iz"] - 76.8e-8) < 1e-12
    assert abs(da["Iy"] - 2 * (38.4e-8 + 9.85e-4 * (0.0191 + 0.004) ** 2)) < 1e-12        # parallel axes
    cda = angle_capacities(da, 250, 410, 1.5, 1.5)
    leg = (0.065 - 0.004) * 0.008
    assert abs(cda["Td"] - min(19.7e-4 * 250e3 / 1.1, 2 * (0.9 * leg * 410e3 / 1.25 + 0.7 * leg * 250e3 / 1.1))) < 1e-9
    lam = math.sqrt(250e3 / (math.pi ** 2 * 2e8)) * 1.5 / math.sqrt(da["Iz"] / da["A"])
    assert abs(cda["Pdz"] - chi(lam, 0.49) * da["A"] * 250e3 / 1.1) < 1e-9                  # curve c
    ra = angle_ratios(cda, 10.0, 0, 0)
    assert abs(ra["cls"] - 2 * 65 / 8 / 25) < 1e-9 and ra["cls"] < 1                          # (b+d)/t = 16.25 < 25
    print("is800 self-check OK", {k: round(v, 3) for k, v in r2.items()})
