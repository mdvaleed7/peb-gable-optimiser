"""
peb_frame_model.py
==================
Parametric 2D multi-span pitched-roof PEB portal frame  ->  STAAD.Pro input files (.std)

Basis     : IS 800:2007 (LSM) frame analysis inputs; loads to IS 875 (Parts 1-3).
Model     : STAAD PLANE (XY), METER-KN, tapered welded I-sections (TAPERED f1..f7).
Outputs   : <name>_ULS_PDELTA.std  primaries + REPEAT LOAD ULS combos  -> PDELTA analysis
            <name>_SLS.std         primaries + unit virtual cases + SLS combos -> linear
            <name>_BUCKLING.std    gravity primaries + gravity REPEAT LOADs -> buckling
            station_map.json       member <-> station <-> knot map + sign conventions
            model_summary.txt      geometry, sections, weights, load totals, benchmarks

WHY .std TEXT (and OpenSTAAD only to open / analyse / read):
  STAAD requires f1 (start depth) >= f3 (end depth) for TAPERED members (TR.20.3).
  Every sub-member is therefore oriented deeper-end-first, and the orientation must
  flip whenever the optimiser reverses a taper. Rewriting the text does that cleanly;
  over COM you would have to delete and recreate members. Node coordinates and member
  IDs never change, so results map to the same stations every cycle.

CONVENTIONS (STAAD TR G.4.3 / TR.20.3; verified against Bentley docs):
  * Global X right, Y up; frame in XY; BETA = 0 on every member.
  * Non-vertical member: local y has a +global-Y component (points "up").
    Vertical member: local z = +global Z, so local y = Z x x_local.
  * TAPERED f1..f7 = start depth, web t, end depth, top-flange b, top-flange t,
    bottom-flange b, bottom-flange t; "top" = local +y side.
  * Member end forces = actions ON the member, local axes, [FX FY FZ MX MY MZ].
    Axial compression N = FX(start) = -FX(end).
    Moment with compression on the local +y face: M(start) = -MZ1, M(end) = +MZ2.
    Stations store M_ic = moment putting the INNER flange in compression (+).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------- constants
RHO_STEEL = 7850.0                      # kg/m3
GAMMA_STEEL = RHO_STEEL * 9.80665 / 1e3  # kN/m3  (76.98)
E_STEEL = 2.0e8                          # kN/m2  (IS 800: 2.0e5 N/mm2)
POISSON = 0.3                            # -> G = 0.769e5 N/mm2 as IS 800
ALPHA_T = 12e-6                          # /degC
TOL = 1e-9


# ============================================================================= INPUT TYPES
@dataclass
class Plates:
    """Plates of one tapered segment (m). 'out' = outer flange, 'in' = inner flange.
    Rafter: outer = top (sheeting side). Exterior column: outer = cladding side."""
    tw: float
    bf_out: float
    tf_out: float
    bf_in: float
    tf_in: float


@dataclass
class Profile:
    """Depth profile of one member line. knots = [(s/L from line start, overall depth D m)].
    plates[k] applies between knot k and k+1. Knots snap to the nearest grid station."""
    knots: List[Tuple[float, float]]
    plates: List[Plates]

    def check(self, name: str) -> None:
        if len(self.knots) < 2 or len(self.plates) != len(self.knots) - 1:
            raise ValueError(f"{name}: need >=2 knots and len(plates) == len(knots)-1")
        s = [k[0] for k in self.knots]
        if abs(s[0]) > TOL or abs(s[-1] - 1.0) > TOL or any(b <= a for a, b in zip(s, s[1:])):
            raise ValueError(f"{name}: knot positions must run 0 -> 1, strictly increasing")
        for p in self.plates:
            if min(p.tw, p.bf_out, p.tf_out, p.bf_in, p.tf_in) <= 0:
                raise ValueError(f"{name}: all plate dimensions must be > 0")
        for s_, d in self.knots:
            if d <= 0:
                raise ValueError(f"{name}: depth must be > 0")


def mirror(p: Profile) -> Profile:
    """Profile given support->ridge, returned ridge->support (for right-hand halves)."""
    return Profile([(1.0 - s, d) for s, d in reversed(p.knots)], list(reversed(p.plates)))


@dataclass
class WindCase:
    """Net pressure on a surface = (Cpe - Cpi) * pd, positive = towards the surface.
    cpe_roof: one value per roof piece, left to right (see FrameModel.pieces)."""
    title: str
    cpi: float
    cpe_wall_left: float
    cpe_wall_right: float
    cpe_roof: List[float]


@dataclass
class Loads:
    q_sdl_slope: float = 0.15      # kN/m2 on slope: sheeting + purlins + insulation (IS 875-1 / supplier)
    q_coll_plan: float = 0.10      # kN/m2 plan: services, lighting (project brief)
    q_ll_plan: float = 0.75        # kN/m2 plan: roof imposed (IS 875-2 Table 2 - VERIFY for slope/access)
    q_wall: float = 0.10           # kN/m2 wall cladding + girts -> vertical on exterior columns
    sw_factor: float = 1.0         # self-weight multiplier (connection/stiffener allowance)
    pd_wind: float = 0.0           # kN/m2, design wind pressure (design_wind_pressure())
    wind_cases: List[WindCase] = field(default_factory=list)
    cpe_verified: bool = False     # set True only after Cpe are taken from IS 875-3 multispan tables


@dataclass
class Crane:
    """EOT crane in one span, running on gantry girders seated on column brackets (IS 875-2 cl. 6.3, 6.4).
    Loads per frame: wheel loads x reaction influence of the gantry girders (simply supported, span = bay)."""
    span: int                    # 0-based span index
    bracket_level: float         # m above the base: gantry seat on the bracket
    e: float                     # m, rail centre line from the column centre line (= bracket length)
    capacity: float              # kN, lifted load (SWL)
    bridge: float                # kN, crane bridge incl. end carriages
    crab: float                  # kN, crab / trolley
    a_min: float                 # m, minimum hook approach to a rail
    wheel_base: float            # m, wheel spacing of one end carriage
    wheels_per_rail: int = 2
    gantry: float = 0.0          # kN/m, gantry girder + rail self-weight (dead load)
    gantry_depth: float = 0.0    # m, bracket seat to rail top (surge acts at rail level, 6.3)
    impact: float = 0.25         # vertical impact on max wheel loads, 6.3(a): 0.25 class III/IV, 0.10 class I/II columns
    surge: float = 0.05          # transverse surge fraction of (crab + lifted), 6.3(c): 0.10 rigid mast, 0.05 others


@dataclass
class Truss:
    """Portal truss in place of the tapered rafters (multigable or single-ridge roofs). Top chord = the roof line
    (the rafter pieces, so every roof load applies unchanged); bottom chord y = eave - h0 + bs (y_roof - eave):
    h0 below the eaves at the eave columns, rising bs x the roof rise (0 flat, 1 parallel chords); a vertical at
    every purlin station and at a ridge between columns; one diagonal per panel (pratt: falls toward the ridge /
    the middle of a column-to-column piece, howe: rises). Verticals and diagonals pin-ended (MEMBER TRUSS),
    chords continuous, the column continues between the chords (knee).
    sections[side][role]: section dict - IS 4923 tube (shape SHS / RHS) or double angle (shape DA) - per side
    'ext' (piece touching an eave column) / 'int' and role 'top', 'bot', 'vert', 'diag'."""
    h0: float
    sections: Dict[str, Dict[str, dict]]
    bs: float = 0.0
    web: str = "pratt"


def crane_reactions(cr: Crane, span: float, bay: float) -> Dict[str, float]:
    """Per-frame bracket loads (kN). Wheels of one end carriage placed at the column and at the wheel base on
    one side: reaction influence il = sum(1 - i w / bay). Impact on the max wheel loads only (6.3(a));
    surge shared by the wheels on one rail (6.3(c)); G = gantry dead load per bracket."""
    Lc, nw = span - 2 * cr.e, cr.wheels_per_rail
    lift = cr.crab + cr.capacity
    if not 0 < cr.a_min < Lc / 2:
        raise ValueError(f"crane a_min {cr.a_min} m must be between 0 and half the rail gauge {Lc / 2:.2f} m")
    pmax = cr.bridge / (2 * nw) + lift * (Lc - cr.a_min) / Lc / nw
    pmin = cr.bridge / (2 * nw) + lift * cr.a_min / Lc / nw
    il = sum(max(0.0, 1 - i * cr.wheel_base / bay) for i in range(nw))
    return dict(Lc=Lc, Pmax=pmax, Pmin=pmin, il=il, Rmax=pmax * (1 + cr.impact) * il, Rmin=pmin * il,
                H=cr.surge * lift / nw * il, G=cr.gantry * bay)


# Load factors: IS 800:2007 Table 4 (LSM) -- VERIFY against your copy before issue.
# ll: "patterns" = every live-load pattern, "ALL" = all spans loaded only.
ULS_RULES = [
    dict(tag="1.5(D+L)",       D=1.5, C=1.5, L=1.5, W=0.0, notional=True,  ll="patterns"),
    dict(tag="1.2(D+L)+0.6W",  D=1.2, C=1.2, L=1.2, W=0.6, notional=False, ll="ALL"),
    dict(tag="1.2(D+L+W)",     D=1.2, C=1.2, L=1.2, W=1.2, notional=False, ll="ALL"),
    dict(tag="1.5(D+W)",       D=1.5, C=1.5, L=0.0, W=1.5, notional=False, ll=None),
    dict(tag="0.9D+1.5W",      D=0.9, C=0.0, L=0.0, W=1.5, notional=False, ll=None),  # collateral excluded
    # crane load CL = second live load (Table 4 note 1): each of L and CL taken as leading in turn
    dict(tag="1.5(D+L)+1.05CL",      D=1.5, C=1.5, L=1.5,  CL=1.05, W=0.0, ll="ALL"),
    dict(tag="1.5(D+CL)+1.05L",      D=1.5, C=1.5, L=1.05, CL=1.5,  W=0.0, ll="ALL"),
    dict(tag="1.2(D+L)+1.05CL+0.6W", D=1.2, C=1.2, L=1.2,  CL=1.05, W=0.6, ll="ALL"),
    dict(tag="1.2(D+CL)+1.05L+0.6W", D=1.2, C=1.2, L=1.05, CL=1.2,  W=0.6, ll="ALL"),
    dict(tag="1.2(D+L+W)+0.53CL",    D=1.2, C=1.2, L=1.2,  CL=0.53, W=1.2, ll="ALL"),
    dict(tag="1.2(D+CL+W)+0.53L",    D=1.2, C=1.2, L=0.53, CL=1.2,  W=1.2, ll="ALL"),
]
SLS_RULES = [
    dict(tag="L",           D=0.0, C=0.0, L=1.0, W=0.0, ll="patterns"),  # Table 6 'live load'
    dict(tag="W",           D=0.0, C=0.0, L=0.0, W=1.0, ll=None),        # Table 6 'wind load'
    dict(tag="D+L",         D=1.0, C=1.0, L=1.0, W=0.0, ll="ALL"),
    dict(tag="D+0.8L+0.8W", D=1.0, C=1.0, L=0.8, W=0.8, ll="ALL"),
    dict(tag="CL",          D=0.0, C=0.0, L=0.0, CL=1.0, W=0.0, ll=None),   # Table 6 crane: rail drift / spread
    dict(tag="0.8(CL+W)",   D=0.0, C=0.0, L=0.0, CL=0.8, W=0.8, ll=None),   # Table 6 'crane + wind'
]
NOTIONAL_FRACTION = 0.005   # IS 800 Cl. 4.3.6: 0.5 % of factored gravity, gravity combos only


@dataclass
class FrameInput:
    name: str
    spans: List[float]                 # column line to column line (m)
    eave_height: float                 # analytical node height at exterior columns (m)
    slope: float                       # rise / run
    roof: str = "multigable"           # "multigable" (ridge per span) | "single_ridge" (ridge at centre)
    bay: float = 7.5                   # frame spacing (m) -> tributary width
    purlin_spacing: float = 1.5        # max purlin spacing along slope (m) -> roof stations
    stations_per_purlin: int = 1       # extra stations between purlins (braces only at purlins)
    girt_spacing: float = 1.5          # max station spacing on exterior columns (m)
    interior_col_subdiv: int = 4       # sub-members per interior column (lets eigen-buckling see it)
    base_ext: str = "PINNED"           # PINNED | FIXED
    base_int: str = "PINNED"
    interior_columns: str = "leaning"  # "leaning" (MZ released at top) | "rigid"
    fy: float = 345.0                  # MPa (recorded; design is external)
    buckling_eigen: bool = False       # True needs STAAD Advanced Analysis licence
    rafter_SR_ext: Optional[Profile] = None   # support->ridge template, exterior support
    rafter_SR_int: Optional[Profile] = None   # support->ridge template, interior support
    rafter_SS: Optional[Profile] = None       # support->support template (single_ridge)
    rafter_overrides: Dict[int, Profile] = field(default_factory=dict)  # by piece index
    col_ext: Optional[Profile] = None         # base->top
    col_int: Optional[Profile] = None         # base->top
    bracket: Optional[Profile] = None         # crane bracket, column -> tip
    cranes: List[Crane] = field(default_factory=list)
    truss: Optional[Truss] = None             # portal truss scheme instead of tapered rafters
    loads: Loads = field(default_factory=Loads)


# ============================================================================= HELPERS
def design_wind_pressure(Vb, k1, k2, k3, k4, Kd, Ka, Kc):
    """IS 875 (Part 3):2015. Vz = Vb k1 k2 k3 k4 (Cl. 6.3); pz = 0.6 Vz^2 (Cl. 7.2);
    pd = Kd Ka Kc pz, not less than 0.7 pz (Cl. 7.2).  Returns (pd, pz, Vz) in kN/m2, m/s."""
    Vz = Vb * k1 * k2 * k3 * k4
    pz = 0.6 * Vz ** 2 / 1000.0
    return max(Kd * Ka * Kc * pz, 0.7 * pz), pz, Vz


def i_section(D: float, p: Plates) -> Dict[str, float]:
    """Welded I properties (m-units). Inner flange at y=0, outer flange at y=D."""
    ti, to, tw = p.tf_in, p.tf_out, p.tw
    hw = D - ti - to
    if hw <= 0:
        raise ValueError(f"web height <= 0 for D={D}")
    parts = [  # (area, centroid y, own I, width, y0, y1)
        (p.bf_in * ti, ti / 2, p.bf_in * ti ** 3 / 12, p.bf_in, 0.0, ti),
        (tw * hw, ti + hw / 2, tw * hw ** 3 / 12, tw, ti, ti + hw),
        (p.bf_out * to, D - to / 2, p.bf_out * to ** 3 / 12, p.bf_out, D - to, D),
    ]
    A = sum(a for a, *_ in parts)
    yc = sum(a * y for a, y, *_ in parts) / A
    Iz = sum(i + a * (y - yc) ** 2 for a, y, i, *_ in parts)
    # plastic neutral axis (equal area) and Zp
    half, acc, ypna = A / 2, 0.0, 0.0
    for a, y, i, b, y0, y1 in parts:
        if acc + a >= half:
            ypna = y0 + (half - acc) / b
            break
        acc += a
    Zp = 0.0
    for a, y, i, b, y0, y1 in parts:
        lo, hi = y0 - ypna, y1 - ypna
        if lo >= 0 or hi <= 0:
            Zp += b * abs(hi ** 2 - lo ** 2) / 2
        else:
            Zp += b * (hi ** 2 + lo ** 2) / 2
    Ifi, Ifo = p.tf_in * p.bf_in ** 3 / 12, p.tf_out * p.bf_out ** 3 / 12
    hf = D - (ti + to) / 2
    return dict(A=A, yc=yc, Iz=Iz, Ze_out=Iz / (D - yc), Ze_in=Iz / yc, Zp=Zp,
                Iy=Ifi + Ifo + hw * tw ** 3 / 12,
                It=(p.bf_in * ti ** 3 + p.bf_out * to ** 3 + hw * tw ** 3) / 3,
                Iw=Ifi * Ifo / (Ifi + Ifo) * hf ** 2)


def _ranges(ids: List[int]) -> List[str]:
    """[1,2,3,7] -> ['1 TO 3', '7'] (one run per line keeps STAAD lines short)."""
    ids, out, i = sorted(ids), [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(f"{ids[i]} TO {ids[j]}" if j > i else f"{ids[i]}")
        i = j + 1
    return out


def _wrap_pairs(pairs: List[Tuple[int, float]], per_line: int = 5) -> List[str]:
    """REPEAT LOAD / LOAD COMBINATION data with '-' continuation (STAAD line width)."""
    toks = [f"{lc} {f:g}" for lc, f in pairs]
    lines = [" ".join(toks[i:i + per_line]) for i in range(0, len(toks), per_line)]
    return [ln + (" -" if k < len(lines) - 1 else "") for k, ln in enumerate(lines)]


def ll_patterns(n: int) -> Dict[str, Tuple[int, ...]]:
    """Roof live-load patterns for a continuous multi-span rafter."""
    pats: Dict[str, Tuple[int, ...]] = {"ALL": tuple(range(n))}
    if n > 1:
        pats["ALT-A"] = tuple(range(0, n, 2))          # max sagging, spans 1,3,5..
        pats["ALT-B"] = tuple(range(1, n, 2))          # max sagging, spans 2,4,6..
        for i in range(1, n):                          # max hogging at interior line i
            s = {i - 1, i} | set(range(i - 3, -1, -2)) | set(range(i + 2, n, 2))
            pats[f"ADJ-C{i}"] = tuple(sorted(s))
    uniq: Dict[str, Tuple[int, ...]] = {}
    for k, v in pats.items():
        if v and v not in uniq.values():
            uniq[k] = v
    return uniq


def end_forces_to_stations(m: dict, f_start: List[float], f_end: List[float]):
    """Local member end forces (actions on member, [FX FY FZ MX MY MZ]) -> station values.
    Returns [(grid_j, N_comp, V_local, M_ic), ...] for the member's two grid stations."""
    n_s, n_e = f_start[0], -f_end[0]
    v_s, v_e = f_start[1], -f_end[1]
    m_s, m_e = -f_start[5], f_end[5]            # compression on local +y face
    sg = m["sign_ic"]
    j_start, j_end = (m["j"] + 1, m["j"]) if m["reversed"] else (m["j"], m["j"] + 1)
    return [(j_start, n_s, v_s, sg * m_s), (j_end, n_e, v_e, sg * m_e)]


# ============================================================================= MODEL
class FrameModel:
    def __init__(self, fi: FrameInput):
        self.fi = fi
        self.warnings: List[str] = []
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self._node_key: Dict[Tuple[float, float], int] = {}
        self.lines: List[dict] = []
        self.members: List[dict] = []
        self.supports: Dict[int, str] = {}
        self.releases: List[Tuple[int, str]] = []
        self._build()

    # ------------------------------------------------------------------ geometry
    def _nid(self, x: float, y: float) -> int:
        key = (round(x, 6), round(y, 6))
        if key not in self._node_key:
            nid = len(self.nodes) + 1
            self._node_key[key] = nid
            self.nodes[nid] = (x, y)
        return self._node_key[key]

    def col_x(self) -> List[float]:
        xs = [0.0]
        for L in self.fi.spans:
            xs.append(xs[-1] + L)
        return xs

    def y_roof(self, x: float) -> float:
        fi, X = self.fi, self.col_x()
        if fi.roof == "multigable":
            for j, L in enumerate(fi.spans):
                if X[j] - TOL <= x <= X[j + 1] + TOL:
                    return fi.eave_height + fi.slope * min(x - X[j], X[j + 1] - x)
        if fi.roof == "single_ridge":
            W = X[-1]
            return fi.eave_height + fi.slope * min(x, W - x)
        raise ValueError(f"roof must be 'multigable' or 'single_ridge', got {fi.roof}")

    def _pieces(self) -> List[dict]:
        fi, X = self.fi, self.col_x()
        if fi.roof == "multigable":
            ridges = [X[j] + L / 2 for j, L in enumerate(fi.spans)]
        else:
            ridges = [X[-1] / 2]
        bps = sorted(set([round(x, 9) for x in X + ridges]))
        pcs = []
        for k, (a, b) in enumerate(zip(bps, bps[1:])):
            span = max(j for j in range(len(fi.spans)) if X[j] <= a + TOL)
            end_a = "S" if any(abs(a - x) < 1e-6 for x in X) else "R"
            end_b = "S" if any(abs(b - x) < 1e-6 for x in X) else "R"
            pcs.append(dict(idx=k, x0=a, x1=b, y0=self.y_roof(a), y1=self.y_roof(b),
                            span=span, kind=f"{end_a}-{end_b}",
                            int_a=0 < a < X[-1] - TOL and end_a == "S",
                            int_b=0 < b < X[-1] - TOL and end_b == "S"))
        return pcs

    def _default_profile(self, pc: dict) -> Profile:
        fi = self.fi
        if pc["idx"] in fi.rafter_overrides:
            return fi.rafter_overrides[pc["idx"]]
        if pc["kind"] == "S-R":
            return (fi.rafter_SR_int if pc["int_a"] else fi.rafter_SR_ext)
        if pc["kind"] == "R-S":
            return mirror(fi.rafter_SR_int if pc["int_b"] else fi.rafter_SR_ext)
        if pc["kind"] == "S-S":
            if fi.rafter_SS is None:
                raise ValueError("single_ridge roof needs rafter_SS (support->support) template")
            return fi.rafter_SS
        raise ValueError(f"unsupported piece kind {pc['kind']}")

    @staticmethod
    def _snap(profile: Profile, fr: List[float], name: str, warn: List[str]) -> List[Tuple[int, float]]:
        out = []
        for s, d in profile.knots:
            j = min(range(len(fr)), key=lambda i: abs(fr[i] - s))
            out.append((j, d))
            if abs(fr[j] - s) > 1e-6:
                warn.append(f"{name}: knot s={s:.3f} snapped to station {j} (s={fr[j]:.3f})")
        js = [j for j, _ in out]
        if any(b <= a for a, b in zip(js, js[1:])):
            raise ValueError(f"{name}: two knots snapped to the same station - refine the grid "
                             f"(stations_per_purlin) or move knots apart. Snapped: {js}")
        return out

    def _add_line(self, name, ltype, kind, p0, p1, n, profile, outer_normal, span=None,
                  purlin_every=1, base_support=None, extra=(), sec=None, truss=False, cut=None):
        """extra: station fractions that must exist (crane bracket levels); the nearest station moves there
        when closer than a quarter spacing, otherwise one is inserted. cut: also a station, with none between it
        and the line end (truss bottom chord level on a column). sec: truss section of every member (tube or
        double angle; profile None); truss: members pin-ended."""
        if sec is not None:
            profile = Profile([(0.0, sec["H"]), (1.0, sec["H"])],
                              [Plates(sec["t"], sec["B"], sec["t"], sec["B"], sec["t"])])
        profile.check(name)
        fr = [j / n for j in range(n + 1)]
        if cut is not None:
            extra = list(extra) + [cut]
        placed = set()
        for s in extra:
            j = min(range(len(fr)), key=lambda i: abs(fr[i] - s))
            if abs(fr[j] - s) < 1e-9:
                placed.add(fr[j])
                continue
            if 0 < j < len(fr) - 1 and abs(fr[j] - s) < 0.25 / n and fr[j] not in placed:
                fr[j] = s                             # never move a station another extra level already took
            else:
                fr = sorted(fr + [s])
            placed.add(s)
        if cut is not None:
            fr = [f for f in fr if f <= cut + 1e-9 or f >= 1 - 1e-9]
        n = len(fr) - 1
        knots = self._snap(profile, fr, name, self.warnings)
        (x0, y0), (x1, y1) = p0, p1
        L = math.hypot(x1 - x0, y1 - y0)
        stations = []
        for j in range(n + 1):
            t = fr[j]
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            # depth by linear interpolation between snapped knots
            for k in range(len(knots) - 1):
                (ja, da), (jb, db) = knots[k], knots[k + 1]
                if ja <= j <= jb:
                    D = da + (db - da) * (fr[j] - fr[ja]) / (fr[jb] - fr[ja])
                    break
            stations.append(dict(j=j, node=self._nid(x, y), x=x, y=y, s=t * L, s_frac=t, D=D,
                                 purlin=(j % purlin_every == 0)))
        line = dict(idx=len(self.lines), name=name, type=ltype, kind=kind, span=span, length=L,
                    n_sub=n, outer_normal=outer_normal, stations=stations,
                    knots=[dict(j=j, s=fr[j] * L, s_frac=fr[j], D=d) for j, d in knots],
                    plates=[asdict(p) for p in profile.plates], members=[])
        base_id = dict(col=10000, raf=20000, brk=30000, bot=40000, web=50000)[ltype] + 100 * int(name[1:])
        if n > 99:
            raise ValueError(f"{name}: {n} sub-members > 99 (ID scheme); increase spacing")
        for j in range(n):
            a, b = stations[j], stations[j + 1]
            seg = max(k for k in range(len(knots) - 1) if knots[k][0] <= j)
            reversed_ = b["D"] > a["D"] + 1e-9            # STAAD: start depth f1 >= end depth f3
            st, en = (b, a) if reversed_ else (a, b)
            ex, ey = en["x"] - st["x"], en["y"] - st["y"]
            Lm = math.hypot(ex, ey)
            xh = (ex / Lm, ey / Lm)
            if abs(xh[0]) < 1e-9:                         # vertical: local z = +Z, y = Z x x
                yh, zz = (-xh[1], 0.0), 1.0
            elif xh[0] > 0:
                yh, zz = (-xh[1], xh[0]), 1.0
            else:
                yh, zz = (xh[1], -xh[0]), -1.0
            top_is_outer = (yh[0] * outer_normal[0] + yh[1] * outer_normal[1]) > 0
            mid = base_id + j + 1
            m = dict(id=mid, line=line["idx"], line_name=name, j=j, seg=seg,
                     node_a=a["node"], node_b=b["node"], start=st["node"], end=en["node"],
                     reversed=reversed_, L=Lm, D_a=a["D"], D_b=b["D"], f1=st["D"], f3=en["D"],
                     plates=asdict(profile.plates[seg]), top_is_outer=top_is_outer,
                     sign_ic=(-1 if top_is_outer else +1), xhat=xh, yhat=yh, zz=zz,
                     purlin_a=a["purlin"], purlin_b=b["purlin"])
            if sec is not None:
                m["sec"] = sec
            if truss:
                m["truss"] = True
            self.members.append(m)
            line["members"].append(mid)
        self.lines.append(line)
        if base_support:
            self.supports[stations[0]["node"]] = base_support
        return line

    def _build(self):
        fi = self.fi
        if fi.roof not in ("multigable", "single_ridge"):
            raise ValueError("roof must be 'multigable' or 'single_ridge'")
        X, nc = self.col_x(), len(fi.spans) + 1
        for cr in fi.cranes:
            if not 0 <= cr.span < len(fi.spans):
                raise ValueError(f"crane span index {cr.span} outside 0..{len(fi.spans) - 1}")
            if not 0 < cr.bracket_level < min(self.y_roof(X[cr.span]), self.y_roof(X[cr.span + 1])):
                raise ValueError(f"crane bracket level {cr.bracket_level} m must be between base and column top")
            if not 0 < 2 * cr.e < fi.spans[cr.span]:
                raise ValueError(f"crane bracket length e = {cr.e} m does not fit the span")
        if fi.cranes and fi.bracket is None:
            raise ValueError("cranes need a bracket profile")
        tr = fi.truss
        if tr:
            if not 0 < tr.h0 < fi.eave_height:
                raise ValueError(f"truss depth h0 {tr.h0} m must be between 0 and the eave height")
            for cr in fi.cranes:
                if cr.bracket_level >= min(self.yb(X[cr.span]), self.yb(X[cr.span + 1])) - 1e-9:
                    raise ValueError("crane brackets must be below the truss bottom chord")
        # columns (base -> top), outer normal points out of the building
        for c, x in enumerate(X):
            h = self.y_roof(x)
            ext = c in (0, nc - 1)
            if ext:
                n = max(2, math.ceil(h / fi.girt_spacing - 1e-9))
                prof, kind = fi.col_ext, ("ext_L" if c == 0 else "ext_R")
                normal = (-1.0, 0.0) if c == 0 else (1.0, 0.0)
                sup = fi.base_ext
            else:
                n, prof, kind, normal, sup = fi.interior_col_subdiv, fi.col_int, "int", (-1.0, 0.0), fi.base_int
            if prof is None:
                raise ValueError(f"column profile missing for {kind}")
            lev = sorted({cr.bracket_level / h for cr in fi.cranes if c in (cr.span, cr.span + 1)})
            self._add_line(f"C{c}", "col", kind, (x, 0.0), (x, h), n, prof, normal, base_support=sup, extra=lev,
                           cut=(self.yb(x) / h if tr else None))
        # roof pieces (left -> right)
        self.pieces = self._pieces()
        k = max(1, int(fi.stations_per_purlin))
        for pc in self.pieces:
            L = math.hypot(pc["x1"] - pc["x0"], pc["y1"] - pc["y0"])
            n_p = max(1, math.ceil(L / fi.purlin_spacing - 1e-9))
            dx, dy = pc["x1"] - pc["x0"], pc["y1"] - pc["y0"]
            normal = (-dy / L, dx / L)
            if tr:
                ln = self._add_line(f"R{pc['idx']}", "raf", pc["kind"], (pc["x0"], pc["y0"]), (pc["x1"], pc["y1"]),
                                    n_p * k, None, normal, span=pc["span"], purlin_every=k,
                                    sec=tr.sections[self.side(pc)]["top"])
                ln["side"] = self.side(pc)
            else:
                self._add_line(f"R{pc['idx']}", "raf", pc["kind"], (pc["x0"], pc["y0"]),
                               (pc["x1"], pc["y1"]), n_p * k, self._default_profile(pc), normal,
                               span=pc["span"], purlin_every=k)
        if tr:
            self._build_truss()
        # crane brackets: one cantilever sub-member from the column into the span; top flange = "outer"
        self.brackets = []
        for c_i, cr in enumerate(fi.cranes):
            for side, col in ((0, cr.span), (1, cr.span + 1)):
                x = X[col]
                tip = x + cr.e if side == 0 else x - cr.e
                self.brackets.append(self._add_line(f"B{2 * c_i + side}", "brk", "brk", (x, cr.bracket_level),
                                                    (tip, cr.bracket_level), 1, fi.bracket, (0.0, 1.0)))
        # releases: leaning interior columns -> MZ released at the TOP node only; with a truss the column part
        # between the chords is pin-ended and the column below is released at the bottom chord
        if fi.interior_columns == "leaning":
            for ln in self.lines:
                if ln["type"] == "col" and ln["kind"] == "int":
                    top = ln["stations"][-2 if tr else -1]["node"]
                    m = self.mem(ln["members"][-2 if tr else -1])
                    self.releases.append((m["id"], "START" if m["start"] == top else "END"))
                    if tr:
                        self.mem(ln["members"][-1])["truss"] = True
        self.mem_index = {m["id"]: m for m in self.members}
        self._build_loads()

    def side(self, pc: dict) -> str:
        """Truss side of a roof piece: 'ext' if it touches an eave (exterior) column, else 'int'."""
        W = self.col_x()[-1]
        if pc["kind"] == "S-S":
            return "ext" if min(pc["x0"], W - pc["x1"]) < 1e-6 else "int"
        return "int" if (pc["int_a"] if pc["kind"] == "S-R" else pc["int_b"]) else "ext"

    def yb(self, x: float) -> float:
        """Truss bottom chord level at x."""
        fi = self.fi
        return fi.eave_height - fi.truss.h0 + fi.truss.bs * (self.y_roof(x) - fi.eave_height)

    def _build_truss(self):
        """Bottom chords (lines L<piece>) and web members (one line W<k> each) of the portal truss."""
        tr, yb = self.fi.truss, self.yb
        wk = 0
        for pc in self.pieces:
            sd, S = self.side(pc), tr.sections[self.side(pc)]
            top = self.line_by_name(f"R{pc['idx']}")
            n = top["n_sub"]
            bot = self._add_line(f"L{pc['idx']}", "bot", pc["kind"], (pc["x0"], yb(pc["x0"])), (pc["x1"], yb(pc["x1"])),
                                 n, None, top["outer_normal"], span=pc["span"], sec=S["bot"])
            bot["side"] = sd
            T, B = top["stations"], bot["stations"]
            # the S-R piece carries the ridge vertical; diagonals fall toward the ridge (S-R, R-S) or toward the
            # middle of a column-to-column piece (S-S, single-ridge roofs)
            centre = dict(**{"S-R": n, "R-S": 0, "S-S": n / 2})[pc["kind"]]
            webs = [("vert", B[j], T[j]) for j in range(1, n)] + ([("vert", B[n], T[n])] if pc["kind"] == "S-R" else [])
            for j in range(n):
                a, b = (j, j + 1) if j + 0.5 < centre else (j + 1, j)   # a = support side, b = centre side
                webs.append(("diag", T[a], B[b]) if tr.web == "pratt" else ("diag", B[a], T[b]))
            for role, s0, s1 in webs:
                dx, dy = s1["x"] - s0["x"], s1["y"] - s0["y"]
                Lw = math.hypot(dx, dy)
                ln = self._add_line(f"W{wk}", "web", role, (s0["x"], s0["y"]), (s1["x"], s1["y"]), 1, None,
                                    (-dy / Lw, dx / Lw), span=pc["span"], sec=S[role], truss=True)
                ln["side"] = sd
                wk += 1

    def mem(self, mid: int) -> dict:
        return next(m for m in self.members if m["id"] == mid)

    def line_by_name(self, name: str) -> dict:
        return next(l for l in self.lines if l["name"] == name)

    def col_top(self, c: int) -> int:
        return self.line_by_name(f"C{c}")["stations"][-1]["node"]

    # ------------------------------------------------------------------ sections / weight
    def member_props(self, m: dict, where: str = "mid") -> Dict[str, float]:
        if m.get("sec"):
            return m["sec"]
        D = {"a": m["D_a"], "b": m["D_b"], "mid": 0.5 * (m["D_a"] + m["D_b"])}[where]
        return i_section(D, Plates(**m["plates"]))

    def steel_weight(self) -> Dict[str, float]:
        w = {}
        for ln in self.lines:
            w[ln["name"]] = sum(self.member_props(self.mem_index[i])["A"] * self.mem_index[i]["L"]
                                for i in ln["members"]) * RHO_STEEL
        return w

    # ------------------------------------------------------------------ loads
    def _build_loads(self):
        fi, ld, X = self.fi, self.fi.loads, self.col_x()
        n = len(fi.spans)
        raf = [m for m in self.members if m["line_name"].startswith("R")]
        ext_cols = [m for m in self.members if self.lines[m["line"]]["kind"] in ("ext_L", "ext_R")]
        self.cases: Dict[int, dict] = {}

        def case(lc, title, ltype, items, selfweight=False, group="primary"):
            self.cases[lc] = dict(lc=lc, title=title, ltype=ltype, items=items,
                                  selfweight=selfweight, group=group)

        # DL: self-weight + SDL on slope + wall cladding
        items = [("M", m["id"], "GY", -ld.q_sdl_slope * fi.bay) for m in raf if ld.q_sdl_slope]
        items += [("M", m["id"], "GY", -ld.q_wall * fi.bay) for m in ext_cols if ld.q_wall]
        self.crane_info = []
        for c_i, cr in enumerate(fi.cranes):
            R = crane_reactions(cr, fi.spans[cr.span], fi.bay)
            bl, br = self.brackets[2 * c_i], self.brackets[2 * c_i + 1]
            self.crane_info.append(dict(crane=c_i, span=cr.span, R=R, rail=cr.bracket_level + cr.gantry_depth,
                                        tips=(bl["stations"][-1]["node"], br["stations"][-1]["node"]),
                                        roots=(bl["stations"][0]["node"], br["stations"][0]["node"]),
                                        depth=cr.gantry_depth))
            if R["G"]:
                items += [("J", tip, 0.0, -R["G"]) for tip in self.crane_info[-1]["tips"]]
        case(1, "DL SELFWT+SDL+WALL" + ("+GANTRY" if fi.cranes else ""), "DEAD", items, selfweight=True)
        case(2, "COLLATERAL", "DEAD",
             [("M", m["id"], "PY", -ld.q_coll_plan * fi.bay) for m in raf] if ld.q_coll_plan else [])
        self.lc_ll = []
        for j in range(n):
            lc = 101 + j
            ms = [m for m in raf if self.lines[m["line"]]["span"] == j]
            case(lc, f"ROOF LL SPAN {j + 1}", "LIVE", [("M", m["id"], "PY", -ld.q_ll_plan * fi.bay) for m in ms])
            self.lc_ll.append(lc)
        # wind
        self.lc_wl = []
        pieces = [l for l in self.lines if l["type"] == "raf"]
        for w, wc in enumerate(ld.wind_cases):
            if len(wc.cpe_roof) != len(pieces):
                raise ValueError(f"wind case '{wc.title}': {len(wc.cpe_roof)} roof Cpe for {len(pieces)} pieces")
            items = []
            for m in ext_cols:
                left = self.lines[m["line"]]["kind"] == "ext_L"
                cpe = wc.cpe_wall_left if left else wc.cpe_wall_right
                p = (cpe - wc.cpi) * ld.pd_wind * fi.bay
                items.append(("M", m["id"], "GX", p if left else -p))
            for pc, cpe in zip(pieces, wc.cpe_roof):
                p = (cpe - wc.cpi) * ld.pd_wind * fi.bay
                items += [("M", mid, "Y", -p) for mid in pc["members"]]   # local y = outward
            lc = 201 + w
            case(lc, f"WL {wc.title}"[:40], "WIND", items)
            self.lc_wl.append(lc)
        # crane: vertical (max wheel loads at one rail, impact included) and surge at rail level (one side of
        # the frame at a time, 6.3(c)); surge = FX at the column node + MZ = -FX d for the rail height d
        self.lc_cl = []
        for ci in self.crane_info:
            b, R, (tl, tr), (rl, rr), d = 501 + 10 * ci["crane"], ci["R"], ci["tips"], ci["roots"], ci["depth"]
            k = ci["crane"] + 1
            case(b, f"CRANE {k} VERT MAX LEFT RAIL", "LIVE", [("J", tl, 0.0, -R["Rmax"]), ("J", tr, 0.0, -R["Rmin"])])
            case(b + 1, f"CRANE {k} VERT MAX RIGHT RAIL", "LIVE", [("J", tl, 0.0, -R["Rmin"]), ("J", tr, 0.0, -R["Rmax"])])
            case(b + 2, f"CRANE {k} SURGE LEFT RAIL +X", "LIVE", [("J", rl, R["H"], 0.0, -R["H"] * d)])
            case(b + 3, f"CRANE {k} SURGE RIGHT RAIL +X", "LIVE", [("J", rr, R["H"], 0.0, -R["H"] * d)])
            self.lc_cl += [b, b + 1, b + 2, b + 3]
        # notional horizontal loads at column tops, +X, per unit gravity case (Cl. 4.3.6)
        trib = [0.0] * (n + 1)
        for j, L in enumerate(fi.spans):
            trib[j] += L / 2
            trib[j + 1] += L / 2
        W_tot = X[-1]
        roof = [m for m in self.members if self.lines[m["line"]]["type"] in ("raf", "bot", "web")]
        raf_sw = sum(self.member_props(m)["A"] * m["L"] for m in roof) * GAMMA_STEEL * ld.sw_factor
        raf_slope_len = sum(m["L"] for m in raf)
        g_dl = raf_sw + ld.q_sdl_slope * fi.bay * raf_slope_len
        g_coll = ld.q_coll_plan * fi.bay * W_tot
        self.notional_basis = dict(DL_roof=g_dl, COLL=g_coll)
        case(301, "NOTIONAL DL+COLL +X", None,
             [("J", self.col_top(c), NOTIONAL_FRACTION * (g_dl + g_coll) * trib[c] / W_tot, 0.0)
              for c in range(n + 1)])
        self.lc_nl = []
        for j, L in enumerate(fi.spans):
            g = ld.q_ll_plan * fi.bay * L
            case(311 + j, f"NOTIONAL LL SPAN {j + 1} +X", None,
                 [("J", self.col_top(j), NOTIONAL_FRACTION * g / 2, 0.0),
                  ("J", self.col_top(j + 1), NOTIONAL_FRACTION * g / 2, 0.0)])
            self.lc_nl.append(311 + j)
        # virtual unit cases (SLS file only): drift per column line, relative deflection per span
        self.virtual = {}
        for c in range(n + 1):
            case(401 + c, f"VIRTUAL UNIT FX TOP C{c}", None, [("J", self.col_top(c), 1.0, 0.0)],
                 group="virtual")
            self.virtual[401 + c] = dict(kind="drift", column_line=c, node=self.col_top(c))
        rnodes = {}
        for ln in pieces:
            for st in ln["stations"]:
                rnodes[st["node"]] = st["x"]
        for j, L in enumerate(fi.spans):
            xm = X[j] + L / 2
            mid = min(rnodes, key=lambda nd: abs(rnodes[nd] - xm))
            if abs(rnodes[mid] - xm) > 1e-6:
                self.warnings.append(f"span {j + 1}: no roof station at mid-span; using x={rnodes[mid]:.3f}")
            a, b = self.col_top(j), self.col_top(j + 1)
            case(451 + j, f"VIRTUAL REL DEFL SPAN {j + 1}", None,
                 [("J", mid, 0.0, -1.0), ("J", a, 0.0, 0.5), ("J", b, 0.0, 0.5)], group="virtual")
            self.virtual[451 + j] = dict(kind="rel_defl", span=j, mid_node=mid, support_nodes=[a, b])
        self._build_combos()

    def crane_variants(self) -> List[Tuple[str, List[Tuple[int, float]]]]:
        """CL arrangements: each crane alone (max wheel loads left / right x surge on either rail x +/-X,
        6.3(c)), and every pair of cranes in two bays (6.4.2(b)) with surge at each crane's max-load rail, same
        direction. ponytail: pairs only, not 3+ cranes together (IS 875-2 6.4.2 asks for two)."""
        out = []
        for ci in self.crane_info:
            b, k = 501 + 10 * ci["crane"], ci["crane"] + 1
            for v in (0, 1):
                for h in (0, 1):
                    for sg in (1, -1):
                        out.append((f"K{k}V{'LR'[v]}H{'LR'[h]}{'+-'[sg < 0]}", [(b + v, 1.0), (b + 2 + h, sg)]))
        for i, a in enumerate(self.crane_info):
            for c in self.crane_info[i + 1:]:
                ba, bc = 501 + 10 * a["crane"], 501 + 10 * c["crane"]
                for va in (0, 1):
                    for vc in (0, 1):
                        for sg in (1, -1):
                            out.append((f"K{a['crane'] + 1}{'LR'[va]}K{c['crane'] + 1}{'LR'[vc]}{'+-'[sg < 0]}",
                                        [(ba + va, 1.0), (ba + 2 + va, sg), (bc + vc, 1.0), (bc + 2 + vc, sg)]))
        return out

    def _combo_pairs(self, r: dict, pat: Optional[Tuple[int, ...]], wlc: Optional[int], nsign: int, cv=None):
        pairs = []
        if r["D"]:
            pairs.append((1, r["D"]))
        if r["C"]:
            pairs.append((2, r["C"]))
        if r["L"] and pat is not None:
            pairs += [(self.lc_ll[j], r["L"]) for j in pat]
        if r["W"] and wlc is not None:
            pairs.append((wlc, r["W"]))
        if r.get("CL") and cv:
            pairs += [(lc, r["CL"] * f) for lc, f in cv]
        if nsign:
            if r["D"] or r["C"]:
                pairs.append((301, nsign * max(r["D"], r["C"])))
            if r["L"] and pat is not None:
                pairs += [(self.lc_nl[j], nsign * r["L"]) for j in pat]
        return pairs

    def _expand(self, rules, start, prefix):
        pats = ll_patterns(len(self.fi.spans))
        out, lc = [], start
        for r in rules:
            plist = [(None, None)]
            if r.get("ll") == "patterns":
                plist = list(pats.items())
            elif r.get("ll") == "ALL":
                plist = [("ALL", pats["ALL"])]
            wl = [None] if not r["W"] else self.lc_wl
            if (r["W"] and not self.lc_wl) or (r.get("CL") and not self.crane_info):
                continue
            nsigns = [1, -1] if r.get("notional") else [0]
            cvs = self.crane_variants() if r.get("CL") else [(None, None)]
            for pname, pat in plist:
                for w in wl:
                    for ns in nsigns:
                        for cname, cv in cvs:
                            tag = r["tag"]
                            if pname:
                                tag += f" L[{pname}]"
                            if w:
                                tag += f" W{w - 200}"
                            if ns:
                                tag += " N+X" if ns > 0 else " N-X"
                            if cname:
                                tag += f" {cname}"
                            out.append(dict(lc=lc, title=f"{prefix}{lc - start + 1:02d} {tag}"[:48],
                                            pairs=self._combo_pairs(r, pat, w, ns, cv),
                                            gravity_only=(r["W"] == 0)))
                            lc += 1
        return out

    def _build_combos(self):
        self.uls = self._expand(ULS_RULES, 1001, "U")
        self.sls = self._expand(SLS_RULES, 2001 + 1000 * (len(self.uls) // 1000), "S")

    # ------------------------------------------------------------------ STAAD writer
    def _std_header(self, kind: str) -> List[str]:
        fi, ld = self.fi, self.fi.loads
        L = ["STAAD PLANE",
             f"* {fi.name} - {kind}",
             "* Generated by peb_frame_model.py. Units METER KN. Code basis IS 800:2007 (LSM).",
             f"* Spans {fi.spans} m, eave {fi.eave_height} m, slope 1:{1 / fi.slope:.3g}, "
             f"roof {fi.roof}, bay {fi.bay} m",
             f"* Interior columns: {fi.interior_columns}; bases ext {fi.base_ext}, int {fi.base_int}",
             "* TAPERED f1..f7 = start D, tw, end D, top bf, top tf, bottom bf, bottom tf (STAAD TR.20.3)",
             "* Sub-members are oriented deeper end first (f1 >= f3); see station_map.json"]
        if ld.wind_cases and not ld.cpe_verified:
            L.append("* !!! WIND Cpe VALUES ARE DEMO PLACEHOLDERS - NOT FROM IS 875-3. DO NOT ISSUE. !!!")
        L += ["UNIT METER KN", "JOINT COORDINATES"]
        L += [f"{nid} {x:.4f} {y:.4f} 0" for nid, (x, y) in sorted(self.nodes.items())]
        L.append("MEMBER INCIDENCES")
        L += [f"{m['id']} {m['start']} {m['end']}" for m in self.members]
        L += ["DEFINE MATERIAL START", "ISOTROPIC STEEL", f"E {E_STEEL:.6g}", f"POISSON {POISSON}",
              f"DENSITY {GAMMA_STEEL * self.fi.loads.sw_factor:.4f}", f"ALPHA {ALPHA_T:g}", "DAMP 0.03",
              "TYPE STEEL", "END DEFINE MATERIAL", "MEMBER PROPERTY"]
        das: Dict[str, List[int]] = {}
        for m in self.members:
            if m.get("sec"):
                if m["sec"]["shape"] == "DA":
                    das.setdefault(m["sec"]["name"], []).append(m["id"])
                continue
            p = m["plates"]
            if m["top_is_outer"]:
                bt, tt, bb, tb = p["bf_out"], p["tf_out"], p["bf_in"], p["tf_in"]
            else:
                bt, tt, bb, tb = p["bf_in"], p["tf_in"], p["bf_out"], p["tf_out"]
            L.append(f"{m['id']} TAPERED {m['f1']:.4f} {p['tw']:.4f} {m['f3']:.4f} "
                     f"{bt:.4f} {tt:.4f} {bb:.4f} {tb:.4f}")
        for nm, ids in sorted(das.items()):   # STAAD 2026 gives TABLE D (double angle) single-angle stiffness
            s_ = next(m["sec"] for m in self.members if m.get("sec") and m["sec"]["name"] == nm)   # (tested):
            L.append(f"* {nm}, gusset {1000 * s_['tg']:g} mm: PRISMATIC (IS 800 check by the optimiser)")
            L += [f"{r} PRIS AX {s_['A']:.5g} IZ {s_['Iz']:.5g} IY {s_['Iy']:.5g} IX {s_['It']:.4g}" for r in _ranges(ids)]
        tubes: Dict[Tuple[str, str], List[int]] = {}
        for m in self.members:
            if m.get("sec") and m["sec"]["shape"] != "DA":
                tubes.setdefault((m["sec"]["shape"], m["sec"]["name"]), []).append(m["id"])
        if tubes:                                    # STAAD's IS 4923 database (same properties as tubes_is4923.json)
            L.append("MEMBER PROPERTY 'INDIA (IS 4923-2017).DB3'")
            for (shape, nm), ids in sorted(tubes.items()):
                L += [f"{r} TABLE '{shape}' ST '{nm}'" for r in _ranges(ids)]
        L += ["CONSTANTS", "MATERIAL STEEL ALL", "SUPPORTS"]
        for nd, s in sorted(self.supports.items()):
            L.append(f"{nd} {s}")
        if self.releases:
            L.append("MEMBER RELEASE")
            L += [f"{mid} {end} MZ" for mid, end in self.releases]
        trs = [m["id"] for m in self.members if m.get("truss")]
        if trs:
            L += ["MEMBER TRUSS"] + [f"{r} -" for r in _ranges(trs)[:-1]] + [_ranges(trs)[-1]]
        return L

    def _std_case(self, c: dict) -> List[str]:
        head = f"LOAD {c['lc']}" + (f" LOADTYPE {c['ltype']}" if c["ltype"] else "") + f" TITLE {c['title']}"
        L = [head]
        if c["selfweight"]:
            L.append("SELFWEIGHT Y -1")      # density already carries sw_factor
        mem = [it for it in c["items"] if it[0] == "M"]
        jnt = [it for it in c["items"] if it[0] == "J"]
        if mem:
            L.append("MEMBER LOAD")
            groups: Dict[Tuple[str, float], List[int]] = {}
            for _, mid, d, w in mem:
                groups.setdefault((d, round(w, 6)), []).append(mid)
            for (d, w), ids in groups.items():
                L += [f"{r} UNI {d} {w:.6g}" for r in _ranges(ids)]
        if jnt:
            L.append("JOINT LOAD")
            for it in jnt:
                nd, fx, fy, mz = it[1], it[2], it[3], (it[4] if len(it) > 4 else 0.0)
                L.append(f"{nd}" + (f" FX {fx:.6g}" if fx else "") + (f" FY {fy:.6g}" if fy else "")
                         + (f" MZ {mz:.6g}" if mz else ""))
        return L

    def write_std(self, path: str, kind: str) -> None:
        kind = kind.upper()
        L = self._std_header(kind)
        prim = [1, 2] + self.lc_ll + self.lc_cl
        if kind == "ULS":
            prim += self.lc_wl + [301] + self.lc_nl
            combos, mode = self.uls, "REPEAT"
        elif kind == "SLS":
            prim += self.lc_wl + sorted(self.virtual)
            combos, mode = self.sls, "COMB"
        elif kind == "BUCKLING":
            prim += [301] + self.lc_nl
            combos, mode = [c for c in self.uls if c["gravity_only"]], "REPEAT"
        else:
            raise ValueError(kind)
        for lc in prim:
            L += self._std_case(self.cases[lc])
        for c in combos:
            if mode == "REPEAT":
                L += [f"LOAD {c['lc']} TITLE {c['title']}", "REPEAT LOAD"] + _wrap_pairs(c["pairs"])
            else:
                L += [f"LOAD COMBINATION {c['lc']} {c['title']}"] + _wrap_pairs(c["pairs"])
        if kind == "ULS":
            L.append("PDELTA 20 ANALYSIS SMALLDELTA PRINT STATICS CHECK")
        elif kind == "SLS":
            L.append("PERFORM ANALYSIS PRINT STATICS CHECK")
        else:
            L.append("PERFORM BUCKLING EIGEN" if self.fi.buckling_eigen
                     else "PERFORM BUCKLING ANALYSIS MAXSTEPS 15")
        L.append("FINISH")
        too_long = [ln for ln in L if len(ln) > 72 and not ln.startswith("*")]
        if too_long:
            raise ValueError(f"{len(too_long)} STAAD lines exceed 72 chars, e.g. {too_long[0]!r}")
        with open(path, "w", newline="\r\n") as f:
            f.write("\n".join(L) + "\n")

    # ------------------------------------------------------------------ load totals
    def load_vector(self, m: dict, d: str, w: float) -> Tuple[float, float]:
        """Global force per unit member length for a member load item."""
        xh, yh = m["xhat"], m["yhat"]
        if d == "GY":
            return 0.0, w
        if d == "GX":
            return w, 0.0
        if d == "PY":
            return 0.0, w * abs(xh[0])            # projected length / member length
        if d == "Y":
            return w * yh[0], w * yh[1]
        raise ValueError(d)

    def case_totals(self, lc: int) -> Dict[str, float]:
        c, fx, fy = self.cases[lc], 0.0, 0.0
        for it in c["items"]:
            if it[0] == "M":
                m = self.mem_index[it[1]]
                gx, gy = self.load_vector(m, it[2], it[3])
                fx, fy = fx + gx * m["L"], fy + gy * m["L"]
            else:
                fx, fy = fx + it[2], fy + it[3]
        if c["selfweight"]:
            fy -= sum(self.member_props(m)["A"] * m["L"] for m in self.members) * GAMMA_STEEL * self.fi.loads.sw_factor
        return dict(FX=fx, FY=fy)

    # ------------------------------------------------------------------ station map
    def station_map(self) -> dict:
        return dict(
            name=self.fi.name, units="METER KN", fy_MPa=self.fi.fy,
            conventions=dict(
                end_forces="STAAD local end actions on member [FX FY FZ MX MY MZ]",
                N_comp="FX(start) = -FX(end), + = compression",
                M_ic="sign_ic * M_ly; M_ly(start) = -MZ1, M_ly(end) = +MZ2; + = inner flange in compression",
                station_of_start="j+1 if reversed else j"),
            lines=[{k: v for k, v in l.items()} for l in self.lines],
            members=[{k: v for k, v in m.items() if k not in ("xhat", "yhat")} for m in self.members],
            cases={lc: dict(title=c["title"], group=c["group"]) for lc, c in self.cases.items()},
            uls=self.uls, sls=self.sls, virtual=self.virtual,
            expected_totals={lc: self.case_totals(lc) for lc in self.cases},
            column_tops={f"C{c}": self.col_top(c) for c in range(len(self.fi.spans) + 1)},
            supports=self.supports, releases=self.releases, warnings=self.warnings)


# ============================================================================= LINEAR CHECK
class LinearFrame2D:
    """Independent first-order 2D frame solver on the SAME members/loads/orientations,
    returning end forces in STAAD's convention. Used to (i) test the sign conventions and
    load directions before STAAD runs, (ii) give benchmark numbers to compare with STAAD.
    Tapered sub-members use mid-length properties (short elements -> small error)."""

    def __init__(self, model: FrameModel, shear: bool = False):
        """shear=True reproduces STAAD's member stiffness: web shear deformation (Timoshenko, As = d tw) on
        constant-depth members; none on tapered ones (matched empirically against STAAD PDELTA runs: As = d tw
        fits prismatic TAPERED members to 0.09 %, while tapered members fit best without shear deformation).
        Default False keeps the closed-form (Euler-Bernoulli) self-checks of test_model.py exact."""
        self.mdl = model
        self.ids = sorted(model.nodes)
        self.dof = {nid: 3 * i for i, nid in enumerate(self.ids)}
        self.ndof = 3 * len(self.ids)
        rel = {mid: end for mid, end in model.releases}
        self.el = []
        K = np.zeros((self.ndof, self.ndof))
        for m in model.members:
            pr = model.member_props(m)
            As = None
            if shear and not m.get("sec") and abs(m["D_a"] - m["D_b"]) < 1e-9:   # STAAD: constant depth only
                p = m["plates"]
                As = (0.5 * (m["D_a"] + m["D_b"]) - p["tf_in"] - p["tf_out"]) * p["tw"]
            k0 = self._k_local(E_STEEL, pr["A"], pr["Iz"], m["L"], As)
            r = [2, 5] if m.get("truss") else [{"START": 2, "END": 5}[rel[m["id"]]]] if m["id"] in rel else []
            k, _ = self._condense(k0, np.zeros(6), r)
            T = self._T(m)
            idx = self.dof[m["start"]] + np.arange(3)
            idx = np.concatenate([idx, self.dof[m["end"]] + np.arange(3)])
            K[np.ix_(idx, idx)] += T.T @ k @ T
            self.el.append(dict(m=m, k=k, k0=k0, T=T, idx=idx, r=r))
        self.K = K
        fixed = []
        for nd, s in model.supports.items():
            d = self.dof[nd]
            fixed += [d, d + 1] + ([d + 2] if s.upper() == "FIXED" else [])
        self.fixed = np.array(sorted(fixed))
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)

    @staticmethod
    def _k_local(E, A, I, L, As=None):
        """Plane frame element; As (shear area) given -> Timoshenko, phi = 12 E I / (G As L^2)."""
        ph = 12 * E * I / (E / (2 * (1 + POISSON)) * As * L * L) if As else 0.0
        a, b = E * A / L, E * I / (L ** 3 * (1 + ph))
        c, d4, d2 = 6 * L * b, (4 + ph) * L * L * b, (2 - ph) * L * L * b
        return np.array([[a, 0, 0, -a, 0, 0], [0, 12 * b, c, 0, -12 * b, c], [0, c, d4, 0, -c, d2],
                         [-a, 0, 0, a, 0, 0], [0, -12 * b, -c, 0, 12 * b, -c], [0, c, d2, 0, -c, d4]])

    @staticmethod
    def _T(m):
        (xx, xy), (yx, yy), zz = m["xhat"], m["yhat"], m["zz"]
        R = np.array([[xx, xy, 0], [yx, yy, 0], [0, 0, zz]])
        T = np.zeros((6, 6))
        T[:3, :3] = R
        T[3:, 3:] = R
        return T

    @staticmethod
    def _condense(k, f, rs):
        """Static condensation of the released end rotations rs (local dof 2 / 5), one after the other."""
        k, f = k.copy(), f.copy()
        for r in rs:
            f = f - k[:, r] * f[r] / k[r, r]
            k = k - np.outer(k[:, r], k[r, :]) / k[r, r]
            k[r, :], k[:, r], f[r] = 0.0, 0.0, 0.0
        return k, f

    def _fef(self, e, qx, qy):
        L = e["m"]["L"]
        f = np.array([-qx * L / 2, -qy * L / 2, -qy * L ** 2 / 12, -qx * L / 2, -qy * L / 2, qy * L ** 2 / 12])
        return self._condense(e["k0"], f, e["r"])[1]

    def solve(self, combo: Dict[int, float]):
        mdl, F = self.mdl, np.zeros(self.ndof)
        fef = {e["m"]["id"]: np.zeros(6) for e in self.el}
        emap = {e["m"]["id"]: e for e in self.el}
        for lc, fac in combo.items():
            c = mdl.cases[lc]
            items = list(c["items"])
            if c["selfweight"]:
                items += [("M", m["id"], "GY", -mdl.member_props(m)["A"] * GAMMA_STEEL * mdl.fi.loads.sw_factor)
                          for m in mdl.members]
            for it in items:
                if it[0] == "J":
                    d = self.dof[it[1]]
                    F[d] += fac * it[2]
                    F[d + 1] += fac * it[3]
                    F[d + 2] += fac * (it[4] if len(it) > 4 else 0.0)
                else:
                    e = emap[it[1]]
                    gx, gy = mdl.load_vector(e["m"], it[2], it[3])
                    (xx, xy), (yx, yy) = e["m"]["xhat"], e["m"]["yhat"]
                    fe = self._fef(e, fac * (gx * xx + gy * xy), fac * (gx * yx + gy * yy))
                    fef[it[1]] += fe
                    F[e["idx"]] -= e["T"].T @ fe
        u = np.zeros(self.ndof)
        Kff = self.K[np.ix_(self.free, self.free)]
        u[self.free] = np.linalg.solve(Kff, F[self.free])
        R = self.K @ u - F
        ends = {}
        for e in self.el:
            f = e["k"] @ (e["T"] @ u[e["idx"]]) + fef[e["m"]["id"]]
            ends[e["m"]["id"]] = ([f[0], f[1], 0.0, 0.0, 0.0, f[2]],
                                  [f[3], f[4], 0.0, 0.0, 0.0, f[5]])
        disp = {nd: tuple(u[self.dof[nd]:self.dof[nd] + 3]) for nd in self.ids}
        react = {nd: tuple(R[self.dof[nd]:self.dof[nd] + 3]) for nd in mdl.supports}
        return dict(u=disp, R=react, ends=ends)

    def stations(self, res) -> Dict[str, List[dict]]:
        """Station values per line from end forces (same code path the STAAD runner uses)."""
        out: Dict[str, Dict[int, dict]] = {}
        for m in self.mdl.members:
            fs, fe = res["ends"][m["id"]]
            ln = self.mdl.lines[m["line"]]
            for j, N, V, M in end_forces_to_stations(m, fs, fe):
                st = ln["stations"][j]
                d = out.setdefault(ln["name"], {}).setdefault(j, dict(j=j, x=st["x"], y=st["y"], s=st["s"],
                                                                        D=st["D"], N=[], V=[], M=[]))
                d["N"].append(N)
                d["V"].append(V)
                d["M"].append(M)
        return {k: [v[j] for j in sorted(v)] for k, v in out.items()}
