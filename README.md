# PEB multi-span 2D frame → STAAD.Pro (.std) + OpenSTAAD runner

## Files
| File | Role |
|---|---|
| `peb_frame_model.py` | Generator: inputs → 3 `.std` files + `station_map.json`; independent linear 2D solver for cross-checks |
| `demo_two_span.py` | Example inputs (2 × 24 m multigable, leaning interior column). Edit and run |
| `openstaad_runner.py` | Windows + running STAAD: open → analyse → extract station tables |
| `test_model.py`, `test_runner_plumbing.py` | Self-checks (closed-form, statics, sign conventions, runner data path) |

## Deploy on another PC (Windows)
`python build_exe.py` (needs `pip install pyinstaller`) builds and self-tests:
- `dist/PEB_Gable_Optimiser_win64.zip` - unzip anywhere and run `PEB_Gable_Optimiser.exe` (keep `_internal`
  beside it); no Python needed. `PEB_Gable_Optimiser.exe --selftest` writes `selftest_result.txt` (analyses of all
  three schemes, a 2-process optimisation, STAAD lookup).
- `dist/PEB_Gable_Optimiser_source.zip` - sources + `requirements.txt` + `run_app.bat` for any PC with Python >= 3.12.
STAAD.Pro is only needed for "Optimise + STAAD verify": found under `C:\Program Files\Bentley\...` (newest
first) or via the environment variable `PEB_STAAD_EXE` = full path of `SProStaad.exe`. The exe is built for
Windows x64; for another OS build it there from the source zip.

## Optimiser app (IS 800:2007)
`python app.py` - desktop app. Enter frame / loads / Cpe / seed sections, then:
- **Check seed** - one analysis of the seed sections, D/C-coloured frame + report.
- **Optimise** - fast engine only (about 5-10 min on 11 cores for the 2-span demo).
- **Optimise + STAAD verify** - then runs `<name>_ULS_PDELTA.std` (+ IS800 LSD CHECK CODE) and `<name>_SLS.std`
  headless in SProStaad, re-checks with STAAD's forces, tightens the target and repeats if they fail.
- **Export .std** - ULS/SLS/BUCKLING files + station_map.json + design JSON + report for the current design.
  The ULS file carries the full design block (`export_std`): `PARAMETER / CODE IS800 LSD / FYLD / FU / STP 2`
  (welded), `LZ / LY / LX` per member from the optimiser's restraint layout, `TST 1` + `TSP c` (web stiffeners),
  `RATIO` if the target is not 1, `TRACK 2`, `CHECK CODE ALL`, after `LOAD LIST` = the ULS combinations. The SLS
  file states the Table 6 limits and prints the joint displacements at the check nodes. STAAD 2026's IS800
  LSD check ignores `DFF / DJ1 / DJ2` on TAPERED members (tested with DFF 20000: no change), so no deflection
  code check is written.

| File | Role |
|---|---|
| `is800.py` | Station checks: Table 2 class (Class 4 rejected), 8.4 shear incl. web buckling, 9.2.2 high shear, 8.2.2 LTB, 7.1.2 buckling, 9.3.1 / 9.3.2.2 interaction |
| `optimiser.py` | P-Delta fast engine ((K - Kg) u = F, N iterated), SLS, lambda_cr, discrete greedy search (repair -> trim -> shake), STAAD verify, report |
| `app.py` | Tkinter GUI |
| `test_optimiser.py` | P-Delta bracket check + single-span optimisation must end feasible and lighter |

Design variables: depth at every knot and tw / bf / tf per segment (symmetric flanges) of each template
(`raf_ext`, `raf_int`, `raf_ss`, `col_ext`, `col_int`), from the catalogue `CAT` in optimiser.py.
Restraints: purlins/girts hold the outer flange; fly braces at every n-th purlin/girt hold the inner flange;
interior columns are braced at the ends only. In-plane length = line length (sway is in the P-Delta forces).
Verified on the demo seed: our P-Delta moments are within 0.5 % of STAAD PDELTA.

**Slender webs** (Limits tab checkbox, `options.slender_web`): where d/tw exceeds the Table 2 semi-compact
limit, IS 800 3.7.2 -> 8.2.1.1(a) is used: moment and axial force taken by the flanges only, web carries
shear only (8.4.2.2(a) shear buckling). Web limited to d/tw <= min(200 eps_w (8.6.1.1a), 345 eps_f^2
(8.6.1.2a)), no transverse stiffeners; slender flanges still rejected. STAAD's IS800 check passes slender
sections on gross properties, so verify excludes STAAD "Section Class: Slender" members from STAAD's max
ratio and lists them; our 8.2.1.1(a) check governs those. Demo: 4809 kg -> 4558 kg (-5.2 %).

**Transverse web stiffeners** (Limits tab, `options.stiffeners` + `stiffener_penalty_kg`): each sub-member
tries no stiffeners, then spacing c = L/1, L/2, L/3 (L = purlin/girt spacing; k = 1 puts them at the
purlin/girt stations) and keeps the fewest that pass. Stiffened webs get 8.4.2.2(a) kv(c/d) (simple
post-critical, no tension field) and the 8.6.1.1(b) / 8.6.1.2(b) d/tw limits (up to 270 eps_w). Stiffeners
are single-sided flat plates, outstand to the flange tip <= 20 t eps (8.7.1.2), Is >= 8.7.2.4 minimum;
8.7.2.5 Fq = V - Vcr/gm0 <= 0 while the shear check passes. Stiffener steel is added to the weight; the
optional penalty (kg per stiffener) stands for fabrication labour in the objective only. STAAD's IS800 LSD
check takes the stiffeners via `TST 1` + `TSP` (spacing): shear capacity 147.0 kN vs ours 148.1 (79.6 unstiffened);
(`STIFF` is not an IS800 LSD parameter - the earlier export used it and STAAD ignored it). Tension-field members are listed
separately in the verify report. Verify also cross-checks STAAD's P-Delta moments against the fast engine
and lists STAAD's 'divergent' P-Delta warnings (seen on near-zero-sway cases; moments still within 0.6 %).

**Tension field** (Limits tab, `options.tension_field`, needs stiffeners): IS 800 8.4.2.2(b) in stiffened
panels with c/d >= 1: Vtf = Av tau_b + 0.9 w_tf tw fv sin(phi) <= Vp, phi = atan(d/c), fv, psi, anchorage
lengths s from the reduced flange moment Mfr (flange force from N and M at the station). Anchorage per
8.5.1 / 8.5.3: each chain of tension-field panels must end in a panel designed by 8.4.2.2(a) that passes a
beam check for R_tf = Hq/2 and M_tf = Hq d/10 (Hq = 1.25 Vp sqrt(1 - Vcr/Vp), reduced when V < Vtf);
its end stiffener takes M_tf / c. End posts (8.5.2) are not designed, so a panel next to a knee, ridge or
base never uses tension field. Stiffeners in tension-field panels are sized for Fq = V - Vcr/gm0 <= Fqd
(8.7.2.5, Fqd per 8.7.1.5: core section + 20 tw web each side, curve c, KL = 0.7 d). The report gives the
8.6.3.3 web-flange weld demand fv tw. NOTE: IS 800 prints w_tf = d cos(phi) + (c - sc - st) sin(phi); the
Porter-Rockey-Evans basis has a minus sign (stronger flanges -> wider band). The tool uses the lower of the two
readings - check your copy / amendments.

**End posts** (Limits tab, `options.end_posts`, needs tension field): IS 800 8.5.2(b) double-stiffener end
post at knee / haunch panels (rafter support ends: eave knee and valley haunch; tops of exterior columns).
The member-end plate plus a flat stiffener at e (0.10-0.40 m) enclose a web strip e x tw, checked as a beam
spanning between the flanges: R_tf <= e tw fy/(sqrt3 gm0), M_tf <= (As e + tw e^2/4) fy/gm0, stiffener
strut M_tf/e <= Fqd (8.7.1.5), Is >= 8.7.2.4 for the remaining panel, and c - e >= d so that panel still
qualifies for tension field. The end post counts as one stiffener in the ranking, so tension field + end
post is only chosen when it beats adding one more intermediate stiffener - in every case tried so far
(demo, 14 m heavy-load frame) halving the spacing under 8.4.2.2(a) was as cheap or cheaper, so end posts
mostly matter when the stiffener spacing is capped. The 8.5.2(a) single-stiffener form (end post no wider /
thicker than the flange) is not implemented; the end plate's bearing and bolts are connection design.

**Purlin loads into the rafter web** (always on; Frame tab `purlin.b1_mm`, `purlin.web_KL`): at every interior purlin
the factored roof load normal to the rafter (from the member loads, tributary half sub-member each side; rafter self-weight
excluded) is checked per ULS combination: 8.7.4 bearing Fw = (b1 + n2) tw fy/gm0 with n2 = 5 tf (1:2.5 through the
flange), also for uplift pulling the flange (8.7.8); 8.7.3.1 web buckling Fcdw = (b1 + n1) tw fcd, n1 = D (45 deg to
mid-depth), web strut r = tw/sqrt12, KL = web_KL x d, curve c. b1 = purlin cleat stiff bearing length (default 20 mm =
8 mm cleat + 2 x 6 mm welds, 8.7.1.3). web_KL default 1.0 (8.7.1.5(b), loaded flange not restrained against rotation);
0.7 (8.7.1.5(a)) if the purlin connection restrains it - this halves the demand ratio for thin webs. A purlin on a
transverse stiffener loads that stiffener (8.7.5.1 with 8.7.2.5: F = max(Fq, Fx); bearing 8.7.5.2 As fy/(0.8 gm0));
with the stiffener option, a load-carrying stiffener is added at any purlin whose bare web fails. Not checked: girt
loads on columns, stiffener eccentricity moment of single-sided stiffeners (small for purlin loads).

**Fast-engine stiffness** (`LinearFrame2D(shear=True)`): web shear deformation (A_s = d tw) on constant-depth
members, none on tapered ones - this reproduces STAAD: prismatic frame 1.93 % -> 0.09 % moment difference,
tapered demo 0.54 % either way (STAAD's tapered stiffness evidently omits shear deformation).

**Optimisation problem** (full statement in the optimiser.py docstring):
min f(x) = steel weight + stiffener mass + penalty x n_stiffeners, over catalogue indices x
s.t. IS 800 D/C <= eta at every station and combination; SLS ratios <= 1; stiffener F/Fqd <= 1;
h_joint: where two roof pieces meet (ridge, valley) the depth and the end-segment plates are equal -> the
rafters mirror each other at the apex splice while each keeps its own taper (the interior haunch can be
deeper than the eave haunch); h_bf (option, default on): one flange width per member. Equalities are
eliminated by linking design variables (36 plate/depth slots -> 28 variables in the demo).

**Six-section rafters** (Limits tab, `options.rafter_six_sections`, default on; raf_ext / raf_int):
knots from the ULS |M| envelope of the current design (support -> ridge): k0 haunch; k1, k2 where |M| has
fallen 1/3 and 2/3 of the way to the low-moment zone; k3..k4 the low-moment zone (|M| <= M_min + 0.2
(M_max - M_min)) with a uniform section; k5 where |M| has risen halfway to the ridge value; k6 ridge. Rules:
D0 >= D1 >= D2 >= D3 = D4 <= D5 <= D6 <= min D0 (deepest at the haunches), plates change only at k2 and k4
(three plate zones, as before), and D5 >= the straight line D4 -> D6: the taper toward the apex may flatten
but never steepen. Knots are re-derived from the optimised design's envelope and the search repeats once
if they move. Why the old apex spiked: in the 2 x 24 m leaning-column frame the ridge |M| envelope (~540 kNm)
equals the knee moment, the ridge depth is one shared variable and nothing limited the last 20 % taper, so
the cheapest fix for the ridge moment was a deep apex point (650 -> 900 -> 675 mm within 2.4 m each side).

**Crane on brackets** (Crane tab, `crane.spans` = 1-based spans with a crane): IS 875-2 6.3 / 6.4. Wheel loads
Pmax / Pmin = bridge/(2n) + (crab + SWL)(Lc - a_min or a_min)/(Lc n), Lc = span - 2e; per bracket x gantry
reaction influence il = sum(1 - i w / bay) (wheels at the column and one wheel base away); impact on Pmax
only (6.3(a)); surge = surge x (crab + SWL)/n x il at rail level, one side of the frame at a time, either
direction (6.3(c)) -> FX + MZ = -FX x gantry depth at the column bracket node. Cases 501+10k: max-left,
max-right, surge left rail, surge right rail. 8 arrangements per crane; with 2+ cranes also every pair in
two bays (6.4.2(b)). ULS (Table 4, crane = second live load, each leading in turn): 1.5(D+L)+1.05CL,
1.5(D+CL)+1.05L, 1.2(D+L)+1.05CL+0.6W, 1.2(D+CL)+1.05L+0.6W, 1.2(D+L+W)+0.53CL, 1.2(D+CL+W)+0.53L. SLS
(Table 6): rail drift <= H_rail/200 (400 cab-operated) and rail gauge change <= 10 mm under CL and 0.8(CL+W).
Brackets are designed as members (template `bracket`, LTB / buckling length 2e - check Table 16 for your
detail). Not included: tractive force along the rails (6.3(d), longitudinal bracing), gantry girder, bracket
tip bearing stiffener, column web stiffeners at the bracket flanges.

**PEB vs portal truss** (Truss / Cost tab; `optimise_both`, used by the app's Optimise buttons): the tapered
welded I-section frame is optimised first; its result is compared with `peb_limits` (depth, plate thickness,
kg/m2, stiffener count, lambda_cr). With `truss.compare = always` (default) the portal truss is always optimised
too; with `on_trigger` only when a limit is crossed. Both get an installed cost per frame from the Cost rates
(material per kg by plate / tube, fabrication per kg + per stiffener + per truss joint, erection per kg,
painting per m2) - the rates shipped are PLACEHOLDERS, replace them. Truss (multigable only): top chord = roof
line (all roof loads unchanged), bottom chord `h0` below the eaves at the columns rising `bs` x the roof rise
(0 flat, 0.5, 1 parallel), verticals at every purlin station + ridge, one diagonal per panel (Pratt / Howe),
webs pin-ended (`MEMBER TRUSS`), chords continuous, the column continues between the chords (knee); leaning
interior columns are pin-ended between the chords and released at the bottom chord. Variables: h0 (400-2600),
bs, web type, and IS 4923 tubes for top / bottom / verticals / diagonals of each half-truss side (ext / int),
plus the column templates. Tubes: `tubes_is4923.json` from STAAD's `India (IS 4923-2017).db3` (same properties
STAAD uses; written as `TABLE 'SHS' ST 'SHS 100x100x4'`), t >= 3 mm, B >= 40 mm, B/t <= 35, and a chain in
which every heavier size has more area, no less Zp and r_min >= 0.98 x the best lighter one - a Pareto list
that is not monotone in r made the greedy search stall (a step up could weaken a strut). Checks (IS 800):
Table 2 class (RHS flange 42 eps, web 126 eps/(1+2 r2) >= 42 eps), 6.2 / 6.3 tension (welded: An = Ag), 7.1.2
compression curve b (Table 10, cold formed; STAAD has no cold-formed choice and uses a), 7.2.4 lengths (in plane
K x connection distance, `truss.K` 0.7-1.0; out of plane purlin spacing / every `fly_every`-th bottom chord point /
member length), 8.4.1.1 Av = A h/(b+h), 9.3.1 / 9.3.2.2, Table 3 KL/r 180 / 250 (compression only with wind) /
400 (tension only), and web/chord width 0.35-1.0 (CIDECT validity range). Not checked: RHS joint resistance and
welds (CIDECT DG3 / EN 1993-1-8 7.5), purlin cleats on the chord.
Demo 2 x 24 m (same loads): PEB 4122 kg, 42 stiffeners, lambda_cr 3.05 (crosses 2 limits) vs truss
3282 kg, h0 1500, bs 0.5, Pratt, lambda_cr 5.9; truss 5.7 % cheaper at the placeholder rates. Both STAAD-verified (demo/run_compare.py verify): PEB ours 0.999 / STAAD 0.827, truss ours 0.994 / STAAD 0.932, no FAIL; truss
truss: moments within 0.39 %, STAAD IS800 max 0.929 on all 142 tubes, no FAIL.

**Update: one grade, angle trusses, single-ridge trusses, crane in every scheme**
- One steel grade for all schemes: the Frame tab fy / fu apply to plates, IS 4923 tubes (order the YSt grade
  with yield >= fy) and IS 808 angles, so PEB and trusses are compared on the same material. One FYLD / FU in
  the STAAD file.
- `truss.family` = tube | angle | both (default both -> three schemes: PEB, Truss SHS/RHS, Truss 2-angles).
  Double equal angles back to back on a gusset (`truss.gusset_mm`), `angles_is808.json` from STAAD's
  `India (IS 808-2021).db3` (ISEA). Properties: in-plane I = 2 Izz, out-of-plane I = 2 [Iyy + A (c + tg/2)^2],
  Ze = 2 Zz. Checks: 7.5.2.1 webs KL in plane min(K, 0.85) L, out of plane L; 7.5.3 chords per 7.2.4; Table 10
  curve c; Table 2 axial class 3 (b + d)/t <= 25 eps (the catalogue holds only such angles at the chosen fy);
  6.2 yielding and 6.3.3 / 6.3.4 rupture per angle 0.9 Anc fu/gm1 + beta Ago fy/gm0 with welded ends and
  beta = 0.7 (its lower bound, no connection length needed); Md = Ze fy/gm0; Table 3 KL/r. Cost: angle rate,
  gusset kg per joint (`cost.gusset_kg`, added to the steel weight). Not checked: gussets, welds, block shear
  (6.4), tack connections (7.8, 10.2.5).
- STAAD 2026 gives `TABLE D` double angles (IS 808-2021 DB and the legacy INDIAN table, PLANE and SPACE) the
  stiffness of ONE angle - tested: 100 kN on a 1 m cantilever shortens 0.508 mm = PL/(E A_single). The STAAD
  file therefore gives double angles as `PRIS AX IZ IY IX` and code-checks only the other members
  (`CHECK CODE MEMB ...`); the double angles are checked by the optimiser. Verified: moments within 0.49 %,
  double-angle axial forces within 0.67 kN (0.17 % of max N).
- Single-ridge roofs take the truss too: bottom chord level computed at every column (interior columns meet
  it higher up), column-to-column pieces get Pratt / Howe diagonals falling toward their middle, sides 'ext'
  (piece touching an eave column) / 'int'.
- Crane: the Crane tab spans apply to every scheme (brackets must sit below the truss bottom chord, so h0 is
  limited by the bracket level). `python demo/run_compare.py crane [verify]` runs the three-scheme comparison
  with the 10 t crane in span 1.

Correction 2026-09-29: shear area of welded I-sections is d tw (8.4.1.1), was h tw.
Correction 2026-09-29: 8.4.2.2(a) tau_b for 0.8 < lambda_w < 1.2 is [1 - 0.8(lambda_w - 0.8)] fyw/sqrt3
(was 0.625 - from another code). Saved demo designs re-checked and still pass.

## Pipeline
1. `python demo_two_span.py demo` → `demo/PEB_2S_ULS_PDELTA.std`, `_SLS.std`, `_BUCKLING.std`, `station_map.json`, `model_summary.txt`, `frame_preview.png`
2. First time only: open each `.std` in STAAD, check the 3D rendering and the `.ANL` for errors.
3. `python openstaad_runner.py demo PEB_2S` → `results_ULS_stations.csv`, `results_ULS_statics.csv`, `results_SLS_stations.csv`, `results_SLS_displacements.csv`, `results_summary.json`

## Conventions
- Units METER KN. Frame in XY, Y up, BETA 0.
- Every sub-member starts at its deeper end (STAAD TR.20.3: TAPERED f1 ≥ f3). `reversed = true` in the map means start = upper grid station.
- `M_ic` > 0 = inner flange in compression. From STAAD local end forces: M(start) = −MZ1, M(end) = +MZ2, then × `sign_ic`.
- `N_comp` = FX(start) = −FX(end), + = compression.
- Member IDs: columns 10000 + 100·line + sub, rafters 20000 + 100·piece + sub (sub counted from base / left).
- Load cases: 1 DL (+ gantry), 2 collateral, 101+ LL per span, 201+ wind, 301/311+ notional, 401+/451+ virtual unit, 501+10k crane k, 1001+ ULS (REPEAT LOAD), 2001+ SLS (LOAD COMBINATION).
- Member IDs: brackets 30000 + 100·bracket + 1.

## Replace before use
- Wind Cpe (demo placeholders): walls and multispan roof tables of IS 875 (Part 3):2015.
- `ULS_RULES` / `SLS_RULES` factors: IS 800:2007 Table 4.
- Roof LL: IS 875 (Part 2) Table 2 for the actual slope and access.
- Seed sections.

## [VERIFY] on first run (version-dependent OpenSTAAD calls)
`OpenSTAADFile`, `AnalyzeEx(1,1,1)` / `Analyze`, `AreResultsAvailable`, `GetNodeDisplacements`, and the `.ANL` buckling-factor text format.
`results_ULS_statics.csv` must show `OK` for every primary case (catches unit and load-direction errors).

## Limitations
Centre-line model (no knee/apex eccentricities or rigid zones); single-ridge option is symmetric only; uniform pd with height; no crane or seismic cases.
