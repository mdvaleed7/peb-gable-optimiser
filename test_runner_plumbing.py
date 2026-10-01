"""Tests the runner's data path with a mock STAAD (the independent linear solver)."""
import json, tempfile
from pathlib import Path
import demo_two_span as d                      # builds demo model + files in ./demo
from openstaad_runner import station_rows, statics_check, buckling_factors

smap = json.loads(Path("demo/station_map.json").read_text())   # through JSON: string keys etc.
combos = {c["lc"]: c["pairs"] for c in smap["uls"] + smap["sls"]}

class Mock:
    def __init__(s): s.cache = {}
    def _res(s, lc):
        if lc not in s.cache:
            s.cache[lc] = d.fe.solve(dict(combos.get(lc, [(lc, 1.0)])))
        return s.cache[lc]
    def end_forces(s, mid, lc): return s._res(lc)["ends"][mid]
    def reactions(s, nd, lc):
        r = s._res(lc)["R"][nd]; return [r[0], r[1], 0, 0, 0, r[2]]

mk = Mock()
rows = station_rows(mk, smap, [1001])
ref = d.fe.stations(d.fe.solve(dict(combos[1001])))
got = {(r["line"], r["j"]): r["M_ic"] for r in rows}
err = max(abs(got[(ln, s["j"])] - v) for ln, sts in ref.items() for s in sts for v in s["M"])
assert err < 1e-9, err
print(f"station_rows: {len(rows)} rows for U01; max |dM| vs solver = {err:.1e}")
chk = statics_check(mk, smap, [1, 2, 101, 102, 201, 206, 301, 311])
assert all(r["status"] == "OK" for r in chk), chk
print("statics_check: all primaries OK (ratios", sorted({round(r['ratioY'] or 1, 6) for r in chk}), ")")
fake = """  LOAD CASE    1001   U01 1.5(D+L) L[ALL] N+X
     ...
     BUCKLING FACTOR =      9.8765
  LOAD CASE    1002   U02
     THE BUCKLING FACTOR FOR THIS CASE IS  0.4321D+01
"""
p = Path(tempfile.mkdtemp()) / "x.ANL"; p.write_text(fake)
bf = buckling_factors(p)
assert [b["load_case_above"] for b in bf] == [1001, 1002] and abs(bf[1]["numbers"][-1] - 4.321) < 1e-9
print("buckling_factors: parsed", [(b["load_case_above"], b["numbers"][-1]) for b in bf])
print("RUNNER PLUMBING OK")
