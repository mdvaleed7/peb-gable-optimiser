"""
openstaad_runner.py
===================
Open the generated .std files in a RUNNING STAAD.Pro, analyse them, and pull results through
OpenSTAAD into station tables keyed by station_map.json.

    python openstaad_runner.py <folder> <model_name>        e.g.  python openstaad_runner.py demo PEB_2S

Requirements: Windows, STAAD.Pro running, `pip install comtypes`.
Status: the data plumbing below was tested against a mock (the independent linear solver);
the COM calls themselves were NOT executed where this file was written (no STAAD there).
Calls tagged [VERIFY] vary between STAAD versions - confirm them once on your install.

Calling pattern per Bentley's "Write an OpenSTAAD Program in Python" (comtypes,
GetActiveObject, _FlagAsMethod, SAFEARRAY passed by reference). Signatures used:
    Output.GetMemberEndForces(member, end 0=start|1=end, loadcase, forces[6], 0=local|1=global)
    Output.GetSupportReactions(node, loadcase, reactions[6])
"""
from __future__ import annotations

import csv
import ctypes
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

from peb_frame_model import end_forces_to_stations


# ----------------------------------------------------------------------------- COM session
class Staad:
    ROOT = ("OpenSTAADFile", "SetSilentMode", "Analyze", "AnalyzeEx", "GetSTAADFile")
    OUT = ("GetMemberEndForces", "GetSupportReactions", "GetNodeDisplacements", "AreResultsAvailable")

    def __init__(self):
        from comtypes import automation, client
        self._auto = automation
        self.os = client.GetActiveObject("StaadPro.OpenSTAAD")
        self.out = self.os.Output
        for n in self.ROOT:
            self._flag(self.os, n)
        for n in self.OUT:
            self._flag(self.out, n)

    @staticmethod
    def _flag(obj, name):
        try:
            obj._FlagAsMethod(name)
        except Exception:
            pass

    def _array(self, n):
        sa = self._auto._midlSAFEARRAY(ctypes.c_double).create([0.0] * n)
        v = self._auto.VARIANT()
        v._.c_void_p = ctypes.addressof(sa)
        v.vt = self._auto.VT_ARRAY | self._auto.VT_R8 | self._auto.VT_BYREF
        return sa, v

    @staticmethod
    def _values(v) -> List[float]:
        return [float(x) for x in v[0]]

    def open_and_analyze(self, std: Path, timeout: float = 900.0) -> Path:
        std = Path(std).resolve()
        anl = std.with_suffix(".ANL")
        t0 = time.time()
        self.os.OpenSTAADFile(str(std))                      # [VERIFY] root method name
        try:
            self.os.SetSilentMode(1)                         # no confirmation dialogs
        except Exception:
            pass
        try:
            self.os.AnalyzeEx(1, 1, 1)                       # [VERIFY] silent, hidden, wait
        except Exception:
            self.os.Analyze()                                # older builds
        while time.time() - t0 < timeout:                    # wait for fresh results
            try:
                if self.out.AreResultsAvailable():           # [VERIFY]
                    return anl
            except Exception:
                if anl.exists() and anl.stat().st_mtime > t0 and time.time() - anl.stat().st_mtime > 3:
                    return anl
            time.sleep(1.0)
        raise TimeoutError(f"no analysis results for {std.name} after {timeout:.0f} s")

    def end_forces(self, mid: int, lc: int):
        res = []
        for end in (0, 1):
            _sa, v = self._array(6)
            self.out.GetMemberEndForces(int(mid), end, int(lc), v, 0)   # local axes
            res.append(self._values(v))
        return res

    def reactions(self, node: int, lc: int) -> List[float]:
        _sa, v = self._array(6)
        self.out.GetSupportReactions(int(node), int(lc), v)
        return self._values(v)

    def displacements(self, node: int, lc: int) -> List[float]:
        _sa, v = self._array(6)
        self.out.GetNodeDisplacements(int(node), int(lc), v)          # [VERIFY] name/units
        return self._values(v)


# ----------------------------------------------------------------------------- extraction
def station_rows(src, smap: dict, lcs: Iterable[int]) -> List[dict]:
    """src: any object with end_forces(member, lc) -> (start[6], end[6]) in STAAD local axes."""
    lines = {l["idx"]: l for l in smap["lines"]}
    rows = []
    for lc in lcs:
        for m in smap["members"]:
            ln = lines[m["line"]]
            fs, fe = src.end_forces(m["id"], lc)
            for j, N, V, M in end_forces_to_stations(m, fs, fe):
                st = ln["stations"][j]
                rows.append(dict(lc=lc, line=ln["name"], j=j, node=st["node"], x=round(st["x"], 4),
                                 y=round(st["y"], 4), s=round(st["s"], 4), D_mm=round(1000 * st["D"], 1),
                                 member=m["id"], N_comp=N, V_local=V, M_ic=M))
    return rows


def statics_check(src, smap: dict, lcs: Iterable[int], tol: float = 0.01) -> List[dict]:
    """Sum of support reactions vs the generator's load totals. Catches unit and load-direction
    errors. DL self-weight is the generator's own estimate, so allow ~1 % there."""
    out = []
    for lc in lcs:
        exp = smap["expected_totals"][str(lc)]
        rx = ry = 0.0
        for nd in smap["supports"]:
            r = src.reactions(int(nd), lc)
            rx, ry = rx + r[0], ry + r[1]
        res = {}
        for k, got, want in (("X", -rx, exp["FX"]), ("Y", -ry, exp["FY"])):
            ratio = got / want if abs(want) > 1e-6 else None
            res[k] = ratio
        ok = all(r is None or abs(r - 1) < tol for r in res.values())
        flag = "OK" if ok else ("UNITS? (x1000)" if any(r and abs(abs(r) - 1000) < 50 for r in res.values())
                                else "CHECK")
        out.append(dict(lc=lc, sumRX=rx, sumRY=ry, expFX=exp["FX"], expFY=exp["FY"],
                        ratioX=res["X"], ratioY=res["Y"], status=flag))
    return out


def buckling_factors(anl: Path) -> List[dict]:
    """Tolerant .ANL scan: every line mentioning a buckling factor, with the numbers on it
    and the last load-case number seen above it. [VERIFY] against your .ANL layout."""
    rows, last_lc = [], None
    for line in Path(anl).read_text(errors="ignore").splitlines():
        u = line.upper()
        m = re.search(r"LOAD(?:ING)?\s*(?:CASE)?\s*(?:NO\.?)?\s*(\d{1,6})", u)
        if m:
            last_lc = int(m.group(1))
        if "BUCKLING" in u and "FACTOR" in u:
            nums = re.findall(r"[-+]?\d*\.\d+(?:[ED][-+]?\d+)?|[-+]?\d+(?:[ED][-+]?\d+)", u)
            rows.append(dict(load_case_above=last_lc, text=line.strip(),
                             numbers=[float(n.replace("D", "E")) for n in nums]))
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- driver
def main(folder: str, name: str) -> None:
    folder = Path(folder)
    smap = json.loads((folder / "station_map.json").read_text())
    groups: Dict[str, List[int]] = {}
    for lc, c in smap["cases"].items():
        groups.setdefault(c["group"], []).append(int(lc))
    prim = sorted(groups.get("primary", []))
    virt = sorted(groups.get("virtual", []))
    st = Staad()
    report = {}

    # 1) ULS, P-Delta: stations for every REPEAT LOAD combination + statics on primaries
    st.open_and_analyze(folder / f"{name}_ULS_PDELTA.std")
    uls = [c["lc"] for c in smap["uls"]]
    write_csv(folder / "results_ULS_stations.csv", station_rows(st, smap, uls))
    chk = statics_check(st, smap, prim)
    write_csv(folder / "results_ULS_statics.csv", chk)
    report["uls_statics_ok"] = all(r["status"] == "OK" for r in chk)

    # 2) SLS, linear: stations for SLS combos and virtual cases; key displacements
    st.open_and_analyze(folder / f"{name}_SLS.std")
    sls = [c["lc"] for c in smap["sls"]]
    write_csv(folder / "results_SLS_stations.csv", station_rows(st, smap, sls + virt))
    keys = sorted({v["node"] for v in smap["virtual"].values() if "node" in v} |
                  {n for v in smap["virtual"].values() for n in [v.get("mid_node")] + v.get("support_nodes", []) if n})
    disp = []
    for lc in sls + virt:
        for nd in keys:
            try:
                d = st.displacements(nd, lc)
                disp.append(dict(lc=lc, node=nd, ux=d[0], uy=d[1], rz=d[5]))
            except Exception as e:                          # keep going; report once
                report["displacement_error"] = repr(e)
                break
    write_csv(folder / "results_SLS_displacements.csv", disp)

    # 3) Buckling: factors from the .ANL
    anl = st.open_and_analyze(folder / f"{name}_BUCKLING.std")
    bf = buckling_factors(anl)
    report["buckling_lines"] = bf
    (folder / "results_summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "buckling_lines"}, indent=1))
    print(f"{len(bf)} buckling-factor lines found in {anl.name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
