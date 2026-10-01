"""
app.py - PEB Gable Optimiser (IS 800:2007) desktop app.

    python app.py

Left: inputs (frame, loads, seed sections, limits). Right: frame drawing coloured by D/C, report, log.
Buttons: Check seed (one analysis), Optimise (fast engine), Optimise + STAAD verify (headless SProStaad,
P-Delta + STAAD IS800 CHECK CODE), Export .std (STAAD files of the current design).
"""
from __future__ import annotations

import copy
import json
import multiprocessing
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import optimiser as opt                                # sets BLAS threads before numpy loads
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

from peb_frame_model import FrameModel  # noqa: E402

FROZEN = getattr(sys, "frozen", False)                 # packaged exe (PyInstaller)
HERE = Path.home() / "Documents" if FROZEN else Path(__file__).parent   # file dialogs start here

# (json path, label, kind)  kind: f=float, i=int, s=string, l=list of floats, c=choice:<a|b>, b=bool
FIELDS = {
    "Frame": [("name", "Model name", "s"), ("spans", "Spans (m, comma list)", "l"),
              ("eave_height", "Eave height (m)", "f"), ("slope", "Roof slope (rise/run)", "f"),
              ("roof", "Roof", "c:multigable|single_ridge"), ("bay", "Bay / frame spacing (m)", "f"),
              ("purlin_spacing", "Purlin spacing (m)", "f"), ("girt_spacing", "Girt spacing (m)", "f"),
              ("interior_columns", "Interior columns", "c:leaning|rigid"),
              ("base_ext", "Exterior bases", "c:PINNED|FIXED"), ("base_int", "Interior bases", "c:PINNED|FIXED"),
              ("fy", "fy (MPa) - ALL schemes: plates, tubes, angles", "f"), ("fu", "fu (MPa) - all schemes", "f"), ("restraint.fly_every", "Fly brace at every n-th purlin/girt (0 = none)", "i"),
              ("purlin.b1_mm", "Purlin stiff bearing length b1 (mm, 8.7.1.3)", "f"),
              ("purlin.web_KL", "Web strut KL/d under purlins (8.7.1.5: 0.7 / 1.0)", "f")],
    "Loads": [("loads.q_sdl_slope", "SDL on slope (kN/m2)", "f"), ("loads.q_coll_plan", "Collateral, plan (kN/m2)", "f"),
              ("loads.q_ll_plan", "Roof live, plan (kN/m2)", "f"), ("loads.q_wall", "Wall cladding (kN/m2)", "f"),
              ("loads.sw_factor", "Self-weight factor", "f"),
              ("wind.Vb", "Vb (m/s)", "f"), ("wind.k1", "k1", "f"), ("wind.k2", "k2", "f"), ("wind.k3", "k3", "f"),
              ("wind.k4", "k4", "f"), ("wind.Kd", "Kd", "f"), ("wind.Ka", "Ka", "f"), ("wind.Kc", "Kc", "f"),
              ("wind.cpi", "Cpi values (comma list)", "l"),
              ("wind.cpe_verified", "Cpe taken from IS 875-3 tables", "b")],
    "Limits": [("limits.dc_target", "Strength D/C target", "f"),
               ("limits.rafter_defl", "Rafter deflection limit span/", "f"),
               ("limits.eave_drift", "Eave drift limit H/", "f"),
               ("options.slender_web", "Allow slender webs (IS 800 8.2.1.1(a), 8.6.1)", "b"),
               ("options.stiffeners", "Transverse web stiffeners where needed (8.4.2.2(a), 8.7.2)", "b"),
               ("options.stiffener_penalty_kg", "Stiffener fabrication penalty (kg per stiffener)", "f"),
               ("options.tension_field", "Tension field method, anchored panels (8.4.2.2(b), 8.5.1)", "b"),
               ("options.end_posts", "End posts at knee / haunch panels (8.5.2(b))", "b"),
               ("options.constant_bf", "Practical: one flange width per member", "b"),
               ("options.rafter_six_sections", "Rafter: six sections from the moment envelope (no apex spike)", "b"),
               ("limits.crane_drift", "Crane + wind drift at rail, H/ (Table 6: 200 pendent, 400 cab)", "f"),
               ("limits.rail_spread_mm", "Rail gauge change limit (mm, Table 6)", "f"),
               ("limits.taper_max", "Max rafter taper outside the haunch (mm depth per m)", "f"),
               ("limits.time_limit_min", "Optimiser time limit per scheme (min, 0 = none)", "f"),
               ("options.screen_combos", "Fast search: screen with the governing combinations (all checked on accept)", "b")],
    "Crane": [("crane.spans", "Crane in spans (1-based list, blank = none)", "l"),
              ("crane.bracket_level", "Bracket (gantry seat) level above base (m)", "f"),
              ("crane.e", "Rail from column centre = bracket length (m)", "f"),
              ("crane.gantry_depth", "Gantry depth, seat to rail top (m)", "f"),
              ("crane.capacity_kN", "Capacity SWL (kN)", "f"), ("crane.bridge_kN", "Crane bridge weight (kN)", "f"),
              ("crane.crab_kN", "Crab / trolley weight (kN)", "f"), ("crane.a_min", "Min hook approach (m)", "f"),
              ("crane.wheel_base", "End-carriage wheel base (m)", "f"),
              ("crane.wheels_per_rail", "Wheels per rail", "i"),
              ("crane.gantry_kNm", "Gantry girder + rail self-weight (kN/m)", "f"),
              ("crane.impact", "Vertical impact, IS 875-2 6.3(a) (0.25 / 0.10)", "f"),
              ("crane.surge", "Surge fraction, 6.3(c) (0.05; 0.10 rigid mast)", "f")],
    "Truss / Cost": [("truss.compare", "Portal-truss alternative", "c:always|on_trigger"),
                     ("truss.family", "Truss members", "c:both|tube|angle"),
                     ("truss.gusset_mm", "Double angles: gusset thickness (mm)", "f"),
                     ("truss.K", "Truss in-plane KL / connection distance (7.2.4: 0.7-1.0)", "f"),
                     ("truss.fly_every", "Bottom chord restrained every n-th panel point", "i"),
                     ("peb_limits.D_max_mm", "PEB limit: max depth (mm)", "f"),
                     ("peb_limits.plate_max_mm", "PEB limit: max plate tw / tf (mm)", "f"),
                     ("peb_limits.kg_m2_max", "PEB limit: max kg/m2 plan", "f"),
                     ("peb_limits.stiffeners_max", "PEB limit: max stiffeners per frame", "f"),
                     ("peb_limits.lam_cr_min", "PEB limit: min lambda_cr", "f"),
                     ("cost.plate_rs_kg", "Rate: plate steel (Rs/kg) PLACEHOLDER", "f"),
                     ("cost.tube_rs_kg", "Rate: IS 4923 tube (Rs/kg) PLACEHOLDER", "f"),
                     ("cost.angle_rs_kg", "Rate: IS 808 angle (Rs/kg) PLACEHOLDER", "f"),
                     ("cost.gusset_kg", "Angle truss: gusset steel per joint (kg)", "f"),
                     ("cost.fab_builtup_rs_kg", "Rate: built-up fabrication (Rs/kg)", "f"),
                     ("cost.fab_truss_rs_kg", "Rate: truss fabrication (Rs/kg)", "f"),
                     ("cost.stiffener_rs", "Rate: per web stiffener (Rs)", "f"),
                     ("cost.joint_rs", "Rate: per truss joint (Rs)", "f"),
                     ("cost.erect_rs_kg", "Rate: erection (Rs/kg)", "f"),
                     ("cost.paint_rs_m2", "Rate: painting (Rs/m2)", "f")],
}


def get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def put(d, path, v):
    ks = path.split(".")
    for k in ks[:-1]:
        d = d[k]
    d[ks[-1]] = v


def floats(s):
    return [float(x) for x in s.replace(";", ",").split(",") if x.strip()]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PEB Gable Optimiser - IS 800:2007")
        self.geometry("1400x860")
        self.inp = copy.deepcopy(opt.DEFAULT_INPUT)
        self.design, self.result, self.out = None, None, None
        self.q, self.stop = queue.Queue(), threading.Event()
        self.vars = {}
        self._build()
        self._to_form()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self):
        left = ttk.Frame(self, padding=6)
        left.pack(side="left", fill="y")
        nb = ttk.Notebook(left, width=430)
        nb.pack(fill="both", expand=True)
        for tab, fields in FIELDS.items():
            f = ttk.Frame(nb, padding=6)
            nb.add(f, text=tab)
            for r, (path, label, kind) in enumerate(fields):
                ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2)
                if kind == "b":
                    v = tk.BooleanVar()
                    ttk.Checkbutton(f, variable=v).grid(row=r, column=1, sticky="w")
                elif kind.startswith("c:"):
                    v = tk.StringVar()
                    ttk.Combobox(f, textvariable=v, values=kind[2:].split("|"), state="readonly",
                                 width=16).grid(row=r, column=1, sticky="w")
                else:
                    v = tk.StringVar()
                    ttk.Entry(f, textvariable=v, width=22).grid(row=r, column=1, sticky="w")
                self.vars[path] = (v, kind)
            if tab == "Loads":
                r = len(fields)
                ttk.Label(f, text="Wind Cpe, one case per line:\nlabel | wall_L wall_R | roof piece 1 2 ... (left->right)"
                          ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(8, 2))
                self.cpe = tk.Text(f, width=52, height=6, font=("Consolas", 9))
                self.cpe.grid(row=r + 1, column=0, columnspan=2, sticky="we")
        f = ttk.Frame(nb, padding=6)
        nb.add(f, text="Sections")
        ttk.Label(f, text="Seed sections (mm). Knots = s/L from the template start.\n"
                          "Plates per segment: tw x bf x tf ; tw x bf x tf ...", justify="left").pack(anchor="w")
        self.tpl = {}
        for t, title in opt.TEMPLATE_NAMES.items():
            if t == "truss":
                continue
            lf = ttk.LabelFrame(f, text=title, padding=4)
            lf.pack(fill="x", pady=3)
            row = {}
            for c, (k, w) in enumerate((("knots", 16), ("D", 18), ("plates", 34))):
                ttk.Label(lf, text=k).grid(row=c, column=0, sticky="w")
                v = tk.StringVar()
                ttk.Entry(lf, textvariable=v, width=w + 12).grid(row=c, column=1, sticky="w")
                row[k] = v
            self.tpl[t] = row

        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=6)
        for text, cmd in (("Load...", self.load), ("Save...", self.save), ("Check seed", self.check_seed),
                          ("Optimise", lambda: self.run(False)), ("Optimise + STAAD verify", lambda: self.run(True)),
                          ("Stop", self.stop.set), ("Export .std...", self.export)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2, pady=2)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status, foreground="#1f4e79").pack(anchor="w")

        right = ttk.PanedWindow(self, orient="vertical")
        right.pack(side="right", fill="both", expand=True)
        self.fig = Figure(figsize=(9, 3.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        right.add(self.canvas.get_tk_widget(), weight=3)
        tabs = ttk.Notebook(right)
        right.add(tabs, weight=2)
        self.report = ScrolledText(tabs, font=("Consolas", 9))
        self.log = ScrolledText(tabs, font=("Consolas", 9))
        tabs.add(self.report, text="Report")
        tabs.add(self.log, text="Log")
        self.tabs = tabs

    # ------------------------------------------------------------------ form <-> dict
    def _to_form(self):
        for path, (v, kind) in self.vars.items():
            val = get(self.inp, path)
            v.set(", ".join(f"{x:g}" for x in val) if kind == "l" else val if kind in ("b",) else str(val))
        self.cpe.delete("1.0", "end")
        for c in self.inp["wind"]["cases"]:
            self.cpe.insert("end", f"{c['label']} | {' '.join(f'{x:g}' for x in c['walls'])} | "
                                   f"{' '.join(f'{x:g}' for x in c['roof'])}\n")
        for t, row in self.tpl.items():
            d = self.inp["templates"][t]
            row["knots"].set(", ".join(f"{x:g}" for x in d["knots"]))
            row["D"].set(", ".join(str(x) for x in d["D"]))
            row["plates"].set(" ; ".join("x".join(str(v) for v in p) for p in d["plates"]))

    def _from_form(self) -> dict:
        inp = copy.deepcopy(self.inp)
        for path, (v, kind) in self.vars.items():
            s = v.get()
            val = (float(s) if kind == "f" else int(float(s)) if kind == "i" else floats(s) if kind == "l"
                   else bool(s) if kind == "b" else s.strip())
            put(inp, path, val)
        cases = []
        for ln in self.cpe.get("1.0", "end").splitlines():
            if ln.strip():
                label, walls, roof = (p.strip() for p in ln.split("|"))
                cases.append(dict(label=label, walls=floats(walls.replace(" ", ",")),
                                  roof=floats(roof.replace(" ", ","))))
        inp["wind"]["cases"] = cases
        for t, row in self.tpl.items():
            plates = [[int(float(v)) for v in p.lower().split("x")] for p in row["plates"].get().split(";") if p.strip()]
            inp["templates"][t] = dict(knots=floats(row["knots"].get()),
                                       D=[int(float(x)) for x in floats(row["D"].get())], plates=plates)
        n = opt.n_roof_pieces(inp)
        for c in cases:
            if len(c["walls"]) != 2 or len(c["roof"]) != n:
                raise ValueError(f"Cpe case '{c['label']}': need 2 wall values and {n} roof values "
                                 f"(one per roof piece, left to right)")
        return inp

    def _read(self):
        try:
            self.inp = self._from_form()
            return True
        except Exception as ex:
            messagebox.showerror("Input error", str(ex))
            return False

    # ------------------------------------------------------------------ actions
    def load(self):
        p = filedialog.askopenfilename(initialdir=HERE, filetypes=[("JSON", "*.json")])
        if p:
            d = json.loads(Path(p).read_text())
            self.inp = d.get("input", d)
            for k, v in opt.DEFAULT_INPUT.items():                # files saved before newer options
                self.inp.setdefault(k, copy.deepcopy(v))
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        self.inp[k].setdefault(k2, copy.deepcopy(v2))
            self.design = d.get("design")
            self._to_form()

    def save(self):
        if not self._read():
            return
        p = filedialog.asksaveasfilename(initialdir=HERE, defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            opt.save_json(p, dict(input=self.inp, design=self.design))

    def check_seed(self):
        if not self._read():
            return
        out = dict(inputs={}, triggers=[])
        keys = ["peb"] + (opt.truss_keys(self.inp) if self.inp["truss"]["compare"] == "always" else [])
        for k in keys:                                                  # seed of each scheme, links + catalogue
            ins = opt.scheme_input(self.inp, k)
            design = opt.compatible(ins, ins["templates"])
            r = opt.evaluate(ins, design)
            if r.get("error"):
                messagebox.showerror("Analysis", f"{k}: {r['error']}")
                return
            out["inputs"][k], out[k] = ins, dict(design=design, result=r)
        out["cost"] = {k: opt.installed_cost(out["inputs"][k], out[k]["result"]) for k in keys}
        self._show(out)
        self.status.set("Seed: " + ", ".join(f"{k} {out[k]['result']['weight']:.0f} kg D/C {out[k]['result']['max_ratio']:.3f}"
                                            for k in keys))

    def run(self, verify: bool):
        if not self._read():
            return
        folder = None
        if verify:
            folder = filedialog.askdirectory(initialdir=HERE, title="Folder for STAAD files (no spaces needed)")
            if not folder:
                return
        self.stop.clear()
        self.log.delete("1.0", "end")
        self.tabs.select(self.log)
        self.status.set("Optimising ...")
        inp = copy.deepcopy(self.inp)

        def work():
            try:
                log = lambda s: self.q.put(("log", s))  # noqa: E731
                out = opt.optimise_both(inp, Path(folder) if verify else None, log=log, stop=self.stop.is_set)
                self.q.put(("done", out))
            except Exception as ex:
                self.q.put(("error", repr(ex)))
        threading.Thread(target=work, daemon=True).start()

    def export(self):
        if not self.design:
            messagebox.showinfo("Export", "Run Check seed or Optimise first.")
            return
        folder = filedialog.askdirectory(initialdir=HERE, title="Export folder")
        if not folder:
            return
        folder = Path(folder)
        try:                                    # analysis + design parameters (IS800 LSD block) per scheme
            for k, ins in self.out["inputs"].items():
                opt.export_std(ins, self.out[k]["design"], folder / k.upper(), log=lambda s: None)
        except Exception as ex:
            messagebox.showerror("Export", str(ex))
            return
        (folder / f"{self.inp['name']}_report.txt").write_text(self._report_text(self.out))
        self.status.set(f"Exported to {folder}")

    # ------------------------------------------------------------------ results
    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", val + "\n")
                    self.log.see("end")
                elif kind == "done":
                    self._show(val)
                    self.status.set("Done: " + " | ".join(
                        f"{k} {c['kg']:.0f} kg, Rs {c['total']:,.0f}" for k, c in val["cost"].items()))
                else:
                    self.status.set("Error")
                    messagebox.showerror("Optimiser", val)
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def _report_text(self, out):
        return (opt.compare_report(out) + "\n\n" if out.get("cost") else "") + "\n\n".join(
            opt.report(out["inputs"][k], out[k]) for k in out["inputs"])

    def _show(self, out):
        """out: optimise_both() result (or the same shape from Check seed): one or two schemes."""
        self.out, self.design, self.result = out, out["peb"]["design"], out["peb"]["result"]
        self.report.delete("1.0", "end")
        self.report.insert("end", self._report_text(out))
        self.tabs.select(self.report)
        self.fig.clear()
        ks = list(out["inputs"])
        for i, k in enumerate(ks):
            self._draw(out[k]["result"], self.fig.add_subplot(len(ks), 1, i + 1), f"{opt.SCHEME_NAMES[k]}: ")
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw(self, r, ax, prefix=""):
        mdl = r["model"]
        dc = {}
        for x in r["rows"]:
            if x["member"]:
                dc[x["member"]] = max(dc.get(x["member"], 0), x["ratio"])
        norm, cmap, seen = mcolors.Normalize(0, 1.2), matplotlib.colormaps["RdYlGn_r"], {}
        for ln in mdl.lines:
            nx, ny = ln["outer_normal"]
            for mid in ln["members"]:
                m = mdl.mem_index[mid]
                a, b = ln["stations"][m["j"]], ln["stations"][m["j"] + 1]
                pts = [(a["x"] + nx * a["D"] / 2, a["y"] + ny * a["D"] / 2), (b["x"] + nx * b["D"] / 2, b["y"] + ny * b["D"] / 2),
                       (b["x"] - nx * b["D"] / 2, b["y"] - ny * b["D"] / 2), (a["x"] - nx * a["D"] / 2, a["y"] - ny * a["D"] / 2)]
                ax.add_patch(Polygon(pts, closed=True, fc=cmap(norm(dc.get(mid, 0))), ec="k", lw=0.3))
            for k in ([] if mdl.mem_index[ln["members"][0]].get("sec") else ln["knots"]):    # "875/975"; not trusses
                st = ln["stations"][k["j"]]
                lab = seen.setdefault((round(st["x"], 2), round(st["y"], 2)), dict(D=[], off=(nx, ny, ln["type"])))
                if round(1000 * k["D"]) not in lab["D"]:
                    lab["D"].append(round(1000 * k["D"]))
        for (x, y), lab in seen.items():
            nx, ny, typ = lab["off"]
            ax.annotate("/".join(map(str, lab["D"])), (x, y), fontsize=7, ha="center",
                        xytext=(nx * 14, ny * 14 + (6 if typ == "raf" else 0)), textcoords="offset points")
        for it in r.get("stiffeners", {}).get("items", []):            # web stiffeners across the depth
            nx, ny = next(ln["outer_normal"] for ln in mdl.lines if ln["name"] == it["line"])
            h = it["D"] / 2
            ax.plot([it["x"] - nx * h, it["x"] + nx * h], [it["y"] - ny * h, it["y"] + ny * h], color="#1f4e79", lw=0.8)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_title(f"{prefix}{r['weight']:.0f} kg  |  max D/C {r['max_ratio']:.3f}  |  lambda_cr (gravity) "
                     f"{r['lam_cr_gravity']:.2f}   (true-scale depth, knot depths in mm)", fontsize=9, pad=18)
        self.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.02, shrink=0.6, label="D/C")


def selftest(path: Path) -> bool:
    """Deployment check (app --selftest): data files, one analysis per scheme, a 2-worker optimisation (process
    pool in the packaged exe) and the STAAD engine lookup. Writes the result to path; True if it passed."""
    lines, ok = [], True
    try:
        inp = copy.deepcopy(opt.DEFAULT_INPUT)
        lines.append(f"catalogues: {len(opt.TUBES)} tubes, {len(opt.ANGLE1)} angles")
        for k in ["peb"] + opt.truss_keys(inp):
            ins = opt.scheme_input(inp, k)
            r = opt.evaluate(ins, opt.compatible(ins, ins["templates"]))
            ok &= not r.get("error")
            lines.append(f"{k}: seed {r['weight']:.0f} kg, max D/C {r['max_ratio']:.3f} {r.get('error') or ''}")
        s = copy.deepcopy(inp)
        s.update(name="SELFTEST", spans=[15.0], eave_height=6.0)
        s["wind"].update(cases=[dict(label="+X", walls=[0.7, -0.3], roof=[-0.9, -0.4])], cpi=[0.2])
        s["templates"]["raf_ext"] = dict(knots=[0, 0.5, 1], D=[700, 450, 450], plates=[[8, 250, 14], [6, 200, 10]])
        s["templates"]["col_ext"] = dict(knots=[0, 1], D=[400, 700], plates=[[8, 250, 14]])
        out = opt.optimise(s, log=lambda _: None, shake_passes=0, workers=2)
        ok &= out["result"]["ok"]
        lines.append(f"optimise (2 worker processes): {out['result']['weight']:.0f} kg, feasible {out['result']['ok']}")
        try:
            lines.append(f"STAAD engine: {opt.engine()}")
        except FileNotFoundError as ex:
            lines.append(f"STAAD engine: not found ({ex}) - fast engine and export still work")
    except Exception as ex:                              # report, don't crash, on a new machine
        ok = False
        lines.append(f"ERROR {ex!r}")
    lines.append("SELFTEST " + ("PASSED" if ok else "FAILED"))
    path.write_text("\n".join(lines) + "\n")
    return ok


if __name__ == "__main__":
    multiprocessing.freeze_support()                     # packaged exe: worker processes must not start the GUI
    if "--selftest" in sys.argv:
        out = Path(sys.executable).with_name("selftest_result.txt") if FROZEN else HERE / "selftest_result.txt"
        sys.exit(0 if selftest(out) else 1)
    App().mainloop()
