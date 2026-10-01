"""Build the deployable packages of the PEB Gable Optimiser (Windows).

    python build_exe.py          (needs: pip install pyinstaller)

dist/PEB_Gable_Optimiser/                 standalone app folder, no Python needed on the target PC
dist/PEB_Gable_Optimiser_win64.zip        the same folder zipped - copy, unzip, run PEB_Gable_Optimiser.exe
dist/PEB_Gable_Optimiser_source.zip       source + requirements.txt + run_app.bat (any PC with Python >= 3.12)
The frozen exe is self-tested (PEB_Gable_Optimiser.exe --selftest) before zipping.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import PyInstaller.__main__

HERE = Path(__file__).parent
NAME = "PEB_Gable_Optimiser"
DATA = ["tubes_is4923.json", "angles_is808.json"]
SOURCE = ["app.py", "optimiser.py", "peb_frame_model.py", "is800.py", "requirements.txt", "run_app.bat", "README.md",
          "test_optimiser.py", "test_model.py", "build_exe.py"] + DATA
GUIDE = """PEB Gable Optimiser - IS 800:2007 (tapered PEB frame vs portal truss)

Run:   PEB_Gable_Optimiser.exe            (keep the _internal folder next to it)
Check: PEB_Gable_Optimiser.exe --selftest  -> writes selftest_result.txt here (PASSED / FAILED)

STAAD.Pro is needed only for "Optimise + STAAD verify". The app finds SProStaad.exe under
C:\\Program Files\\Bentley\\...; if STAAD is installed elsewhere set the environment variable
PEB_STAAD_EXE to the full path of SProStaad.exe. Save STAAD files to a folder whose file names have no
spaces (the model name is the file name). Wind Cpe and the cost rates in the app are placeholders.
"""


def main():
    dist, work = HERE / "dist", HERE / "build"
    shutil.rmtree(dist / NAME, ignore_errors=True)
    PyInstaller.__main__.run([
        str(HERE / "app.py"), "--name", NAME, "--onedir", "--windowed", "--noconfirm", "--clean",
        "--distpath", str(dist), "--workpath", str(work), "--specpath", str(work),
        *[a for f in DATA for a in ("--add-data", f"{HERE / f};.")],
        "--exclude-module", "PyQt5", "--exclude-module", "PySide6", "--exclude-module", "IPython"])
    app = dist / NAME
    (app / "HOW_TO_RUN.txt").write_text(GUIDE)
    shutil.copy(HERE / "README.md", app / "README.md")
    r = subprocess.run([str(app / f"{NAME}.exe"), "--selftest"], timeout=900)
    result = (app / "selftest_result.txt").read_text()
    print(result)
    (app / "selftest_result.txt").unlink()
    if r.returncode:
        sys.exit("frozen selftest FAILED - not zipping")
    for zname, files in ((f"{NAME}_win64.zip", [p for p in app.rglob("*") if p.is_file()]),
                         (f"{NAME}_source.zip", [HERE / f for f in SOURCE])):
        root = app.parent if zname.endswith("win64.zip") else HERE
        with zipfile.ZipFile(dist / zname, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, Path(NAME) / p.relative_to(root) if root == HERE else p.relative_to(root))
        print(f"{dist / zname}  {(dist / zname).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
