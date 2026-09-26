"""Guard against running against a STALE game mod.

The probe DLL is built from probe/src and copied into the game's mods directory. Nothing
forces a rebuild, so editing a source file and not rebuilding leaves the game running old
code -- silently, because the mod loads and works fine, it is just not the code you edited.

That cost a full day of comparisons: RunServe.cs and RunHarness.cs were edited at 11:29 and
the DLL was not rebuilt until 18:17, so every measurement taken in between ran against a
different game binary than the source described. The rebuild then changed the run-level
observations underneath a trained policy, and the same checkpoint went from 0.550 to 0.167
with no code change on the Python side at all.
"""
from __future__ import annotations

import os
import glob

DEFAULT_GAME = os.path.expanduser(
    "~/.local/share/Steam/steamapps/common/Slay the Spire 2")


def check(game_dir=DEFAULT_GAME, src_dir=None, fatal=False):
    """Returns a list of source files newer than the installed DLL."""
    dll = os.path.join(game_dir, "mods", "alphaslayer_probe", "alphaslayer_probe.dll")
    if src_dir is None:
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               os.pardir, "probe", "src")
    src_dir = os.path.abspath(src_dir)
    if not os.path.exists(dll):
        msg = f"probe DLL not installed at {dll}"
        if fatal:
            raise RuntimeError(msg)
        print(f"  WARNING: {msg}", flush=True)
        return []
    dll_t = os.path.getmtime(dll)
    stale = [f for f in glob.glob(os.path.join(src_dir, "*.cs"))
             if os.path.getmtime(f) > dll_t]
    if stale:
        names = ", ".join(sorted(os.path.basename(f) for f in stale))
        msg = (f"STALE MOD: {len(stale)} probe source file(s) are newer than the installed "
               f"DLL ({names}). The game is running code that does not match the source. "
               f"Rebuild with probe/run.sh before measuring anything.")
        if fatal:
            raise RuntimeError(msg)
        print(f"  WARNING: {msg}", flush=True)
    return stale
