#!/usr/bin/env bash
# Builds a PRIVATE game data directory for headless training and prints the XDG_DATA_HOME to use.
#
# Why: the game loads every subscribed Steam Workshop mod into the headless probe process too.
# One of them subscribes to CombatStateTracker.CombatStateChanged, which the game forbids under
# TestMode ("Backend should not be subscribing to CombatStateChanged!"), so every fight failed
# at setup and every run failed at its first travel (09-14). Two of the mods also declare
# affects_gameplay=true. The game's per-user settings.save carries an is_enabled flag per mod,
# but that file is shared with normal play, so instead the headless game gets its own copy of
# ~/.local/share/SlayTheSpire2 (Godot honours XDG_DATA_HOME) with Workshop mods disabled there.
# Profile/progress are copied so the unlock state matches the real profile.
set -euo pipefail
SRC="$HOME/.local/share/SlayTheSpire2"
HH="${HEADLESS_HOME:-$HOME/.local/share/AlphaSlayerHeadless}"
mkdir -p "$HH"
rsync -a --delete --exclude logs --exclude shader_cache --exclude sentry --exclude 'sentry.dat' \
      "$SRC/" "$HH/SlayTheSpire2/"
python3 - "$HH/SlayTheSpire2" <<'PY'
import json, glob, sys
root = sys.argv[1]
for f in glob.glob(f"{root}/steam/*/settings.save*"):
    d = json.load(open(f))
    ml = d.setdefault("mod_settings", {}).setdefault("mod_list", [])
    off = [m["id"] for m in ml if m.get("source") == "steam_workshop"]
    for m in ml:
        if m.get("source") == "steam_workshop":
            m["is_enabled"] = False
    json.dump(d, open(f, "w"), indent=2)
    print(f"{f}: disabled {len(off)} workshop mod(s): {', '.join(off)}", file=sys.stderr)
PY
echo "$HH"
