#!/usr/bin/env bash
# Build, install and run the AlphaSlayer probe headlessly.
#
#   ./run.sh                          # inventory only (safe, fast)
#   ./run.sh inventory,throughput     # add the throughput measurement
#   ./run.sh inventory,throughput,snapshot
#
# Extra flags pass through, e.g.:
#   ./run.sh throughput --probe-combats 50 --probe-character silent
#
# Uninstall: rm -rf "$GAME/mods/alphaslayer_probe"
set -euo pipefail

GAME="${STS2_DIR:-$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2}"
MODID=alphaslayer_probe
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${PROBE_OUT:-/tmp/alphaslayer_probe}"
PHASES="${1:-inventory}"
shift || true

[ -d "$GAME" ] || { echo "Game not found at: $GAME" >&2; exit 1; }

echo "==> building"
dotnet build "$HERE/AlphaSlayerProbe.csproj" -c Release -v quiet --nologo

echo "==> installing to $GAME/mods/$MODID"
mkdir -p "$GAME/mods/$MODID"
cp "$HERE/bin/Release/$MODID.dll" "$GAME/mods/$MODID/"
cp "$HERE/$MODID.json"            "$GAME/mods/$MODID/"

mkdir -p "$OUT"
echo "==> running headless (phases: $PHASES)"
cd "$GAME"
# --fixed-fps uncaps Godot's main loop (it sits at 60fps otherwise even with MaxFps=0),
# which is the single biggest throughput lever: 40.2s -> 1.28s for identical episodes.
SteamAppId=2868840 SteamGameId=2868840 \
  ./SlayTheSpire2 --headless --fixed-fps "${FIXED_FPS:-5000}" \
    --probe="$PHASES" --probe-out="$OUT" --probe-quit "$@" \
    2>&1 | grep -E "^\[probe\]|FATAL|Unhandled|error CS" || true

echo
echo "==> report: $OUT/probe_report.json"
[ -f "$OUT/probe_report.json" ] && cat "$OUT/probe_report.json" || echo "(no report written — see $OUT/probe.log)"
