"""Refuse to run compute on a GPU that drives a monitor.

Four gfx-ring timeouts (2026-09-03 x2, 09-26, 10-15) all hit the 7900 XT at PCI 03:00.0 while
ROCm compute shared it with Hyprland: the ring reset ("Illegal opcode in command stream",
blamed on Hyprland) takes the desktop, Steam, Firefox and the trainer down together. Since
the second 7900 XT (11:00.0) has no monitors, compute belongs there. HIP device indices are
only an ordering, so this checks the live PCI bus of the selected device against the cards
whose DRM connectors report "connected".
"""
from __future__ import annotations

import glob
import os


def display_buses() -> set[int]:
    buses = set()
    for card in glob.glob("/sys/class/drm/card[0-9]*"):
        if "-" in os.path.basename(card):
            continue
        connected = any(open(s).read().strip() == "connected"
                        for s in glob.glob(f"{card}-*/status"))
        if connected:
            pci = os.path.basename(os.path.realpath(os.path.join(card, "device")))  # 0000:03:00.0
            try:
                buses.add(int(pci.split(":")[1], 16))
            except (IndexError, ValueError):
                pass
    return buses


def refuse_display_gpu(dev, allow: bool = False):
    """Raises SystemExit if torch device ``dev`` is a GPU driving a monitor (unless ``allow``)."""
    import torch
    if dev.type != "cuda" or allow or os.environ.get("ALPHASLAYER_ALLOW_DISPLAY_GPU") == "1":
        return
    bus = torch.cuda.get_device_properties(dev).pci_bus_id
    if bus in display_buses():
        raise SystemExit(
            f"refusing to run on the GPU at PCI bus {bus:02x}: it drives a monitor, and compute "
            f"there has caused 4 desktop-killing GPU resets. Use the headless card (exp_m4.sh "
            f"GPU=1) or set ALPHASLAYER_ALLOW_DISPLAY_GPU=1.")
