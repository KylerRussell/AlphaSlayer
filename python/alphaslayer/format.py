"""Reader for the packed binary episode format written by the C# probe.

Layout is defined by ``probe/src/EpisodeWriter.cs``; the two must be changed together and
``FORMAT_VERSION`` bumped. All integers are little-endian.

The file is a gzip stream of:

    header:  u32 magic 'ASLE' | u16 version | i32 vocab_hash
    records: u8 type, then either a STEP (0) or TERMINAL (1) body

Records are ragged rather than padded: fixed-width padding for worst-case hand/enemy/legal
counts measured the same size as the JSON it replaced, because a typical step uses a small
fraction of the caps.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator

MAGIC = 0x454C5341  # "ASLE"
FORMAT_VERSION = 3  # v3: relics carry (idx, counter, melted)

REC_STEP = 0
REC_TERMINAL = 1

# legal-action kind codes; must match EpisodeWriter / Obs.LegalAction
ACTION_KINDS = ("play", "end_turn", "select_card", "select_done")

# Must match EpisodeWriter.Outcomes.
OUTCOMES = (
    "enemies_cleared",
    "player_dead",
    "turn_cap",
    "combat_ended",
    "no_legal_actions",
    "decision_timeout",
    "turn_timeout",
    "unknown",
)

GLOBALS = (
    "turn", "round", "phase", "side", "hp", "max_hp", "block",
    "energy", "max_energy", "stars", "gold", "draw_count",
)

CARD_TOKEN = (
    "card_idx", "upgrade", "ench_idx", "ench_amount", "ench_disabled",
    "cost", "cost_x", "star_cost", "type", "rarity", "target_type",
    "playable", "_reserved",
)
CARD_TOKEN_WIDTH = len(CARD_TOKEN)

ENEMY_SCALARS = ("monster_idx", "hp", "max_hp", "block", "alive", "hittable")


class FormatError(RuntimeError):
    pass


@dataclass
class Step:
    ep: int
    t: int
    globals: tuple            # len(GLOBALS) ints, see GLOBALS
    player_powers: list       # [(power_idx, amount)]
    relics: list              # [(relic_idx, counter, melted)]; counter is -1 if none
    potions: list             # [potion_idx]
    hand: list                # [tuple(CARD_TOKEN_WIDTH)]
    draw_bag: list            # [(card_idx, count)]
    discard_bag: list
    exhaust_bag: list
    enemies: list             # [dict(scalars + powers + intents)]
    legal: list               # [(kind, hand_idx, target_idx, card_idx)]; kind indexes ACTION_KINDS.
                              # For select_* actions hand_idx indexes the prompt's option list,
                              # not the hand.
    action_idx: int
    hp_delta: int

    def g(self, name: str) -> int:
        return self.globals[GLOBALS.index(name)]


@dataclass
class Terminal:
    ep: int
    outcome: str
    won: bool
    turns: int
    steps: int
    hp_start: int
    hp_end: int
    reward: float


@dataclass
class Header:
    version: int
    vocab_hash: int


class _Cursor:
    """Sequential little-endian reader over an in-memory buffer."""

    __slots__ = ("buf", "off")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.off = 0

    def u8(self) -> int:
        v = self.buf[self.off]
        self.off += 1
        return v

    def i8(self) -> int:
        v = struct.unpack_from("<b", self.buf, self.off)[0]
        self.off += 1
        return v

    def u16(self) -> int:
        v = struct.unpack_from("<H", self.buf, self.off)[0]
        self.off += 2
        return v

    def i16(self) -> int:
        v = struct.unpack_from("<h", self.buf, self.off)[0]
        self.off += 2
        return v

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.buf, self.off)[0]
        self.off += 4
        return v

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.buf, self.off)[0]
        self.off += 4
        return v

    def f32(self) -> float:
        v = struct.unpack_from("<f", self.buf, self.off)[0]
        self.off += 4
        return v

    def i16s(self, n: int) -> tuple:
        v = struct.unpack_from(f"<{n}h", self.buf, self.off)
        self.off += 2 * n
        return v

    @property
    def eof(self) -> bool:
        return self.off >= len(self.buf)


def _read_bag(c: _Cursor) -> list:
    return [(c.i16(), c.i16()) for _ in range(c.u8())]


def _read_powers(c: _Cursor) -> list:
    return [(c.i16(), c.i16()) for _ in range(c.u8())]


def _read_step(c: _Cursor) -> Step:
    ep = c.i32()
    t = c.u16()
    g = c.i16s(len(GLOBALS))
    player_powers = _read_powers(c)
    relics = [(c.i16(), c.i16(), c.i16()) for _ in range(c.u8())]
    potions = [c.i16() for _ in range(c.u8())]
    hand = [c.i16s(CARD_TOKEN_WIDTH) for _ in range(c.u8())]
    draw_bag = _read_bag(c)
    discard_bag = _read_bag(c)
    exhaust_bag = _read_bag(c)

    enemies = []
    for _ in range(c.u8()):
        scalars = c.i16s(len(ENEMY_SCALARS))
        e = dict(zip(ENEMY_SCALARS, scalars))
        e["powers"] = _read_powers(c)
        e["intents"] = [(c.i16(), c.i16(), c.i16()) for _ in range(c.u8())]
        enemies.append(e)

    legal = [(c.u8(), c.i8(), c.i8(), c.i16()) for _ in range(c.u16())]
    action_idx = c.u16()
    hp_delta = c.i16()
    return Step(ep, t, g, player_powers, relics, potions, hand,
                draw_bag, discard_bag, exhaust_bag, enemies, legal, action_idx, hp_delta)


def _read_terminal(c: _Cursor) -> Terminal:
    ep = c.i32()
    code = c.u8()
    won = bool(c.u8())
    turns = c.u16()
    steps = c.u16()
    hp_start = c.i16()
    hp_end = c.i16()
    reward = c.f32()
    outcome = OUTCOMES[code] if code < len(OUTCOMES) else "unknown"
    return Terminal(ep, outcome, won, turns, steps, hp_start, hp_end, reward)


def read(path: str):
    """Yields ``(header, iterator)``; the iterator produces Step and Terminal records in order."""
    with gzip.open(path, "rb") as fh:
        buf = fh.read()
    c = _Cursor(buf)
    magic = c.u32()
    if magic != MAGIC:
        raise FormatError(f"bad magic 0x{magic:08X}, expected 0x{MAGIC:08X}")
    version = c.u16()
    if version != FORMAT_VERSION:
        raise FormatError(
            f"format version {version} != reader {FORMAT_VERSION}; "
            "regenerate the dataset or update alphaslayer.format")
    vocab_hash = c.i32()

    def it() -> Iterator:
        while not c.eof:
            kind = c.u8()
            if kind == REC_STEP:
                yield _read_step(c)
            elif kind == REC_TERMINAL:
                yield _read_terminal(c)
            else:
                raise FormatError(f"unknown record type {kind} at offset {c.off - 1}")

    return Header(version, vocab_hash), it()


def load(path: str):
    """Eager convenience: returns ``(header, steps, terminals)``."""
    header, it = read(path)
    steps, terminals = [], []
    for rec in it:
        (steps if isinstance(rec, Step) else terminals).append(rec)
    return header, steps, terminals
