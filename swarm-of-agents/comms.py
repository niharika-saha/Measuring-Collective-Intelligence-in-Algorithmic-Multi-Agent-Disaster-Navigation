"""
comms.py
--------
Explicit communication layer for the swarm.

Replaces three things that were implicit before:
  * a global, instantaneous shared memory,
  * a hard-coded 500 m danger-memory merge,
  * pheromones that were always on.

Everything an agent learns from another agent now travels as a Message through
a MessageBus, so radius, delay, message counts and bytes are all measurable and
all three channels can be switched on and off independently.

Channels (CommConfig switches)
------------------------------
  shared_memory     agents post what they sense to a global board; every agent
                    reads it. Posts go through the bus (kind="board") so
                    `board_delay_steps` models uplink latency.
  local_messaging   agent-to-agent radio. Kinds:
                      "hazard"    delta of newly sensed hazards, broadcast
                      "sync"      full knowledge exchanged when two agents
                                  come into range (optional, sync_on_contact)
                      "advisory"  consolidated hazards from an ASSESSOR hub
                      "assign"    goal assignment from an ALLOCATOR hub
  pheromones        trail deposited by agents that reach their goal.

Link rule
---------
A transmission from A reaches B iff
        distance(A, B) <= max(range(A), range(B))
where range(X) = radius_m * role radius multiplier. The max() lets a hub with a
large antenna both hear and be heard across its footprint. Distance is measured
when the message is sent.

Cost model
----------
One broadcast = one TRANSMISSION (what the sender pays) and one DELIVERY per
receiver (what the swarm receives). Bytes use a fixed, documented size model
(see payload_bytes) so runs are comparable; it is not a wire format.

Timing
------
A message sent at step t with delay d is delivered in the comms phase of step
t + d, so with d = 0 it is usable by the receiver on step t + 1.
"""

import copy
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

BOARD = -1   # pseudo-recipient id for the shared board

HEADER_BYTES = 16
NODE_ID_BYTES = 8
DANGER_ENTRY_BYTES = 12    # node id + float32
SCALAR_BYTES = 4


@dataclass(frozen=True)
class CommConfig:
    shared_memory:    bool  = True
    local_messaging:  bool  = True
    pheromones:       bool  = True
    radius_m:         float = 500.0   # base radio range
    delay_steps:      int   = 0       # local message latency
    board_delay_steps: int  = 0       # latency before a post is visible on the board
    sync_on_contact:  bool  = True    # exchange full knowledge when agents meet

    def __post_init__(self):
        for name in ("delay_steps", "board_delay_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.radius_m, bool) or not isinstance(self.radius_m, (int, float)):
            raise ValueError("radius_m must be a finite positive number")
        if not math.isfinite(float(self.radius_m)) or float(self.radius_m) <= 0:
            raise ValueError("radius_m must be a finite positive number")
        for name in ("shared_memory", "local_messaging", "pheromones", "sync_on_contact"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")

    def with_(self, **kw) -> "CommConfig":
        """Copy with overrides, for ablation sweeps: cfg.with_(local_messaging=False)."""
        from dataclasses import replace
        return replace(self, **kw)


def build_delay_matrix_configs(local_delays, board_delays):
    """Build a full paired matrix of independent local-radio and board delays.

    Returns an insertion-ordered mapping whose keys identify both delays.
    Every condition keeps all three information channels enabled so the matrix
    isolates timing rather than channel availability.
    """
    local = tuple(local_delays)
    board = tuple(board_delays)
    for label, values in (("local_delays", local), ("board_delays", board)):
        if not values:
            raise ValueError(f"{label} must contain at least one delay")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise ValueError(f"{label} must contain non-negative integers")
        if len(set(values)) != len(values):
            raise ValueError(f"{label} must not contain duplicates")
    return {
        f"local_{ld}_board_{bd}": CommConfig(
            shared_memory=True,
            local_messaging=True,
            pheromones=True,
            delay_steps=ld,
            board_delay_steps=bd,
        )
        for ld in local for bd in board
    }


def payload_bytes(payload: dict) -> int:
    n = HEADER_BYTES
    for k, v in payload.items():
        if k == "blocked":
            n += NODE_ID_BYTES * len(v)
        elif k == "danger":
            n += DANGER_ENTRY_BYTES * len(v)
        elif k == "goal":
            n += NODE_ID_BYTES
        else:
            n += SCALAR_BYTES
    return n


@dataclass
class Message:
    id: int
    kind: str
    sender: int
    sent_step: int
    payload: dict
    size_bytes: int = 0
    scheduled_delay_steps: int = 0


@dataclass(order=True)
class _Pending:
    deliver_step: int
    seq: int
    recipient: int = field(compare=False)
    msg: Message = field(compare=False)


class MessageBus:
    def __init__(self, cfg: CommConfig):
        self.cfg = cfg
        self._heap: list = []
        self._seq = 0
        self._next_id = 0
        self.tx = defaultdict(int)            # kind -> transmissions
        self.rx = defaultdict(int)            # kind -> deliveries
        self.bytes_tx = defaultdict(int)
        self.bytes_rx = defaultdict(int)
        self.dropped = defaultdict(int)       # kind -> deliveries lost (receiver dead)
        self.tx_by_sender = defaultdict(int)
        self.bytes_by_sender = defaultdict(int)
        self.rx_by_recipient = defaultdict(int)
        self.dropped_by_recipient = defaultdict(int)
        self._delay_sum = 0
        self._scheduled_delay_sum = 0
        self._delivery_delay_samples = 0
        self._late_delivery_count = 0
        self._delay_sum_by_kind = defaultdict(int)
        self._delivery_samples_by_kind = defaultdict(int)
        self._late_delivery_by_kind = defaultdict(int)

    # ------------------------------------------------------------------ #
    def send(self, sender: int, kind: str, payload: dict, step: int,
             recipients: Iterable[int], delay: Optional[int] = None) -> Optional[Message]:
        """One transmission, delivered to each recipient after `delay` steps.
        Returns None (and costs nothing) if there is nobody to deliver to."""
        # Validate the request even when it has no recipients. Invalid sends
        # must not become silently acceptable just because the radio is empty.
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if isinstance(sender, bool) or not isinstance(sender, int) or sender < 0:
            raise ValueError("sender must be a non-negative integer agent id")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("kind must be a non-empty string")
        if delay is None:
            delay = self.cfg.delay_steps
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise ValueError("message delay must be a non-negative integer")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        recipients = list(dict.fromkeys(recipients))
        for recipient in recipients:
            if (isinstance(recipient, bool) or not isinstance(recipient, int)
                    or (recipient < 0 and recipient != BOARD)):
                raise ValueError("recipients must be non-negative agent ids or BOARD")
        if not recipients:
            return None
        # Snapshot payload at send time: later mutation by the sender must not
        # silently alter a message already in flight.
        payload_snapshot = copy.deepcopy(payload)
        size = payload_bytes(payload_snapshot)
        msg = Message(
            self._next_id, kind, sender, step, payload_snapshot, size,
            scheduled_delay_steps=delay,
        )
        self._next_id += 1
        self.tx[kind] += 1
        self.bytes_tx[kind] += size
        self.tx_by_sender[sender] += 1
        self.bytes_by_sender[sender] += size
        for r in recipients:
            self._seq += 1
            heapq.heappush(self._heap, _Pending(step + delay, self._seq, r, msg))
        return msg

    def deliver(self, step: int, is_alive=lambda r: True) -> list:
        """Pop everything due at or before `step`. Returns [(recipient, msg)].
        Deliveries to dead recipients are counted as dropped, not delivered."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        out = []
        while self._heap and self._heap[0].deliver_step <= step:
            p = heapq.heappop(self._heap)
            if p.recipient != BOARD and not is_alive(p.recipient):
                self.dropped[p.msg.kind] += 1
                self.dropped_by_recipient[p.recipient] += 1
                continue
            self.rx[p.msg.kind] += 1
            self.bytes_rx[p.msg.kind] += p.msg.size_bytes
            self.rx_by_recipient[p.recipient] += 1
            actual_delay = step - p.msg.sent_step
            self._delay_sum += actual_delay
            self._scheduled_delay_sum += p.msg.scheduled_delay_steps
            self._delivery_delay_samples += 1
            self._delay_sum_by_kind[p.msg.kind] += actual_delay
            self._delivery_samples_by_kind[p.msg.kind] += 1
            if actual_delay > p.msg.scheduled_delay_steps:
                self._late_delivery_count += 1
                self._late_delivery_by_kind[p.msg.kind] += 1
            out.append((p.recipient, p.msg))
        return out

    def pending(self) -> int:
        return len(self._heap)

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        kinds = sorted(set(self.tx) | set(self.rx) | set(self.dropped))
        n_rx = sum(self.rx.values())
        pending_by_kind = defaultdict(int)
        pending_by_recipient = defaultdict(int)
        for pending in self._heap:
            pending_by_kind[pending.msg.kind] += 1
            pending_by_recipient[pending.recipient] += 1
        return {
            "transmissions":   sum(self.tx.values()),
            "deliveries":      n_rx,
            "dropped":         sum(self.dropped.values()),
            "bytes_tx":        sum(self.bytes_tx.values()),
            "bytes_rx":        sum(self.bytes_rx.values()),
            "mean_delay_steps": (self._delay_sum / n_rx) if n_rx else None,
            "mean_scheduled_delay_steps": (
                self._scheduled_delay_sum / self._delivery_delay_samples
                if self._delivery_delay_samples else None
            ),
            "late_deliveries": self._late_delivery_count,
            "pending_deliveries": self.pending(),
            "pending_by_kind": dict(pending_by_kind),
            "pending_by_recipient": dict(pending_by_recipient),
            "dropped_by_recipient": dict(self.dropped_by_recipient),
            "in_flight":       self.pending(),
            "by_kind": {k: {"tx": self.tx[k], "rx": self.rx[k],
                            "bytes_tx": self.bytes_tx[k], "bytes_rx": self.bytes_rx[k],
                            "dropped": self.dropped[k],
                            "pending": pending_by_kind.get(k, 0),
                            "late_deliveries": self._late_delivery_by_kind[k],
                            "mean_delay_steps": (
                                self._delay_sum_by_kind[k] / self._delivery_samples_by_kind[k]
                                if self._delivery_samples_by_kind[k] else None
                            )} for k in kinds},
            "by_channel": {
                channel: {
                    "transmissions": sum(self.tx[k] for k in channel_kinds),
                    "deliveries": sum(self.rx[k] for k in channel_kinds),
                    "dropped": sum(self.dropped[k] for k in channel_kinds),
                    "pending_deliveries": sum(pending_by_kind.get(k, 0) for k in channel_kinds),
                    "bytes_tx": sum(self.bytes_tx[k] for k in channel_kinds),
                    "bytes_rx": sum(self.bytes_rx[k] for k in channel_kinds),
                    "late_deliveries": sum(self._late_delivery_by_kind[k] for k in channel_kinds),
                    "mean_delay_steps": (
                        sum(self._delay_sum_by_kind[k] for k in channel_kinds) /
                        sum(self._delivery_samples_by_kind[k] for k in channel_kinds)
                        if sum(self._delivery_samples_by_kind[k] for k in channel_kinds) else None
                    ),
                    "delivery_delay_samples": sum(
                        self._delivery_samples_by_kind[k] for k in channel_kinds
                    ),
                }
                for channel, channel_kinds in {
                    "shared_memory_board": [k for k in kinds if k == "board"],
                    "local_messaging": [k for k in kinds if k != "board"],
                }.items()
            },
            "tx_by_sender":    dict(self.tx_by_sender),
            "bytes_by_sender": dict(self.bytes_by_sender),
            "rx_by_recipient": dict(self.rx_by_recipient),
        }
