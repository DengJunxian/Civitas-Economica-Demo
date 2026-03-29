"""Market realism metrics for pipeline-v2 reporting."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class MarketMetrics:
    """Compute compact microstructure + behavior metrics."""

    @staticmethod
    def compute(
        *,
        snapshot: Mapping[str, Any],
        trade_tape: list[Mapping[str, Any]] | None = None,
        role_flows: Mapping[str, float] | None = None,
    ) -> Dict[str, float]:
        depth = snapshot.get("depth")
        total_bid = 0.0
        total_ask = 0.0
        if isinstance(depth, Mapping):
            for row in depth.get("bids", []) or []:
                if isinstance(row, Mapping):
                    total_bid += _safe_float(row.get("qty"))
            for row in depth.get("asks", []) or []:
                if isinstance(row, Mapping):
                    total_ask += _safe_float(row.get("qty"))
        denom = max(total_bid + total_ask, 1.0)
        depth_imbalance = (total_bid - total_ask) / denom

        best_bid = _safe_float(snapshot.get("best_bid"))
        best_ask = _safe_float(snapshot.get("best_ask"))
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0
        spread_pct = spread / mid if mid > 0 else 0.0

        tape = trade_tape or []
        total_notional = 0.0
        total_qty = 0.0
        for trade in tape:
            if not isinstance(trade, Mapping):
                continue
            px = _safe_float(trade.get("price"))
            qty = _safe_float(trade.get("quantity"))
            total_notional += px * qty
            total_qty += qty
        vwap = total_notional / total_qty if total_qty > 0 else _safe_float(snapshot.get("last_price"))

        flows = role_flows or {}
        herding_proxy = 0.0
        if flows:
            gross = sum(abs(_safe_float(v)) for v in flows.values())
            directional = abs(sum(_safe_float(v) for v in flows.values()))
            herding_proxy = directional / gross if gross > 0 else 0.0

        return {
            "spread": float(spread),
            "spread_pct": float(spread_pct),
            "depth_bid_total": float(total_bid),
            "depth_ask_total": float(total_ask),
            "depth_imbalance": float(depth_imbalance),
            "trade_count": float(snapshot.get("trade_count", len(tape))),
            "vwap": float(vwap),
            "herding_proxy": float(herding_proxy),
        }

