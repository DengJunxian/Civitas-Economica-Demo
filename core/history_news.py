"""Historical news retrieval, aggregation, and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from core.event_store import EventRecord, EventStore, EventType

try:
    import akshare as ak
except Exception:  # pragma: no cover - optional dependency
    ak = None


_POSITIVE_TOKENS = (
    "支持",
    "提振",
    "回暖",
    "改善",
    "稳增长",
    "降准",
    "降息",
    "宽松",
    "反弹",
    "企稳",
    "利好",
)
_NEGATIVE_TOKENS = (
    "风险",
    "下行",
    "压力",
    "收紧",
    "紧缩",
    "冲击",
    "抛售",
    "暴跌",
    "通胀",
    "违约",
    "利空",
)
_MACRO_INDEX_TOKENS = (
    "政策",
    "央行",
    "财政",
    "货币",
    "监管",
    "经济",
    "宏观",
    "市场",
    "指数",
    "上证",
    "沪深300",
    "中证",
    "流动性",
    "利率",
    "risk appetite",
    "macro",
    "index",
    "liquidity",
)


@dataclass(slots=True)
class DailyNewsDigest:
    date: str
    summary: str
    shock_score: float
    news_count: int
    headlines: List[str] = field(default_factory=list)
    source_mix: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "summary": self.summary,
            "shock_score": float(self.shock_score),
            "news_count": int(self.news_count),
            "headlines": list(self.headlines),
            "source_mix": dict(self.source_mix),
        }


@dataclass(slots=True)
class HistoryNewsBundle:
    source_strategy: str
    scope: str
    symbol: str
    start_date: str
    end_date: str
    items_by_day: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    daily_digests: List[DailyNewsDigest] = field(default_factory=list)
    coverage: Dict[str, Any] = field(default_factory=dict)
    persistence: Dict[str, Any] = field(default_factory=dict)

    def digest_map(self) -> Dict[str, DailyNewsDigest]:
        return {item.date: item for item in self.daily_digests}

    def digest_rows(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.daily_digests]


class HistoryNewsService:
    """Builds day-level news digest for replay windows with reproducibility metadata."""

    def __init__(
        self,
        *,
        event_store: Optional[EventStore] = None,
        seed_store_path: str | Path = "data/seed_events.jsonl",
    ) -> None:
        self.event_store = event_store or EventStore()
        self.seed_store_path = Path(seed_store_path)

    def build_news_bundle(
        self,
        *,
        start_date: str,
        end_date: str,
        symbol: str,
        source_strategy: str = "mixed",
        scope: str = "macro_index",
        topk_per_day: int = 8,
        persist: bool = True,
        persist_dataset: str = "history_replay_news",
        scenario_prefix: str = "history_replay_news",
    ) -> HistoryNewsBundle:
        start_ts = self._to_day_start(start_date)
        end_ts = self._to_day_end(end_date)
        strategy = str(source_strategy or "mixed").strip().lower()
        if strategy not in {"mixed", "online", "local"}:
            strategy = "mixed"

        online_rows: List[Dict[str, Any]] = []
        local_rows: List[Dict[str, Any]] = []
        if strategy in {"mixed", "online"}:
            online_rows = self._load_online_rows(start_ts, end_ts)
        if strategy in {"mixed", "local"}:
            local_rows = self._load_local_rows(start_ts, end_ts)

        merged = self._dedupe_rows([*online_rows, *local_rows])
        filtered = self._filter_rows(merged, symbol=symbol, scope=scope)
        grouped = self._group_by_day(filtered, topk_per_day=max(1, int(topk_per_day)))

        digests = [
            self._build_daily_digest(day, rows, symbol=symbol)
            for day, rows in sorted(grouped.items(), key=lambda item: item[0])
        ]

        coverage = self._build_coverage(
            grouped=grouped,
            start_ts=start_ts,
            end_ts=end_ts,
            online_rows=online_rows,
            local_rows=local_rows,
        )
        persistence: Dict[str, Any] = {"enabled": bool(persist)}
        if persist:
            persistence = self._persist_rows(
                grouped=grouped,
                dataset_version=str(persist_dataset or "history_replay_news"),
                start_ts=start_ts,
                end_ts=end_ts,
                symbol=symbol,
                scenario_prefix=scenario_prefix,
            )

        return HistoryNewsBundle(
            source_strategy=strategy,
            scope=str(scope or "macro_index"),
            symbol=str(symbol or ""),
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
            items_by_day=grouped,
            daily_digests=digests,
            coverage=coverage,
            persistence=persistence,
        )

    def _load_online_rows(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Dict[str, Any]]:
        if ak is None:
            return []
        fetchers: Sequence[Tuple[str, Callable[[], pd.DataFrame]]] = (
            ("cls", lambda: ak.stock_info_global_cls(symbol="重点")),
            ("eastmoney", ak.stock_info_global_em),
            ("sina", ak.stock_info_global_sina),
        )
        rows: List[Dict[str, Any]] = []
        for source_name, fn in fetchers:
            try:
                frame = fn()
            except Exception:
                continue
            if frame is None or frame.empty:
                continue
            rows.extend(self._normalize_frame_rows(frame, source_name=source_name, start_ts=start_ts, end_ts=end_ts))
        return rows

    def _load_local_rows(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        rows.extend(self._load_seed_store_rows(start_ts, end_ts))
        rows.extend(self._load_event_store_rows(start_ts, end_ts))
        return rows

    def _load_seed_store_rows(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Dict[str, Any]]:
        if not self.seed_store_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        try:
            lines = self.seed_store_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return rows
        for line in lines:
            text = str(line or "").strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            title = str(payload.get("title", "")).strip()
            body = str(payload.get("summary", "") or payload.get("raw_text", "")).strip()
            published = self._parse_ts(payload.get("created_at") or payload.get("processed_at"))
            if published is None or published < start_ts or published > end_ts:
                continue
            rows.append(
                {
                    "title": title,
                    "content": body,
                    "source": str(payload.get("source", "seed_store")),
                    "source_url": str(payload.get("source_url", "")),
                    "published_at": published,
                    "origin": "seed_store",
                    "sentiment": float(payload.get("sentiment", 0.0) or 0.0),
                    "impact_level": str(payload.get("impact_level", "")),
                    "confidence": 0.82,
                }
            )
        return rows

    def _load_event_store_rows(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        datasets = self._candidate_local_datasets()
        for dataset in datasets:
            try:
                frame = self.event_store.query_events(
                    dataset,
                    event_types=[EventType.NEWS.value],
                    start_time=start_ts.isoformat(),
                    end_time=end_ts.isoformat(),
                    visible_at=end_ts.isoformat(),
                )
            except Exception:
                continue
            if frame is None or frame.empty:
                continue
            for _, item in frame.iterrows():
                payload_raw = item.get("payload_json", "{}")
                try:
                    payload = payload_raw if isinstance(payload_raw, dict) else json.loads(str(payload_raw or "{}"))
                except Exception:
                    payload = {}
                published = self._parse_ts(item.get("timestamp"))
                if published is None or published < start_ts or published > end_ts:
                    continue
                rows.append(
                    {
                        "title": str(payload.get("headline") or payload.get("title") or payload.get("summary") or "").strip(),
                        "content": str(payload.get("content") or payload.get("body") or payload.get("summary") or "").strip(),
                        "source": str(payload.get("source") or item.get("source") or "event_store"),
                        "source_url": str(payload.get("source_url") or payload.get("url") or ""),
                        "published_at": published,
                        "origin": f"event_store:{dataset}",
                        "sentiment": float(payload.get("sentiment", 0.0) or 0.0),
                        "impact_level": str(payload.get("impact_level", "")),
                        "confidence": float(item.get("confidence", 0.75) or 0.75),
                    }
                )
        return rows

    def _candidate_local_datasets(self) -> List[str]:
        root = self.event_store.root_dir
        if not root.exists():
            return []
        preferred = ["history_replay_news", "policy_lab", "default"]
        existing = [path.name for path in root.iterdir() if path.is_dir()]
        ordered: List[str] = []
        for name in [*preferred, *sorted(existing)]:
            if name in existing and name not in ordered:
                ordered.append(name)
        return ordered

    def _normalize_frame_rows(
        self,
        frame: pd.DataFrame,
        *,
        source_name: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        title_col = self._pick_column(frame.columns, ["标题", "title", "headline", "内容", "摘要", "text"])
        content_col = self._pick_column(frame.columns, ["内容", "摘要", "content", "text", "标题", "title"])
        time_col = self._pick_column(frame.columns, ["发布时间", "发布日期", "时间", "date", "datetime"])
        link_col = self._pick_column(frame.columns, ["链接", "url", "source_url", "link"])

        rows: List[Dict[str, Any]] = []
        for _, item in frame.iterrows():
            title = str(item.get(title_col, "") if title_col else "").strip()
            content = str(item.get(content_col, "") if content_col else "").strip()
            published = self._parse_ts(item.get(time_col)) if time_col else None
            if published is None:
                continue
            if published < start_ts or published > end_ts:
                continue
            rows.append(
                {
                    "title": title,
                    "content": content or title,
                    "source": source_name,
                    "source_url": str(item.get(link_col, "") if link_col else ""),
                    "published_at": published,
                    "origin": f"online:{source_name}",
                    "sentiment": 0.0,
                    "impact_level": "",
                    "confidence": 0.7,
                }
            )
        return rows

    @staticmethod
    def _pick_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
        normalized = {str(name): str(name) for name in columns}
        for cand in candidates:
            if cand in normalized:
                return normalized[cand]
        return None

    @staticmethod
    def _parse_ts(value: Any) -> Optional[pd.Timestamp]:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts

    @staticmethod
    def _to_day_start(value: str) -> pd.Timestamp:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp.utcnow().normalize()
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.normalize()

    @staticmethod
    def _to_day_end(value: str) -> pd.Timestamp:
        start = HistoryNewsService._to_day_start(value)
        return start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    def _dedupe_rows(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        unique: List[Dict[str, Any]] = []
        for row in rows:
            title = str(row.get("title", "")).strip()
            content = str(row.get("content", "")).strip()
            norm = re.sub(r"\s+", "", f"{title}|{content[:80]}").lower()
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            item = dict(row)
            item["relevance_score"] = float(self._relevance_score(item))
            unique.append(item)
        unique.sort(
            key=lambda item: (
                float(item.get("relevance_score", 0.0)),
                float(item.get("confidence", 0.0)),
                float(item.get("published_at", pd.Timestamp(0, tz="UTC")).timestamp()),
            ),
            reverse=True,
        )
        return unique

    def _relevance_score(self, row: Mapping[str, Any]) -> float:
        text = f"{row.get('title', '')} {row.get('content', '')}".lower()
        macro_hits = sum(1 for token in _MACRO_INDEX_TOKENS if token.lower() in text)
        pos_hits = sum(1 for token in _POSITIVE_TOKENS if token in text)
        neg_hits = sum(1 for token in _NEGATIVE_TOKENS if token in text)
        intensity = abs(float(row.get("sentiment", 0.0) or 0.0))
        impact = str(row.get("impact_level", "")).strip().lower()
        impact_bonus = {"critical": 0.4, "high": 0.25, "medium": 0.12}.get(impact, 0.0)
        return float(macro_hits * 0.45 + (pos_hits + neg_hits) * 0.12 + intensity * 0.35 + impact_bonus)

    def _filter_rows(self, rows: Sequence[Dict[str, Any]], *, symbol: str, scope: str) -> List[Dict[str, Any]]:
        scope_norm = str(scope or "macro_index").strip().lower()
        if scope_norm != "macro_index":
            return list(rows)
        filtered = []
        symbol_hint = str(symbol or "").lower()
        for item in rows:
            text = f"{item.get('title', '')} {item.get('content', '')}".lower()
            macro_hit = any(token.lower() in text for token in _MACRO_INDEX_TOKENS)
            symbol_hit = bool(symbol_hint and symbol_hint in text)
            if macro_hit or symbol_hit:
                filtered.append(item)
        if filtered:
            return filtered
        return list(rows)

    def _group_by_day(self, rows: Sequence[Dict[str, Any]], *, topk_per_day: int) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in rows:
            published = item.get("published_at")
            if not isinstance(published, pd.Timestamp):
                continue
            day = published.tz_convert("UTC").strftime("%Y-%m-%d")
            grouped.setdefault(day, []).append(item)
        for day in list(grouped.keys()):
            entries = sorted(
                grouped[day],
                key=lambda row: (
                    float(row.get("relevance_score", 0.0)),
                    float(row.get("confidence", 0.0)),
                    float(row["published_at"].timestamp()) if isinstance(row.get("published_at"), pd.Timestamp) else 0.0,
                ),
                reverse=True,
            )
            grouped[day] = entries[:topk_per_day]
        return grouped

    def _build_daily_digest(self, day: str, rows: Sequence[Mapping[str, Any]], *, symbol: str) -> DailyNewsDigest:
        llm = self._llm_digest(day=day, rows=rows, symbol=symbol)
        if llm is None:
            summary, shock_score = self._fallback_digest(rows)
        else:
            summary = str(llm.get("summary", "")).strip()
            shock_score = float(llm.get("shock_score", 0.0) or 0.0)
        shock_score = float(max(-1.0, min(1.0, shock_score)))
        source_mix: Dict[str, int] = {}
        for item in rows:
            source = str(item.get("source", "unknown"))
            source_mix[source] = int(source_mix.get(source, 0) + 1)
        headlines = [str(item.get("title", "")) for item in rows if str(item.get("title", "")).strip()]
        return DailyNewsDigest(
            date=str(day),
            summary=summary,
            shock_score=shock_score,
            news_count=len(rows),
            headlines=headlines[:8],
            source_mix=source_mix,
        )

    def _llm_digest(self, *, day: str, rows: Sequence[Mapping[str, Any]], symbol: str) -> Optional[Dict[str, Any]]:
        if not rows:
            return {"summary": "当日未检索到可用新闻，沿用政策主线。", "shock_score": 0.0}
        try:
            from core.inference.api_backend import APIBackend
        except Exception:
            return None
        payload_rows = [
            {
                "title": str(item.get("title", "")),
                "content": str(item.get("content", ""))[:300],
                "source": str(item.get("source", "")),
                "published_at": str(item.get("published_at", "")),
                "relevance": float(item.get("relevance_score", 0.0)),
            }
            for item in list(rows)[:8]
        ]
        prompt = "\n".join(
            [
                "你是回测事件编排助手，请根据当日新闻生成政策冲击摘要。",
                "仅输出 JSON，不要输出代码块。",
                '格式: {"summary":"...","shock_score":-1到1之间小数}',
                f"交易日: {day}",
                f"指数代码: {symbol}",
                f"新闻数据: {json.dumps(payload_rows, ensure_ascii=False, sort_keys=True)}",
            ]
        )
        try:
            backend = APIBackend(model="deepseek-chat", max_tokens=180, temperature=0.2)
            text = str(
                backend.generate(
                    prompt,
                    system_prompt="你是金融政策仿真助手，输出简洁且可执行。",
                    timeout_budget=15.0,
                    fallback_response="",
                )
                or ""
            ).strip()
        except Exception:
            return None
        if not text or text.startswith("[API Error]"):
            return None
        parsed = self._parse_json_fragment(text)
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _parse_json_fragment(text: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _fallback_digest(self, rows: Sequence[Mapping[str, Any]]) -> Tuple[str, float]:
        if not rows:
            return "当日未检索到可用新闻，沿用政策主线。", 0.0
        top = list(rows)[:3]
        headlines = [str(item.get("title", "")).strip() for item in top if str(item.get("title", "")).strip()]
        summary = "；".join(headlines) if headlines else "当日出现宏观与指数相关新闻。"

        score = 0.0
        for item in rows:
            text = f"{item.get('title', '')} {item.get('content', '')}"
            pos_hits = sum(1 for token in _POSITIVE_TOKENS if token in text)
            neg_hits = sum(1 for token in _NEGATIVE_TOKENS if token in text)
            score += float(item.get("sentiment", 0.0) or 0.0)
            score += 0.12 * float(pos_hits - neg_hits)
        score /= max(len(rows), 1)
        score = max(-1.0, min(1.0, score))
        return summary, score

    def _build_coverage(
        self,
        *,
        grouped: Mapping[str, Sequence[Mapping[str, Any]]],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        online_rows: Sequence[Mapping[str, Any]],
        local_rows: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        bdays = pd.bdate_range(start=start_ts.tz_convert(None).date(), end=end_ts.tz_convert(None).date())
        day_count = int(len(bdays))
        day_hits = int(len(grouped))
        source_distribution: Dict[str, int] = {}
        for rows in grouped.values():
            for item in rows:
                source = str(item.get("source", "unknown"))
                source_distribution[source] = int(source_distribution.get(source, 0) + 1)
        return {
            "window_days": day_count,
            "days_with_news": day_hits,
            "coverage_rate": float(day_hits / max(day_count, 1)),
            "selected_news_count": int(sum(len(items) for items in grouped.values())),
            "online_candidates": int(len(online_rows)),
            "local_candidates": int(len(local_rows)),
            "source_distribution": source_distribution,
        }

    def _persist_rows(
        self,
        *,
        grouped: Mapping[str, Sequence[Mapping[str, Any]]],
        dataset_version: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        symbol: str,
        scenario_prefix: str,
    ) -> Dict[str, Any]:
        records: List[EventRecord] = []
        for day, rows in grouped.items():
            for idx, item in enumerate(rows):
                published = item.get("published_at")
                if not isinstance(published, pd.Timestamp):
                    published = self._to_day_start(day)
                payload = {
                    "headline": str(item.get("title", "")),
                    "content": str(item.get("content", "")),
                    "source": str(item.get("source", "")),
                    "source_url": str(item.get("source_url", "")),
                    "sentiment": float(item.get("sentiment", 0.0) or 0.0),
                    "impact_level": str(item.get("impact_level", "")),
                    "relevance_score": float(item.get("relevance_score", 0.0) or 0.0),
                    "window_day": str(day),
                    "symbol": str(symbol or ""),
                }
                records.append(
                    EventRecord(
                        timestamp=published.isoformat(),
                        visibility_time=published.isoformat(),
                        source=str(item.get("origin", "history_news_service")),
                        confidence=float(item.get("confidence", 0.75) or 0.75),
                        event_type=EventType.NEWS,
                        payload=payload,
                        metadata={"module": "history_news_service", "rank_in_day": int(idx + 1)},
                    )
                )

        scenario_id = self._scenario_id(
            scenario_prefix=scenario_prefix,
            dataset_version=dataset_version,
            start_ts=start_ts,
            end_ts=end_ts,
            symbol=symbol,
        )
        snapshot_id = ""
        if records:
            self.event_store.append_events(dataset_version=dataset_version, events=records)
            snapshot = self.event_store.create_snapshot(
                dataset_version=dataset_version,
                seed=0,
                config_hash="history-news",
                feature_flags={"persist_news_events": True},
                scenario_id=scenario_id,
                metadata={"symbol": str(symbol or ""), "module": "history_news_service"},
            )
            snapshot_id = snapshot.snapshot_id
            self.event_store.write_scenario_manifest(
                dataset_version=dataset_version,
                scenario_id=scenario_id,
                snapshot_id=snapshot.snapshot_id,
                start_time=start_ts.isoformat(),
                end_time=end_ts.isoformat(),
                seed=0,
                config_hash="history-news",
                event_types=[EventType.NEWS.value],
                metadata={"symbol": str(symbol or ""), "module": "history_news_service"},
            )
        return {
            "enabled": True,
            "dataset_version": dataset_version,
            "scenario_id": scenario_id,
            "snapshot_id": snapshot_id,
            "persisted_news_count": int(len(records)),
        }

    @staticmethod
    def _scenario_id(
        *,
        scenario_prefix: str,
        dataset_version: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        symbol: str,
    ) -> str:
        digest = hashlib.sha256(
            f"{dataset_version}|{symbol}|{start_ts.isoformat()}|{end_ts.isoformat()}|{datetime.now(timezone.utc).isoformat()}".encode("utf-8")
        ).hexdigest()[:12]
        return f"{scenario_prefix}_{digest}"
