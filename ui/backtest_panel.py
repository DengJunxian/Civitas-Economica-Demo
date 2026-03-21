# file: ui/backtest_panel.py
"""Historical backtest panel."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtester import (
    BacktestConfig,
    BacktestReportGenerator,
    BacktestResult,
    HistoricalBacktester,
)
from core.tear_sheet import (
    build_standard_tear_sheet,
    compare_scenarios,
    export_tear_sheet_html,
    export_tear_sheet_json,
)

INDEX_OPTIONS = {
    "上证指数 (sh000001)": "sh000001",
    "沪深300 (sh000300)": "sh000300",
    "深证成指 (sz399001)": "sz399001",
    "创业板指 (sz399006)": "sz399006",
}

STRATEGY_OPTIONS = {
    "动量策略": "momentum",
    "均值回归": "mean_reversion",
    "风险平价": "risk_parity",
    "消息驱动": "news_driven",
    "Portfolio System (政策/情绪约束)": "portfolio_system",
}


def _default_window(period_days: int) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=max(period_days, 30))
    return start, end


def _to_frame(result: BacktestResult) -> pd.DataFrame:
    if not result or not result.dates:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result.dates),
            "equity": result.equity_curve,
            "benchmark": result.benchmark_curve,
            "drawdown": result.drawdowns,
            "position": result.position_series,
            "turnover": result.turnover_series,
            "ret": result.returns,
            "bench_ret": result.benchmark_returns,
            "real_price": result.real_prices,
            "sim_price": result.simulated_prices,
        }
    )
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _render_metrics(result: BacktestResult) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("总收益", f"{result.total_return:.2%}")
    c2.metric("年化收益", f"{result.cagr:.2%}")
    c3.metric("夏普", f"{result.sharpe_ratio:.2f}")
    c4.metric("最大回撤", f"{result.max_drawdown:.2%}")
    c5.metric("可信度", f"{result.credibility_score:.2f}")
    c6.metric("IR", f"{result.information_ratio:.2f}")


def _render_equity_chart(frame: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frame["date"], y=frame["equity"], mode="lines", name="策略净值", line=dict(width=2.5, color="#00c48c"))
    )
    fig.add_trace(
        go.Scatter(x=frame["date"], y=frame["benchmark"], mode="lines", name="基准净值", line=dict(width=1.8, color="#2f80ed"))
    )
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=35, b=10), title="策略 vs 基准")
    st.plotly_chart(fig, width="stretch", key="backtest_equity_chart")


def _render_risk_chart(frame: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frame["date"], y=frame["drawdown"], fill="tozeroy", name="回撤", line=dict(width=1.5, color="#ff6b6b"))
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["position"],
            mode="lines",
            name="仓位",
            line=dict(width=1.5, color="#f2c94c", dash="dot"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=35, b=10),
        title="回撤与仓位",
        yaxis=dict(tickformat=".0%"),
        yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False),
    )
    st.plotly_chart(fig, width="stretch", key="backtest_risk_chart")


def _result_payload(result: BacktestResult) -> Dict[str, Any]:
    return {
        "summary": result.get_summary(),
        "metadata": result.metadata,
        "top_factors": result.factor_snapshot[:10],
        "total_trades": result.total_trades,
        "total_days": result.total_days,
    }


def _clone_backtest_config(config: BacktestConfig, **updates: Any) -> BacktestConfig:
    payload = asdict(config)
    payload.update(updates)
    return BacktestConfig(**payload)


def _run_policy_ab_comparison(
    *,
    base_backtester: HistoricalBacktester,
    policy_a: float,
    policy_b: float,
    output_root: str = "outputs/policy_ab_reports",
) -> Dict[str, Any]:
    if base_backtester is None:
        raise ValueError("base_backtester is required")
    if base_backtester.historical_data is None or base_backtester.historical_data.empty:
        raise ValueError("Backtester has no loaded historical_data")

    cfg_a = _clone_backtest_config(base_backtester.config, policy_shock=float(policy_a))
    cfg_b = _clone_backtest_config(base_backtester.config, policy_shock=float(policy_b))

    bt_a = HistoricalBacktester(cfg_a)
    bt_a.historical_data = base_backtester.historical_data.copy()
    bt_a.benchmark_data = base_backtester.benchmark_data.copy()

    bt_b = HistoricalBacktester(cfg_b)
    bt_b.historical_data = base_backtester.historical_data.copy()
    bt_b.benchmark_data = base_backtester.benchmark_data.copy()

    result_a = bt_a.run_backtest()
    result_b = bt_b.run_backtest()

    payload_a = build_standard_tear_sheet(
        scenario_name=f"policy_A_{policy_a:+.2f}",
        returns=result_a.returns[1:] if len(result_a.returns) > 1 else result_a.returns,
        benchmark_returns=(
            result_a.benchmark_returns[1:] if len(result_a.benchmark_returns) > 1 else result_a.benchmark_returns
        ),
        dates=result_a.dates[1:] if len(result_a.dates) > 1 else result_a.dates,
        metadata={"policy_shock": policy_a, "strategy_name": result_a.strategy_name},
    )
    payload_b = build_standard_tear_sheet(
        scenario_name=f"policy_B_{policy_b:+.2f}",
        returns=result_b.returns[1:] if len(result_b.returns) > 1 else result_b.returns,
        benchmark_returns=(
            result_b.benchmark_returns[1:] if len(result_b.benchmark_returns) > 1 else result_b.benchmark_returns
        ),
        dates=result_b.dates[1:] if len(result_b.dates) > 1 else result_b.dates,
        metadata={"policy_shock": policy_b, "strategy_name": result_b.strategy_name},
    )
    compare_df = compare_scenarios([payload_a, payload_b])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_a_json = export_tear_sheet_json(payload_a, str(out_dir / "policy_A_tearsheet.json"))
    policy_a_html = export_tear_sheet_html(payload_a, str(out_dir / "policy_A_tearsheet.html"))
    policy_b_json = export_tear_sheet_json(payload_b, str(out_dir / "policy_B_tearsheet.json"))
    policy_b_html = export_tear_sheet_html(payload_b, str(out_dir / "policy_B_tearsheet.html"))

    compare_csv_path = out_dir / "policy_AB_comparison.csv"
    compare_df.to_csv(compare_csv_path, index=False, encoding="utf-8-sig")
    compare_json_path = out_dir / "policy_AB_comparison.json"
    compare_json_path.write_text(compare_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "policy_a": float(policy_a),
        "policy_b": float(policy_b),
        "result_a": result_a,
        "result_b": result_b,
        "payload_a": payload_a.to_dict(),
        "payload_b": payload_b.to_dict(),
        "compare_df": compare_df,
        "files": {
            "policy_a_json": policy_a_json,
            "policy_a_html": policy_a_html,
            "policy_b_json": policy_b_json,
            "policy_b_html": policy_b_html,
            "compare_csv": str(compare_csv_path.resolve()),
            "compare_json": str(compare_json_path.resolve()),
        },
    }


def render_backtest_panel(ctrl: Any = None) -> None:
    st.markdown("## 历史回测与研究面板")
    st.caption("支持 portfolio_system 策略，并支持一键生成政策A/B自动对比报告。")

    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None
    if "backtester" not in st.session_state:
        st.session_state.backtester = None
    if "backtest_cfg" not in st.session_state:
        st.session_state.backtest_cfg = {}
    if "policy_ab_compare" not in st.session_state:
        st.session_state.policy_ab_compare = None

    cfg_state = st.session_state.get("backtest_cfg", {})

    with st.form("backtest_config_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            symbol_label = st.selectbox(
                "交易标的",
                options=list(INDEX_OPTIONS.keys()),
                index=list(INDEX_OPTIONS.values()).index(cfg_state.get("symbol", "sh000001"))
                if cfg_state.get("symbol", "sh000001") in INDEX_OPTIONS.values()
                else 0,
            )
            benchmark_label = st.selectbox(
                "基准",
                options=list(INDEX_OPTIONS.keys()),
                index=list(INDEX_OPTIONS.values()).index(cfg_state.get("benchmark_symbol", "sh000001"))
                if cfg_state.get("benchmark_symbol", "sh000001") in INDEX_OPTIONS.values()
                else 0,
            )
            custom_window = st.checkbox("自定义日期窗口", value=cfg_state.get("custom_window", False))
            if custom_window:
                default_start_date, default_end_date = _default_window(int(cfg_state.get("period_days", 756)))
                start_date = st.date_input("开始日期", value=cfg_state.get("start_date_obj", default_start_date))
                end_date = st.date_input("结束日期", value=cfg_state.get("end_date_obj", default_end_date))
                period_days = 0
            else:
                start_date = None
                end_date = None
                # Streamlit 的 slider 默认值必须与 step 对齐，否则前端会出现告警。
                raw_period_days = int(cfg_state.get("period_days", 756))
                clipped_period_days = max(180, min(2000, raw_period_days))
                period_days_default = 180 + round((clipped_period_days - 180) / 10) * 10
                period_days = st.slider(
                    "回看天数",
                    min_value=180,
                    max_value=2000,
                    value=period_days_default,
                    step=10,
                )

        with col_b:
            strategy_label = st.selectbox(
                "策略模板",
                options=list(STRATEGY_OPTIONS.keys()),
                index=list(STRATEGY_OPTIONS.values()).index(cfg_state.get("strategy_name", "momentum"))
                if cfg_state.get("strategy_name", "momentum") in STRATEGY_OPTIONS.values()
                else 0,
            )
            lookback = st.slider("观察窗口", min_value=5, max_value=120, value=int(cfg_state.get("lookback", 20)))
            rebalance_frequency = st.select_slider(
                "调仓频率(交易日)",
                options=[1, 2, 3, 5, 10, 20],
                value=int(cfg_state.get("rebalance_frequency", 5)),
            )
            max_position = st.slider(
                "最大仓位",
                min_value=0.2,
                max_value=1.5,
                value=float(cfg_state.get("max_position", 1.0)),
                step=0.1,
            )
            allow_short = st.checkbox("允许做空", value=bool(cfg_state.get("allow_short", False)))

        with col_c:
            commission_bps = st.slider("佣金 (bps)", min_value=0.0, max_value=20.0, value=float(cfg_state.get("commission_bps", 2.5)), step=0.1)
            stamp_bps = st.slider("印花税 (bps, 卖出)", min_value=0.0, max_value=20.0, value=float(cfg_state.get("stamp_bps", 5.0)), step=0.1)
            slippage_bps = st.slider("滑点 (bps)", min_value=0.0, max_value=40.0, value=float(cfg_state.get("slippage_bps", 5.0)), step=0.5)
            market_impact = st.slider("冲击系数", min_value=0.0, max_value=0.5, value=float(cfg_state.get("market_impact", 0.05)), step=0.01)
            policy_shock = st.slider("政策冲击因子", min_value=-1.0, max_value=1.0, value=float(cfg_state.get("policy_shock", 0.0)), step=0.05)
            sentiment_weight = st.slider("情绪权重", min_value=0.0, max_value=1.0, value=float(cfg_state.get("sentiment_weight", 0.5)), step=0.05)
            civitas_factor_weight = st.slider(
                "Civitas 因子权重",
                min_value=0.0,
                max_value=1.0,
                value=float(cfg_state.get("civitas_factor_weight", 0.5)),
                step=0.05,
            )

        export_qlib = st.checkbox("回测后自动导出 Qlib 研究数据束", value=bool(cfg_state.get("export_qlib_bundle", False)))
        qlib_bundle_path = st.text_input("Qlib 导出路径", value=str(cfg_state.get("qlib_bundle_path", "outputs/backtest_qlib_bundle")))
        submitted = st.form_submit_button("运行历史回测", width="stretch")

    if submitted:
        config = BacktestConfig(
            symbol=INDEX_OPTIONS[symbol_label],
            benchmark_symbol=INDEX_OPTIONS[benchmark_label],
            start_date=str(start_date) if custom_window and start_date else None,
            end_date=str(end_date) if custom_window and end_date else None,
            period_days=int(period_days),
            strategy_name=STRATEGY_OPTIONS[strategy_label],
            lookback=int(lookback),
            rebalance_frequency=int(rebalance_frequency),
            allow_short=bool(allow_short),
            max_position=float(max_position),
            commission_rate=float(commission_bps) / 10000.0,
            stamp_duty_rate=float(stamp_bps) / 10000.0,
            slippage_bps=float(slippage_bps),
            market_impact=float(market_impact),
            policy_shock=float(policy_shock),
            sentiment_weight=float(sentiment_weight),
            civitas_factor_weight=float(civitas_factor_weight),
            export_qlib_bundle=bool(export_qlib),
            qlib_bundle_path=qlib_bundle_path,
        )

        st.session_state.backtest_cfg = {
            "symbol": config.symbol,
            "benchmark_symbol": config.benchmark_symbol,
            "custom_window": custom_window,
            "start_date_obj": start_date,
            "end_date_obj": end_date,
            "period_days": period_days,
            "strategy_name": config.strategy_name,
            "lookback": config.lookback,
            "rebalance_frequency": config.rebalance_frequency,
            "allow_short": config.allow_short,
            "max_position": config.max_position,
            "commission_bps": commission_bps,
            "stamp_bps": stamp_bps,
            "slippage_bps": slippage_bps,
            "market_impact": market_impact,
            "policy_shock": policy_shock,
            "sentiment_weight": sentiment_weight,
            "civitas_factor_weight": civitas_factor_weight,
            "export_qlib_bundle": export_qlib,
            "qlib_bundle_path": qlib_bundle_path,
        }

        run_backtester = HistoricalBacktester(config)
        st.session_state.backtester = run_backtester

        progress = st.progress(0)
        status = st.empty()

        def _on_progress(cur: int, total: int, msg: str) -> None:
            progress.progress(cur / max(total, 1))
            status.text(msg)

        population = None
        market_manager = None
        if ctrl is not None:
            try:
                population = ctrl.model.population
            except Exception:
                population = None
            try:
                market_manager = ctrl.market
            except Exception:
                market_manager = None

        run_result = run_backtester.run_backtest(
            population=population,
            market_manager=market_manager,
            progress_callback=_on_progress,
        )
        progress.empty()
        status.empty()
        st.session_state.backtest_result = run_result
        st.session_state.policy_ab_compare = None

        if run_result.total_days > 0:
            st.success(f"回测完成: {run_result.total_days} 个交易日, {run_result.total_trades} 笔交易")
        else:
            st.error("回测未得到有效结果，请检查参数或数据窗口。")

    result: Optional[BacktestResult] = st.session_state.get("backtest_result")
    if not result:
        st.info("请先配置参数并运行一次回测。")
        return

    frame = _to_frame(result)
    if frame.empty:
        st.warning("回测结果为空。")
        return

    _render_metrics(result)
    left, right = st.columns(2)
    with left:
        _render_equity_chart(frame)
    with right:
        _render_risk_chart(frame)

    if result.factor_snapshot:
        st.markdown("### 因子诊断")
        st.dataframe(pd.DataFrame(result.factor_snapshot).head(20), width="stretch", hide_index=True)
    if result.trade_log:
        st.markdown("### 交易明细")
        st.dataframe(pd.DataFrame(result.trade_log).tail(200), width="stretch", hide_index=True)

    st.markdown("### 回测报告")
    st.markdown(BacktestReportGenerator.generate_html_report(result), unsafe_allow_html=True)

    perf_export = frame.copy()
    perf_export["date"] = perf_export["date"].dt.strftime("%Y-%m-%d")
    perf_csv = perf_export.to_csv(index=False).encode("utf-8-sig")
    payload_json = json.dumps(_result_payload(result), ensure_ascii=False, indent=2).encode("utf-8")

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button("下载绩效序列 CSV", data=perf_csv, file_name="backtest_performance.csv", mime="text/csv", width="stretch")
    with dl_col2:
        st.download_button("下载回测摘要 JSON", data=payload_json, file_name="backtest_summary.json", mime="application/json", width="stretch")
    with dl_col3:
        if st.button("导出 Qlib 数据束", width="stretch"):
            session_backtester: Optional[HistoricalBacktester] = st.session_state.get("backtester")
            if not session_backtester:
                st.error("未找到回测实例，请先运行回测。")
            else:
                try:
                    target_dir = st.session_state.get("backtest_cfg", {}).get("qlib_bundle_path", "outputs/backtest_qlib_bundle")
                    bundle_path = session_backtester.export_qlib_bundle(target_dir)
                    st.success(f"已导出到: {bundle_path}")
                except Exception as exc:
                    st.error(f"导出失败: {exc}")

    st.markdown("### 政策A/B自动对比报告")
    cfg_live = st.session_state.get("backtest_cfg", {})
    base_policy_shock = float(cfg_live.get("policy_shock", 0.0))
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        policy_a = st.number_input("政策A冲击", value=base_policy_shock, step=0.05, key="policy_a_shock_input")
    with p_col2:
        policy_b = st.number_input("政策B冲击", value=base_policy_shock + 0.30, step=0.05, key="policy_b_shock_input")
    with p_col3:
        run_compare = st.button("生成政策A/B自动对比报告", width="stretch")

    if run_compare:
        compare_backtester: Optional[HistoricalBacktester] = st.session_state.get("backtester")
        if not compare_backtester:
            st.error("未找到回测实例，请先运行回测。")
        else:
            with st.spinner("正在生成政策A/B对比回测..."):
                try:
                    generated_compare_bundle = _run_policy_ab_comparison(
                        base_backtester=compare_backtester,
                        policy_a=float(policy_a),
                        policy_b=float(policy_b),
                    )
                    st.session_state.policy_ab_compare = generated_compare_bundle
                    st.success(f"已生成政策A/B对比报告，运行ID: {generated_compare_bundle['run_id']}")
                except Exception as exc:
                    st.error(f"政策A/B自动对比失败: {exc}")

    compare_bundle: Any = st.session_state.get("policy_ab_compare")
    if compare_bundle:
        compare_df = compare_bundle.get("compare_df", pd.DataFrame())
        if isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
            st.dataframe(compare_df, width="stretch", hide_index=True)
            csv_data = compare_df.to_csv(index=False).encode("utf-8-sig")
            json_data = compare_df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")

            dl_a, dl_b = st.columns(2)
            with dl_a:
                st.download_button(
                    "下载政策A/B对比 CSV",
                    data=csv_data,
                    file_name=f"policy_ab_compare_{compare_bundle['run_id']}.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with dl_b:
                st.download_button(
                    "下载政策A/B对比 JSON",
                    data=json_data,
                    file_name=f"policy_ab_compare_{compare_bundle['run_id']}.json",
                    mime="application/json",
                    width="stretch",
                )
        files = compare_bundle.get("files", {})
        if files:
            st.caption(
                "Tear sheet 已写入: "
                f"{files.get('policy_a_html', '')} | {files.get('policy_b_html', '')}"
            )
