# History replay report: Case replay

- Report no: CIVITAS-HIS-20260324-153826
- Date: 2026年03月24日
- Engine mode: agent
- Feature flags: {'agent_replay': True, 'history_replay_event_driven_v2': True, 'history_replay_rolling_calibration_v1': True}

## Replay summary
- Symbol: CSI 300
- Date window: 2024-02-01 to 2024-02-07
- Backdrop: test
- Intensity: 1.0
- Explanation: Case replay: path shape is close to the historical series.

## Metrics
- Trend alignment: 75%
- Turning points: 50%
- Drawdown gap: 2.00%
- Vol similarity: 80%
- Response lag: 1 days

## Authenticity layers
### Path layer
- Return correlation: 0.983
- Normalized RMSE: 0.0400
### Microstructure layer
- Range fit: 95%
- Volume fit: 0.966
- Cancel-rate proxy: 62%
### Stylized facts layer
- Tail fit: 50%
- Volatility clustering: 0.623
- Regime-switch alignment: 100%

## Risks
- Replay result is intended for defense, comparison, and calibration.
- Simulated prices in agent mode come from the trade-tape close, not equity scaling.
- All reproducibility info is recorded in the payload metadata.