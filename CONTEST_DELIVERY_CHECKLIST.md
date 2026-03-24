# CONTEST_DELIVERY_CHECKLIST

## Reproducibility Gate
- [ ] Every major run emits:
  - [ ] `git_commit`
  - [ ] `config_hash`
  - [ ] `seed`
  - [ ] `dataset_snapshot_id`
- [ ] Event visibility respects `visibility_time` (no future leakage).
- [ ] Scenario manifest and snapshot manifest are archived.

## Functional Gate
- [ ] Policy lab remains launchable.
- [ ] History replay remains launchable.
- [ ] Legacy interfaces remain backward compatible.
- [ ] Feature flags can disable new high-risk paths.

## Testing Gate
- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] Determinism/regression tests pass.
- [ ] Core required checks:
  - [ ] agent replay determinism
  - [ ] policy parser determinism
  - [ ] execution plan -> child orders
  - [ ] trade tape -> OHLCV bars
  - [ ] regulator optimization adapter determinism

## Reporting Gate
- [ ] Realism report exported (markdown/json/docx/pdf).
- [ ] Policy A/B report exported.
- [ ] Design chapter draft exported.
- [ ] Architecture / causal graph exported.
- [ ] Defense outline exported.

## Compliance Gate
- [ ] `THIRD_PARTY_OPEN_SOURCE_DISCLOSURE.md` updated.
- [ ] `AI_TOOL_DISCLOSURE.md` updated.
- [ ] This checklist completed for final submission package.
