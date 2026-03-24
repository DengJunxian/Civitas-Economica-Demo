# THIRD_PARTY_OPEN_SOURCE_DISCLOSURE

## Purpose
This file tracks third-party open-source components used in Civitas-Economica-Demo for reproducible contest delivery.

## Core Dependencies (Runtime)
| Component | License | Usage Scope | Notes |
| --- | --- | --- | --- |
| Python | PSF | Runtime platform | Main execution runtime |
| NumPy | BSD-3-Clause | Numerical computing | Vectorized simulation/math |
| pandas | BSD-3-Clause | DataFrame pipeline | Backtest/replay/event store IO |
| Streamlit | Apache-2.0 | UI runtime | Interactive dashboards |
| Plotly | MIT | Visualization | UI charts and export bundles |
| reportlab | BSD | PDF export | Reporting pipeline |
| python-docx | MIT | DOCX export | Formal report generation |
| pyarrow | Apache-2.0 | Parquet IO | Event Store storage format |
| pytest | MIT | Testing | Unit/integration/regression suites |

## External Data / Service Tooling
| Component | License / Terms | Usage Scope | Notes |
| --- | --- | --- | --- |
| AkShare | MIT | Market data adapter | Optional fallback provider |
| yfinance | Apache-2.0 | Market data adapter | Optional fallback provider |

## Internal Attribution Rule
- `core/*`, `agents/*`, `ui/*` under this repository are treated as project source.
- Reused OSS libraries are dependency-level only unless explicitly copied into source.

## Maintenance Checklist
1. Update this file when adding/removing dependency packages.
2. Verify license compatibility before contest submission.
3. Keep dependency lockfiles consistent with this disclosure.
