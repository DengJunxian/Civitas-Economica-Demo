# Civitas 优化设计（绩效评估 + 组合系统 + 跨学科传播）

日期: 2026-03-15

## 1. 目标与范围

本次改造聚焦三类能力：

1. 回测可信度: 将风险与绩效指标从回测执行引擎中解耦，形成统一指标层，支持仿真/回测/反事实复用。  
2. 组合系统化: 引入统一组合构建层，使策略输出从单点信号升级为可约束、可比较、可执行的权重系统。  
3. 跨学科融合: 为社会传播层提供标准模型后端接口（NDlib/EoN 兼容），并补齐行为金融 CPT 参数化能力。

## 2. 架构改动

### 2.1 风险绩效与报告

- 新增 `core/performance.py`  
  - `compute_performance_metrics`: 统一计算 Sharpe/Sortino/Calmar/Alpha/Beta/IR/VaR/CVaR/Omega/Tail Ratio/Stability。  
  - `compute_backtest_credibility`: 结合样本规模、回撤、波动、换手得到可信度分值。  
  - 兼容 `empyrical`（可选依赖），无依赖时自动 fallback。

- 新增 `core/tear_sheet.py`  
  - `build_standard_tear_sheet`: 标准化 tear sheet 载荷。  
  - `compare_scenarios`: 多情景对比表（政策 A/B、有无国家队、不同熔断阈值）。  
  - `export_tear_sheet_html/json` 与 `export_quantstats_html`（可选依赖 quantstats）。

- 回测集成 (`core/backtester.py`)  
  - 回测结果新增 tail risk 与可信度字段。  
  - `BacktestReportGenerator` 新增 tear sheet payload 与文件导出接口。

### 2.2 组合构建层

- 新增 `core/portfolio/construction.py`  
  - `PortfolioConstructionLayer`: `equal_weight / inverse_vol / mean_variance / hrp` 统一接口。  
  - 支持政策风险、情绪风险惩罚项。  
  - 支持行业暴露上限、换手约束、目标最大回撤防线。

- 回测策略扩展  
  - 新增 `portfolio_system` 策略（风险资产 + 现金的约束式权重分配），用于把政策/情绪约束直接映射到可执行仓位。

### 2.3 社会传播与行为金融融合

- `core/society/network.py` 新增：
  - `NDlibBackendAdapter`: 标准传播模型后端适配（可选依赖，缺失时回退到本地扩散）。  
  - `EoNBackendAdapter`: EoN 数值传播适配（缺失时回退离散 SIR 近似）。

- `core/behavioral_finance.py` 新增：
  - `CPTAgentProfile` 与 `get_cpt_profile`（retail/institution/quant）。  
  - `cpt_decision_utility` / `cpt_expected_utility`，将 CPT 价值函数与概率加权固化为可复用组件。

## 3. 许可与复用边界

1. 允许优先复用（Apache/MIT/BSD）  
   - pyfolio, QuantStats, NDlib, PyPortfolioOpt, OASIS, EoN, CPT 仓库（MIT/BSD/Apache）。  
2. GPL 风险控制  
   - `cvxportfolio (GPLv3)` 仅作方法参考，不直接复制代码。  
   - 如需凸优化代码复用，优先使用 `cvxportfolio-FORM (Apache-2.0)` 或自行重写数学目标。  
3. 未明确许可仓库  
   - 仅参考思路，不直接搬运实现。

## 4. 风险与后续

1. Optional dependency 策略  
   - 所有外部生态库均为可选依赖，不影响核心运行；测试覆盖 fallback。  
2. 当前回测仍以单标的执行为主  
   - 组合构建层已就位；下一步建议将多资产行情输入纳入 `HistoricalBacktester`，把组合层从“策略内嵌”升级为“执行主路径”。  
3. 推荐新增评估工序  
   - 固化 “基线 + 多情景 + Walk-forward + tear sheet 对比” 为统一实验模板。

