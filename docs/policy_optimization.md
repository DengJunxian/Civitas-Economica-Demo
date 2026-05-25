# 监管优化与 Pareto 输出

## 1. 目标

监管优化模块回答的问题是：在市场承压、谣言冲击或政策预期变化时，监管者应该选择哪种干预组合，才能在稳定指数、控制风险、减少交易扭曲和降低政策成本之间取得更优权衡。

## 2. 优化入口

主要路径：

- `core/optimization/objectives.py`
- `core/optimization/constraints.py`
- `core/optimization/bayes_search.py`
- `core/optimization/nsga_search.py`
- `core/optimization/report_generator.py`
- `ui/regulator_optimization.py`
- `ui/components/optimization_report_panel.py`

项目保留轻量 black-box search 和多目标搜索入口。BoTorch、Ax、Nevergrad、Optuna 的思想被用于参数搜索和多目标权衡设计，但不作为项目启动必需依赖。

## 3. 目标函数

典型目标：

- 最小化指数路径误差。
- 最小化最大回撤和尾部风险。
- 最小化恐慌度和羊群强度。
- 最大化事件窗方向命中率。
- 控制政策成本、交易扭曲和流动性副作用。

约束：

- 政策强度上下限。
- 干预次数。
- 监管动作冷却期。
- 市场会话规则。
- 成本预算。

## 4. Pareto 输出

监管页面应展示：

- 候选方案列表。
- Pareto front。
- 推荐方案。
- 推荐理由。
- 与基线/替代方案的 A/B 差分。
- 风险提示。
- 复现信息：experiment_id、config_hash、snapshot hash、seed、parameter_set_id。

## 5. 运行中追加事件

政策实验运行中可以追加：

- 重大新闻。
- 新政策。
- 监管动作。
- 宏观冲击。
- 谣言和辟谣。

这使监管优化不是一次性静态调参，而是能随事件演化持续更新。

## 6. 答辩讲法

建议表述：

“我们不是让大模型直接推荐监管政策，而是把候选监管动作放入可复现的市场沙箱中，分别运行反事实路径，计算风险、微观结构和行为指标，再输出 Pareto front。LLM 负责解释和结构化，优化器负责搜索，撮合引擎负责生成市场事实。”

