# PythonAgent 准确率优化记录

## 背景

PythonAgent 是整个系统的执行层——接收自然语言查询 + 数据 Schema，生成 pandas 代码并执行。它的准确率直接决定了下游所有环节（异常检测、因果推理、报告）的质量。

---

## 问题定义

直接用 LLM 生成 pandas 代码，主要失败模式集中在三类：

| 失败类型 | 示例 |
|---------|------|------|
| 聚合方式错误 | 转化率用 `mean()` 而非 `sum(分子)/sum(分母)` |
| 时间窗口计算错误 | "上月"用 `datetime.now()` 而非数据最新日期推算 |
| 输出格式不符 | 环比对比输出多行（按月分行）而非单行对比结果 |
| 其他 | 列名映射错误、过滤值中英文不匹配等 |


---

## 解决方案: Skill 动态注入系统

而非逐个 case 硬编码修复，设计了可插拔的 Skill 注入机制：

```
┌─ System Prompt ─────────────────────────────────┐
│  Layer 1: Base（角色定义，稳定）                   │
│  Layer 2: Skills（按查询内容动态加载）              │  ← SkillLoader 注入
│  Layer 3: Output Rules（输出格式，稳定）            │
│  Layer 4: Hard Constraints（不可违反规则，尾部）    │
└─────────────────────────────────────────────────┘
```

**SkillLoader 工作原理**: 扫描用户查询中的关键词 → 匹配相关 Skill 文件 → 将领域知识注入 System Prompt 的 Layer 2 位置。

### Skill 文件清单

| Skill 文件 | 触发关键词 | 解决的问题 |
|-----------|-----------|-----------|
| common_pitfalls.md | 始终加载 | 通用陷阱（datetime.now 禁用、列名映射等） |
| result_construction.md | 始终加载 | 输出变量规范（result_df / answer / summary） |
| ecommerce_metrics.md | ROI、转化率、客单价 | ratio 指标必须用 sum(分子)/sum(分母) |
| time_analysis.md | 趋势、同比、环比 | 时间窗口计算 + 环比对比输出单行 |
| statistical_methods.md | 平均、标准差、分布 | 统计量计算方法 |
| comparison_analysis.md | 对比、增长、归因 | 对比分析列命名规范 |

---

## 准确率变化（基准评测集）

| 配置 | 准确率 | 变化 | 说明 | app版本|
|------|-------|------|------|-----|
| 无 Skill（基线） | [51]% | — | 仅 Base + Output Rules + Hard Constraints | v4_5_3 |
| 全部 Skills | [97.1]% | +[_46.1] | 当前生产配置 | v5_5-1 |


---

## 关键发现

### 1. ratio 聚合是最大的错误来源

[描述: ecommerce_metrics Skill 加入前后的具体变化。例如——

加入前: LLM 对"各渠道转化率"生成 `df.groupby('channel')['conversion_rate'].mean()`，
这是错误的，因为不同渠道流量不同，应该用加权平均。

加入后: Skill 注入了"转化率 = sum(支付买家数)/sum(访客数)"的规则，
LLM 生成正确的 `sum(buyers)/sum(traffic)` 代码。]

### 2. Hard Constraints 放在尾部的效果

[描述: 将"ratio 必须用 sum/sum"从 Layer 1 移到 Layer 4 后的准确率变化。
DeepSeek 模型对 System Prompt 尾部内容的遵循度更高。]

### 3. 仍存在的失败模式

| 失败模式 | 占当前错误比例 | 改进方向 |
|---------|-------------|---------|
| [评测脚本比对错误] | [2.9]% | [暂无] |

---

## 评测方法

- **基准测试集**: [35] 个业务查询，数据源为 test_data_business_metrics.csv（4992行），覆盖基础聚合（sum/grouped/filtered）、环比计算（6题）、加权平均（3题）、多维度分析（3题）及统计分析（标准差/百分位/相关性/趋势等）。Complex 前置能力组占比 49%，平均难度 6.3。
- **complex测试集**: [35] 个业务查询，与基准集使用相同数据源但题目完全不重复（通过切换月份/维度/指标实现差异化）。Complex 前置能力组占比提升至 86%（30题），其中环比计算 9 题、加权平均 7 题、基础聚合 8 题、多维度分析 6 题。难度分布与基准集一致（平均 6.3），大幅增加了 traffic/marketing_spend/avg_price 等非 GMV 指标覆盖。
- **schema_adapt测试集**: [35] 个业务查询，35 个业务查询，更换为全新数据源 test_data_schema_adapt.csv（1005行），列名全部变更（date→order_date, channel→platform, category→product_line, gmv→revenue 等），维度值从国内电商（Douyin/JD/Tmall/WeChat）切换为跨境电商（Amazon/Shopee/Lazada），品类从 4 个变为 5 个，时间范围从 2025全年缩短至 2024年4-6月。前置能力组占比 86%，题型结构与 Complex 集对齐，核心测试模型对未知 schema 的自适应能力而非对特定列名的记忆。
- **评判标准**: 代码执行成功 + 结果数值正确 + 输出格式符合规范
- **执行方式**: [评测脚本 / 对应评测集]

