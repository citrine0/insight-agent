# 统计分析方法指南

## 适用场景
用户提到：平均、波动、异常、分布、集中趋势、离散程度、显著性

## 核心原则

### ⚠️ 禁止擅自预聚合（最常见的错误）

**原始数据是多维明细（日期×渠道×品类），除非用户明确要求特定粒度，否则直接对分组后的原始记录计算统计指标，不要额外做时间/维度聚合。**

这条原则影响所有统计计算：mean、std、变异系数、分位数等。

#### 典型错误 1：平均值计算
```python
# 用户问："周末的平均GMV比工作日高出多少？"

# ❌ 错误：先按天 sum 再除以天数 → 得到"日均总GMV"（值偏大 N 倍）
daily = october_df.groupby(['day_type', 'date'])['gmv'].sum().reset_index()
avg_wrong = daily.groupby('day_type')['gmv'].mean()
# weekend: 7,742,133  ← 是每天所有渠道×品类的总和的平均

# ✅ 正确：直接 mean → 得到"记录级平均GMV"
avg_correct = october_df.groupby('day_type')['gmv'].mean()
# weekend: 483,883   ← 每条记录的平均
```

#### 典型错误 2：离散度计算
```python
# 用户问："各渠道GMV的变异系数"

# ❌ 错误：先按日聚合再算 CV → 品类差异被 sum 抹平，CV 严重偏低
daily_gmv = october_df.groupby(['channel', 'date'])['gmv'].sum().reset_index()
cv_wrong = daily_gmv.groupby('channel')['gmv'].apply(lambda x: x.std() / x.mean() * 100)

# ✅ 正确：直接对原始记录算 CV
cv_correct = october_df.groupby('channel')['gmv'].apply(lambda x: x.std() / x.mean() * 100)
```

#### 什么时候需要预聚合？
只有用户**明确提到**时间粒度关键词时才预聚合：
| 用户说的 | 做法 |
|---------|------|
| "平均GMV" / "GMV变异系数" | 直接对原始记录算，**不要预聚合** |
| "**日均**GMV" / "每天的平均GMV" | 先 groupby('date').sum()，再 mean |
| "**日**GMV的波动" | 先 groupby('date').sum()，再算 std/CV |
| "**月**GMV的波动" | 先 groupby('month').sum()，再算 std/CV |

### 比率类指标的正确聚合
❌ 错误：mean(conversion_rate)  → 简单平均，忽略流量权重
✅ 正确：sum(gmv) / sum(traffic * avg_price) → 加权计算

### 增长率的正确计算
❌ 错误：mean(每日增长率) → 波动大时严重失真
✅ 正确：(期末值 - 期初值) / 期初值 → 整体增长率

### 离散度/波动度分析（变异系数、标准差等）

当用户问"波动最大"、"最稳定"、"变异系数"时，遵循上面的**禁止擅自预聚合**原则。

#### 比率指标的离散度
对转化率等比率指标，如需计算离散度，应先正确加权聚合到目标粒度：
```python
# 例："各渠道日转化率的波动"（用户明确说了"日"）
daily = october_df.groupby(['channel', 'date']).agg(
    total_gmv=('gmv', 'sum'),
    total_traffic=('traffic', 'sum')
).reset_index()
daily['weighted_cr'] = daily['total_gmv'] / (daily['total_traffic'] * daily['avg_price_需另算'])
# 再对 weighted_cr 算变异系数
```

### 异常值检测
当用户问"为什么突然下降/上升"时：
1. 先计算 IQR (Q3-Q1)
2. 标记 < Q1-1.5*IQR 或 > Q3+1.5*IQR 的数据点
3. 对异常点进行维度下钻（按channel、category分别看）

### 贡献度分解
当用户问"哪个因素影响最大"时：
- 使用差值分解法：overall_change = Σ(segment_weight × segment_change)
- 按贡献度绝对值排序，返回 top N

### 相关性分析
当用户问"X和Y有没有关系"时：
- 使用 df[col_x].corr(df[col_y])
- |r| > 0.7 强相关，0.3-0.7 中等，< 0.3 弱相关
- 提醒：相关不等于因果