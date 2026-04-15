# 时间序列分析模式

## 适用场景
用户提到：趋势、走势、同比、环比、月度、周度、季节性

## 同比 vs 环比
- 同比(YoY)：与去年同期比 → 消除季节性
- 环比(MoM/WoW)：与上一期比 → 看短期变化
- 当数据不足一年时，只能用环比

## ⚠️ 环比计算：必须扩展筛选范围纳入对比基期（必须）

计算环比时，**最常见的 bug 是筛选数据时只选了目标月份，漏掉了对比基期月份**，导致 shift(1) 得到 NaN。

**核心原则：筛选时必须把最早目标月份的前一个月也纳入，计算完后再裁剪输出。**

```python
# 例：用户问 7月、8月、9月 的环比增长率
target_months = [7, 8, 9]

# ✅ 正确：扩展筛选范围，纳入 6 月作为 7 月的对比基期
min_target = min(target_months)
# 用 to_period 计算前一个月（自动处理跨年，如1月→上年12月）
min_target_period = pd.Period(f'{target_year}-{min_target:02d}', freq='M')
prev_period = min_target_period - 1  # 前一个月的 Period

# 筛选时包含对比基期月份
filtered_df = df[
    (df['date'].dt.to_period('M') >= prev_period) &
    (df['date'].dt.to_period('M') <= pd.Period(f'{target_year}-{max(target_months):02d}', freq='M'))
].copy()

# 按月聚合
filtered_df['month'] = filtered_df['date'].dt.to_period('M').astype(str)
monthly_gmv = filtered_df.groupby('month')['gmv'].sum().reset_index()
monthly_gmv = monthly_gmv.sort_values('month')

# 计算环比
monthly_gmv['previous_gmv'] = monthly_gmv['gmv'].shift(1)
monthly_gmv['change_pct'] = round(
    (monthly_gmv['gmv'] - monthly_gmv['previous_gmv']) / monthly_gmv['previous_gmv'] * 100, 2
)

# ✅ 计算完后裁剪：只保留目标月份，去掉辅助用的对比基期行
target_month_strs = [f'{target_year}-{m:02d}' for m in target_months]
result_df = monthly_gmv[monthly_gmv['month'].isin(target_month_strs)].copy()

# ❌ 错误：只筛目标月份，漏掉对比基期
# filtered_df = df[df['date'].dt.month.isin([7, 8, 9])]  ← 6月不在里面，7月环比必然NaN
```

**如果对比基期在原始数据中确实不存在**（如数据从 7 月才开始），则在 summary 中明确说明：
```python
if prev_period.strftime('%Y-%m') not in monthly_gmv['month'].values:
    summary += f"（注：{prev_period} 数据不存在，{min_target}月无法计算环比）"
```

## 时间粒度选择建议
- 数据跨度 > 6个月 → 建议按月
- 数据跨度 1-6个月 → 建议按周
- 数据跨度 < 1个月 → 建议按日

## ⚠️ 月份输出格式规范（必须）

趋势分析按月聚合时，month 列**必须输出 `YYYY-MM` 格式**（如 `2024-07`），**禁止输出纯整数**（如 `7`）。

原因：纯整数月份在跨年数据中无法区分年份，且在图表 x 轴上缺乏可读性。

```python
# ✅ 正确：使用 to_period 生成 YYYY-MM 格式
filtered_df['month'] = filtered_df['date'].dt.to_period('M').astype(str)
# 输出示例：'2024-07', '2024-08', '2025-01'

# ❌ 错误：输出纯整数月份
filtered_df['month'] = filtered_df['date'].dt.month
# 输出示例：7, 8, 1（跨年时无法区分）
```

按月聚合标准写法：
```python
filtered_df['month'] = filtered_df['date'].dt.to_period('M').astype(str)
monthly = filtered_df.groupby('month')[target_col].sum().reset_index()
monthly = monthly.sort_values('month')  # YYYY-MM 格式天然支持字符串排序
```

## 移动平均（平滑噪声）
当日数据波动大时：
rolling_avg = df.groupby(group_col)[target].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

## 周末/节假日效应
电商数据通常周末高于工作日，比较时注意：
- 按周聚合可消除日内波动
- 按月聚合可消除周内波动
- 比较单日数据时，应比较"同一星期几"

### 周末 vs 工作日对比的标准写法
```python
# 判断周末/工作日
df['day_type'] = df['date'].dt.dayofweek.apply(lambda x: 'weekend' if x >= 5 else 'weekday')

# "平均GMV" → 直接 mean，不要先按天 sum 再除天数
avg_by_type = df.groupby('day_type')['gmv'].mean()

# 计算百分比差异
diff_pct = (avg_by_type['weekend'] - avg_by_type['weekday']) / avg_by_type['weekday'] * 100
```
⚠️ 不要写成 `groupby(['day_type','date'])['gmv'].sum()` 再除以天数，那样得到的是"日均总GMV"而非"记录级平均GMV"，值会偏大 N 倍（N = 每天的记录行数）。
