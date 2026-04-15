# Python Agent 常见陷阱

> 比率指标 sum/sum、禁 sklearn、预聚合禁令等硬性规则已在 System Prompt 中定义，此处只列出需要代码示例才能避免的陷阱。

## DataFrame 赋值陷阱
筛选后的子集赋新列必须用 `.copy()`，否则触发 SettingWithCopyWarning：
```python
# ❌ 触发警告
oct_df = df[df['date'].dt.month == 10]
oct_df['day_type'] = 'weekday'

# ✅ 筛选时加 .copy()
oct_df = df[df['date'].dt.month == 10].copy()
oct_df['day_type'] = 'weekday'
```

## 日均 / 日级统计量陷阱
当用户问"日均""日波动""每天平均"等以**天**为单位的统计量时，**必须先按天聚合，再对天级数据求统计量**。
直接用 `行数` 做除数是错误的——因为同一天通常有多条记录（每个产品线/渠道各一行）。

```python
# ❌ 错误：用行数代替天数
may_df['day_type'] = may_df['order_date'].dt.dayofweek.apply(
    lambda x: 'weekend' if x >= 5 else 'weekday')
weekend_rows = (may_df['day_type'] == 'weekend').sum()   # 120 行，不是 8 天！
weekday_rows = (may_df['day_type'] == 'weekday').sum()   # 345 行，不是 23 天！
avg_weekend = may_df[may_df['day_type']=='weekend']['revenue'].sum() / weekend_rows  # ❌

# ✅ 正确：先按天汇总 → 再按类型求均值
may_df['day_type'] = may_df['order_date'].dt.dayofweek.apply(
    lambda x: 'weekend' if x >= 5 else 'weekday')
daily = may_df.groupby(['order_date', 'day_type'])['revenue'].sum().reset_index()
type_avg = daily.groupby('day_type')['revenue'].mean().reset_index()
type_avg.columns = ['day_type', 'avg_daily_revenue']
```

**判断规则**：只要问题涉及"日均""日波动""每天""天数"等词，一律走 `groupby(date).sum()` → 天级 DataFrame → 再算 mean/std/count。
唯一天数用 `.nunique()` 而非 `.sum()` 或 `.count()`。

## 聚合顺序陷阱
❌ 先 round 再排序/取极值 → ✅ 先排序再 round
```python
# ❌ round(2) 后 0.4050 和 0.4144 都变成 0.41，排名不确定
# ✅ sort_values() 之后，仅在最终输出时 round
```

## 百分比切片陷阱
"前10%记录"用 int() 向下截断，不要用 ceil 向上取整：
```python
# ❌ ceil(496 * 0.1) = 50 → 占比10.08%，超出10%
# ✅ int(496 * 0.1) = 49 → 占比9.88%，不超过10%
top_n = int(total * 0.1)
```

## 线性回归陷阱
手动实现时 X 和 y 必须都是 1D 数组：
```python
# ❌ reshape(-1,1) → (X-mean)*(y-mean) 广播成 (31,31) 矩阵，slope≈0
X = daily_gmv['day_num'].values.reshape(-1, 1)

# ✅ 1D 向量，逐元素相乘
X = daily_gmv['day_num'].values  # shape=(31,)
y = daily_gmv['daily_total_gmv'].values
slope, intercept = np.polyfit(X, y, 1)  # 最简写法
```

## 数据类型陷阱
1. date 列可能是 string → 必须 pd.to_datetime() 后再比较
2. 数值列可能有 NaN → 聚合前考虑 fillna(0) 或 dropna()
3. 分类列大小写 → 用 .str.lower() 或检查 enum_values

## 时间筛选陷阱
"2025-8-1" > "2025-07-31" 字符串排序错误 → 必须转 datetime 后比较
月份筛选如果跨年，需要同时限制 year
