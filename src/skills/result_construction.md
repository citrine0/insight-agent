# result_df 构建规范与列命名标准

## 一、三个输出变量的职责

| 变量 | 职责 | 消费者 |
|------|------|--------|
| `result_df` | **完整数据表**，含过程数据和答案 | 前端表格展示、下游分析 |
| `summary` | **一句话直接回答**用户问题 | 前端文字展示 |
| `answer` | **答案指针**，只含用户问的最终数值。单行结果用 `dict`；多行分组结果用 `list[dict]` | 评测系统精确提取 |

### 核心原则

1. **result_df 保持完整** — 包含上下文数据，让前端表格有意义
2. **summary 直接回答问题** — 用户看到文字就能得到答案
3. **answer 只放最终数值** — key 用下方"列命名标准"的英文名，value 是数字
   - 单行/单值结果：`answer = {"field": value}`
   - 多行分组结果：`answer = [{"group_col": key, "value_col": val}, ...]`，key 必须与 result_df 列名一致
   - 极值对比（最高/最低）：`answer = {"best_dim": val, "best_metric": num, "worst_dim": val, "worst_metric": num}`（扁平 dict，见模式 G）

### 判断什么进 answer

> 把用户问题改写成填空题，空格处需要的数值 = answer 的内容

- 问："前10%贡献了总GMV的___？" → `answer = {"gmv_share_pct": 36.81}`
- 问："斜率___，R²___？" → `answer = {"slope": -12349.58, "r_squared": 0.03}`
- 问："10月比9月变化了___？" → `answer = {"change": 5000.00, "change_pct": 2.35}`
- 问："ROI最高的是___，最低的是___？" → `answer = {"best_platform": "X", "best_roi": 22.03, "worst_platform": "Y", "worst_roi": 13.80}`

---

## 二、列命名标准

列名不规范会导致评测系统无法匹配字段。三条原则：
1. 维度列保持原名（channel, category, date）
2. 数值列用英文、下划线分隔、语义明确
3. 对比分析用 current/previous，**不用具体月份名**

### 聚合类
| 场景 | ✅ 正确 | ❌ 常见错误 |
|------|---------|------------|
| 求和 | total_gmv | gmv_sum, 总gmv |
| 均值 | avg_gmv | mean_gmv, 平均GMV |
| 加权平均 | weighted_avg_cvr | weighted_cvr, 加权转化率 |
| 计数 | record_count | cnt, num, 条数 |

### 对比/变化类
| 场景 | ✅ 正确 | ❌ 常见错误 |
|------|---------|------------|
| 当期值 | current_gmv | aug_gmv, oct_gmv, 8月GMV |
| 上期值 | previous_gmv | jul_gmv, sep_gmv, 7月GMV |
| 绝对变化 | change | gmv_change, diff, 变化量 |
| 环比变化率 | change_pct | mom_pct, mom_change, 环比变化率(%) |
| 同比变化率 | yoy_pct | yoy_change, 同比% |
| 增长率 | growth_rate_pct | growth, 增长率% |

### 占比/派生类
| 场景 | ✅ 正确 | ❌ 常见错误 |
|------|---------|------------|
| 占比 | gmv_share_pct | share, 占比(%), gmv占比 |
| ROI | roi | ROI, Roi |
| 客单价 | avg_order_value | 客单价, aov, unit_price |

---

## 三、构建模式

### A. 统计量 / 单值结果

```python
result_df = pd.DataFrame({
    'metric': ['slope', 'r_squared'],
    'value': [round(slope, 4), round(r_squared, 4)]
})
summary = f"斜率{slope:.2f}，R²{r_squared:.4f}"
answer = {"slope": round(slope, 4), "r_squared": round(r_squared, 4)}
```

### B. 占比 / 贡献度

```python
share_pct = round(top_gmv / total_gmv * 100, 2)
result_df = pd.DataFrame({
    'metric': ['total_records', 'top_n_records', 'total_gmv', 'top_gmv', 'gmv_share_pct'],
    'value': [len(oct_df), top_n, round(total_gmv, 2), round(top_gmv, 2), share_pct]
})
summary = f"前10%记录贡献了总GMV的{share_pct}%"
answer = {"gmv_share_pct": share_pct}
```

占比分组标准写法：
```python
grouped = filtered.groupby('channel')['gmv'].sum().reset_index()
total = grouped['gmv'].sum()
grouped['gmv_share_pct'] = round(grouped['gmv'] / total * 100, 2)
result_df = grouped.sort_values('gmv_share_pct', ascending=False)
```

### C. 对比 / 环比结果

```python
# ✅ 列名用 current/previous，不绑定具体月份
result_df = pd.DataFrame({
    'period': ['对比结果'],
    'current_gmv': [round(oct_gmv, 2)],
    'previous_gmv': [round(sep_gmv, 2)],
    'change': [change],
    'change_pct': [change_pct]
})
summary = f"10月GMV较9月变化{change_pct:.2f}%"
answer = {"change": change, "change_pct": change_pct}
```

### D. 分组聚合（多行 → list[dict]）

```python
result_df = grouped[['channel', 'total_gmv', 'roi']]
summary = f"共{len(result_df)}个渠道，GMV最高: {result_df.iloc[0]['channel']}"
answer = result_df[['channel', 'total_gmv']].to_dict(orient='records')
```

单行极值提取：
```python
answer = {"channel": result_df.iloc[0]['channel'], "total_gmv": round(result_df.iloc[0]['total_gmv'], 2)}
```

### E. 明细 + 汇总

```python
avg_growth = round(monthly_data['change_pct'].mean(), 2)
avg_row = pd.DataFrame({'period': ['平均'], 'total_gmv': [None], 'change_pct': [avg_growth]})
result_df = pd.concat([monthly_data, avg_row], ignore_index=True)
summary = f"平均月环比增长率{avg_growth}%"
answer = {"growth_rate_pct": avg_growth}
```

### F. 时序 + 最终值

```python
last_ma = daily_data.iloc[-1]['gmv_7d_ma']
result_df = daily_data[['date', 'daily_gmv', 'gmv_7d_ma']]
summary = f"最后一天的7日移动平均为{last_ma:,.2f}"
answer = {"gmv_7d_ma": round(last_ma, 2)}
```

### G. 极值对比（最高/最低、best/worst）

当问题同时要求最高和最低（或 best 和 worst），answer 用**扁平 dict + `best_`/`worst_` 前缀**，不用 `list[dict]`。

```python
highest = sorted_df.iloc[0]
lowest = sorted_df.iloc[-1]

result_df = pd.DataFrame({
    'platform': [highest['platform'], lowest['platform']],
    'product_line': [highest['product_line'], lowest['product_line']],
    'roi': [round(highest['roi'], 4), round(lowest['roi'], 4)],
    'rank': ['最高', '最低']
})
summary = f"ROI最高: {highest['platform']}-{highest['product_line']}({highest['roi']:.4f})，最低: {lowest['platform']}-{lowest['product_line']}({lowest['roi']:.4f})"
# ✅ 扁平 dict，维度+数值都带 best_/worst_ 前缀
answer = {
    "best_platform": highest['platform'],
    "best_product_line": highest['product_line'],
    "best_roi": round(highest['roi'], 4),
    "worst_platform": lowest['platform'],
    "worst_product_line": lowest['product_line'],
    "worst_roi": round(lowest['roi'], 4)
}
```

> **为什么不用 `list[dict]`？** 评测系统对 list 只能按首个字符串列构建复合键，无法映射 `best_`/`worst_` 语义，导致字符串维度字段匹配失败。扁平 dict 走精确键匹配，6/6 命中。

---

## 四、检查清单

1. ✅ `result_df` 是否包含了用户问题的最终答案数值？
2. ✅ `summary` 是否直接回答了用户问题？
3. ✅ 单行结果是否设了 `answer`（dict）？多行分组结果是否设了 `answer`（list[dict]）？极值对比是否用了扁平 dict + `best_`/`worst_` 前缀（模式 G）？
4. ✅ 列名是否使用上方标准英文名？（特别是对比分析：current/previous/change/change_pct）
