# 对比分析方法论

## 适用场景
用户提到：对比、比较、差异、变化、增减、涨跌

## 绝对值 vs 百分比
- 小基数变大百分比（100→200 = +100%）但绝对值小
- 大基数变小百分比（1M→1.1M = +10%）但绝对值大
- 建议：同时展示绝对变化和百分比变化

## 多维度对比的展示
当对比 N 个渠道/品类时：
- N ≤ 5：并排柱状图
- N > 5：横向柱状图（方便阅读标签）
- 突出差异最大的项（颜色标注）

## 归因分析（Mix Effect vs Rate Effect）
GMV变化 = 流量变化效应 + 转化率变化效应 + 价格变化效应
- 流量效应 = (new_traffic - old_traffic) × old_cr × old_price
- 转化效应 = old_traffic × (new_cr - old_cr) × old_price
- 价格效应 = old_traffic × old_cr × (new_price - old_price)