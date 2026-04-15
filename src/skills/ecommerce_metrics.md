# 电商指标计算规范

## 适用场景
用户查询涉及：GMV、转化率、客单价、ROI、ROAS、复购率等电商术语

## 标准计算公式

### GMV (Gross Merchandise Volume)
gmv = traffic × conversion_rate × avg_price
⚠️ 如果数据中已有 gmv 字段，直接使用，不要重复计算

### ROI / ROAS
roi = sum(gmv) / sum(marketing_spend)
⚠️ 必须用 sum/sum，禁止 mean(gmv/marketing_spend)

### 客单价
avg_order_value = sum(gmv) / sum(traffic × conversion_rate)
⚠️ 不是 mean(avg_price)，因为 avg_price 是商品均价，不是订单均价

### ⚠️ 加权平均价格（区分"算客单价" vs "对已有价格做加权"）

当用户说"加权平均客单价/价格"且**指定了加权方式**（如"按流量加权"）时：
- 数据中已有 avg_price 字段 → 直接加权：`Σ(traffic × avg_price) / Σ(traffic)`
- 数据中已有 competitor_price 字段 → 同理：`Σ(traffic × competitor_price) / Σ(traffic)`
- **禁止**用 GMV÷订单数 倒推，那是"订单均价"，不是"按流量加权的平均价格"

判断依据：用户说"按X加权"→ 公式必须是 `Σ(X × 价格) / Σ(X)`，自身价格和竞品价格用同一套加权逻辑。

### 转化效率 / 单次转化产出
两种场景，选错结果偏差 1%-5%：

场景A —— 问题含"总X / 总Y"、"有效客单价"：
```python
# 分子分母各自汇总后再除
result = sum(gmv) / sum(traffic * conversion_rate / 100)
```

场景B —— 问题给出逐条记录公式，如"GMV / (流量×转化率/100)"：
```python
# 逐行算比率，再按分组求均值
per_row = gmv / (traffic * conversion_rate / 100)   # 逐行
result = per_row.groupby('channel').mean()           # 再分组平均
```

判断依据：
- 公式中的变量代表"一个群体的汇总"，含"总"字 → 场景A
- 公式中的变量代表"某一行的值"，给出的是逐条计算公式 → 场景B

### 转化率聚合
weighted_conversion = sum(traffic × conversion_rate) / sum(traffic)
⚠️ 禁止 mean(conversion_rate)，因为各渠道流量不同

### 营销效率
cost_per_acquisition = sum(marketing_spend) / sum(traffic × conversion_rate)

## 维度下钻顺序（根因分析时）
1. channel（渠道）→ 哪个渠道异常
2. category（品类）→ 哪个品类异常  
3. channel × category（交叉）→ 精确定位
4. 时间维度 → 什么时候开始异常
