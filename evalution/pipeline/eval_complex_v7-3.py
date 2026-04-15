
"""
Complex 路径评测脚本 v6.0 (适配 app_v9 + 单平台单表 MVP)
========================================================
基于 v5.1, 适配 v9 架构升级:
  1. ✅ Router 二分类 (VALID/INVALID), 所有合法查询 route 都是 "complex"
  2. ✅ 新增 analysis_depth 评测 (descriptive / diagnostic / causal)
  3. ✅ 单平台单表数据格式 (店铺经营概况.csv, 中文列名)
  4. ✅ 保留 Layer C 误差归因分析

v6.0 相对 v5.1 的改动:
  [适配 app_v9 统一管线]
  1. RouteValidator 改为 AnalysisDepthValidator (软匹配, causal↔diagnostic 可容)
  2. V8TraceAnalyzer 新增 analysis_depth 属性
  3. MockAgent 输出 analysis_depth 字段, route 统一为 "complex"

  [适配单平台单表]
  4. DIMENSION_KEYWORD_MAP 扩展中文列名 (支付转化率/访客数/客单价/支付金额 等)
  5. _load_data 简化: 单文件加载 (店铺经营概况.csv)
  6. _run_agent 移除 _merge_tables 降级路径 (app_v9 接受 pd.DataFrame)

用法:
  python eval_complex_v5_1.py --eval-file eval/cases.json --data-dir eval/datasets/
  python eval_complex_v5_1.py --eval-file eval/cases.json --data-dir eval/datasets/ \\
      --agent-module app_v9 --run-step3
"""

import argparse
import glob
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================================
# 1. 配置
# ============================================================================

SCORING_WEIGHTS = {
    "analysis_depth":   0.05,  # v6: 原 route 维度,改为 analysis_depth 软匹配
    "scan_quality":     0.15,  # Step 1: 异常扫描质量
    "execution":        0.10,  # 执行效率
    "root_cause":       0.20,  # Step 2: 根因正确性
    "deep_rca_trigger": 0.15,  # Step 2: needs_deep_rca + suggested_data
    "report":           0.10,  # 报告完整性
    "reasoning":        0.15,  # 因果推理质量
    "deep_rca":         0.10,  # Step 3: 深度分析质量 (仅 B 组有效)
}

# ── v6: 扩展维度关键词映射, 同时覆盖英文字段、电商后台字段、中文列名 ──
DIMENSION_KEYWORD_MAP = {
    # === 转化率相关 ===
    "conversion_rate":   ["conversion", "转化", "cvr", "uv_pay_rate", "pay_conversion_rate",
                          "转化率", "支付转化", "支付转化率"],
    # === 流量相关 ===
    "traffic":           ["traffic", "流量", "visit", "uv", "pv", "访客", "访客数", "浏览量",
                          "bounce_rate", "跳失率", "avg_stay_seconds", "平均停留时长",
                          "加购人数", "add_cart"],
    # === 价格 / 客单价 ===
    "avg_price":         ["price", "价格", "客单价", "均价", "aov", "avg_price",
                          "avg_order_amount", "avg_item_price", "件单价", "笔单价",
                          "actual_selling_price", "listing_price", "unit_price", "售价"],
    # === 竞品 ===
    "competitor_price":  ["competitor", "竞品", "comp_price", "竞争",
                          "competitor_price", "competitor_monitor", "竞品监控",
                          "竞品售价", "竞品店铺"],
    # === 营销 / 广告 ===
    "marketing_spend":   ["marketing", "营销", "推广", "广告", "spend",
                          "ad_campaign", "cost", "投放", "roi", "cpc", "ctr",
                          "impressions", "clicks", "花费", "点击率", "展现量",
                          "广告投放", "平均点击花费"],
    # === GMV / 销售额 ===
    "gmv":               ["gmv", "销售额", "revenue", "pay_amount", "支付金额",
                          "pay_order_cnt", "订单数", "支付订单", "支付订单数",
                          "支付买家数", "支付件数"],
    # === 退款 ===
    "refund_rate":       ["refund", "退货", "退款", "refund_amount", "refund_order_cnt",
                          "售后", "退款率", "退款金额", "退款订单数"],
    # === 加购 / 漏斗 ===
    "add_cart":          ["add_cart", "加购", "加购件数", "加购率", "加购人数"],
    # === 库存 ===
    "inventory":         ["inventory", "库存", "缺货", "out_of_stock", "available_qty",
                          "发货时长", "avg_ship_hours", "可用库存"],
    # === 促销 ===
    "promotion":         ["promotion", "促销", "活动", "优惠", "折扣", "promo",
                          "coupon", "discount", "促销日历", "活动名称"],
}

# ── v5.1: suggested_data 从抽象概念 → 具体报表文件名 ──
# 评测时同时接受旧枚举和新文件名
SUGGESTED_DATA_TYPES = [
    # 新版: 具体报表文件名 (eval_plan_v2)
    "ad_campaign",
    "order_detail",
    "competitor_monitor",
    "inventory_status",
    "promotion_calendar",
    "product_performance",
    "refund_after_sale",
    # 旧版: 保留向后兼容
    "marketing_detail",
    "customer_segment",
    "competitor_pricing",
    "inventory_logistics",
    "traffic_source_detail",
    "product_sku_detail",
    "market_benchmark",
]

# ── v5.1: 新旧 suggested_data 别名映射 (双向匹配) ──
SUGGESTED_DATA_ALIASES = {
    "marketing_detail":      ["ad_campaign", "ad_campaign.csv"],
    "ad_campaign":           ["marketing_detail"],
    "ad_campaign.csv":       ["marketing_detail", "ad_campaign"],
    "customer_segment":      ["order_detail", "order_detail.csv"],
    "order_detail":          ["customer_segment"],
    "order_detail.csv":      ["customer_segment", "order_detail"],
    "competitor_pricing":    ["competitor_monitor", "competitor_monitor.csv"],
    "competitor_monitor":    ["competitor_pricing"],
    "competitor_monitor.csv":["competitor_pricing", "competitor_monitor"],
    "inventory_logistics":   ["inventory_status", "inventory_status.csv"],
    "inventory_status":      ["inventory_logistics"],
    "inventory_status.csv":  ["inventory_logistics", "inventory_status"],
    "product_sku_detail":    ["product_performance", "product_performance.csv"],
    "product_performance":   ["product_sku_detail"],
    "product_performance.csv":["product_sku_detail", "product_performance"],
    "promotion_calendar":    ["promotion_calendar.csv"],
    "promotion_calendar.csv":["promotion_calendar"],
    "refund_after_sale":     ["refund_after_sale.csv"],
    "refund_after_sale.csv": ["refund_after_sale"],
    "market_benchmark":      ["competitor_monitor", "competitor_monitor.csv"],
    "traffic_source_detail": [],  # 已内含在 traffic_summary 中
}


# ============================================================================
# 2. 工具函数
# ============================================================================

def fuzzy_match_dimension(name: str, keyword: str) -> bool:
    name_lower = name.lower()
    keywords = DIMENSION_KEYWORD_MAP.get(keyword, [keyword])
    return any(kw in name_lower for kw in keywords)


def fuzzy_match_suggested_data(actual: str, expected: str) -> bool:
    """v5.1: suggested_data 模糊匹配，支持新旧别名"""
    a = actual.lower().replace(".csv", "").strip()
    e = expected.lower().replace(".csv", "").strip()
    if a == e:
        return True
    # 检查别名
    aliases_for_expected = SUGGESTED_DATA_ALIASES.get(expected, [])
    aliases_for_expected_stripped = [x.lower().replace(".csv", "").strip() for x in aliases_for_expected]
    if a in aliases_for_expected_stripped:
        return True
    aliases_for_actual = SUGGESTED_DATA_ALIASES.get(actual, [])
    aliases_for_actual_stripped = [x.lower().replace(".csv", "").strip() for x in aliases_for_actual]
    if e in aliases_for_actual_stripped:
        return True
    return False


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


# ============================================================================
# 3. V8 Trace 分析器
# ============================================================================

class V8TraceAnalyzer:
    """从 v8 agent 输出中提取三步走的关键信息, 向下兼容 v7/v5"""

    def __init__(self, agent_result: Dict):
        self.result = agent_result or {}
        self.trace = self.result.get("trace_log") or {}
        self.events = self.trace.get("events", [])
        self._arch = self._detect_arch()

    def _detect_arch(self) -> str:
        if self.result.get("scan_state") is not None:
            return "v8"
        if self.result.get("reason_result") is not None:
            return "v8"
        if "needs_deep_rca" in self.result:
            return "v8"
        if self.result.get("scan_data") is not None:
            return "v7"
        if self.result.get("confirmed_anomalies") is not None:
            return "v7"
        return "v5"

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def route(self) -> str:
        for ev in self.events:
            if ev.get("event_type") == "intent_classification":
                return (ev.get("output_data") or {}).get("route", "unknown")
        return self.result.get("route", "unknown")

    @property
    def analysis_depth(self) -> str:
        """v6: v9 架构新增字段, descriptive / diagnostic / causal"""
        depth = self.result.get("analysis_depth")
        if depth:
            return depth
        # 尝试从 trace events 中解析 Commander 输出
        for ev in self.events:
            if ev.get("event_type") in ("commander_plan", "commander_output"):
                out = ev.get("output_data") or {}
                if out.get("analysis_depth"):
                    return out["analysis_depth"]
        return "unknown"

    @property
    def total_time_ms(self) -> int:
        return self.result.get("total_time_ms", 0)

    # ── Step 1: Scan ──

    def get_scanned_anomalies(self) -> List[Dict]:
        if self._arch == "v8":
            scan_state = self.result.get("scan_state") or {}
            # ✅ P0 FIX: agent ScanState 实际写的字段是 "all_anomalies"，
            # 之前读 "anomalies" 一直拿到空列表，导致 actual_anomaly_dims 永远为空，
            # A 组扫描得分被结构性卡在 0.6（recall=0 的天花板）。
            # 同时保留对旧字段名的兼容，便于过渡期。
            return (scan_state.get("all_anomalies")
                    or scan_state.get("anomalies", []))
        elif self._arch == "v7":
            return self.result.get("confirmed_anomalies", [])
        else:
            report = self.result.get("final_report") or {}
            return [{"dimension": c} for c in report.get("confirmed_hypotheses", [])]

    def get_scanned_dimensions(self) -> Set[str]:
        if self._arch == "v8":
            scan_state = self.result.get("scan_state") or {}
            dims = set()
            # ✅ P0 FIX: 同 get_scanned_anomalies，读 "all_anomalies"/"all_normal"
            # ✅ Task 2A FIX: 优先取 canonical_dim 保持与 get_anomaly_dimensions 一致
            for a in (scan_state.get("all_anomalies") or scan_state.get("anomalies", [])):
                dims.add(a.get("canonical_dim") or a.get("dimension", ""))
            for n in (scan_state.get("all_normal") or scan_state.get("normal_dimensions", [])):
                if isinstance(n, str):
                    dims.add(n)
                else:
                    dims.add(n.get("canonical_dim") or n.get("dimension", ""))
            return dims - {""}
        elif self._arch == "v7":
            dims = set()
            for item in self.result.get("scan_data", []):
                if isinstance(item, dict):
                    dims.add(item.get("dimension", ""))
            return dims - {""}
        else:
            dims = set()
            for ev in self.events:
                if ev.get("event_type") == "hypothesis_selected":
                    dim = (ev.get("input_data") or {}).get("dimension", "")
                    if dim:
                        dims.add(dim)
            return dims

    def get_anomaly_dimensions(self) -> Set[str]:
        # ✅ Task 2A FIX: 优先取 canonical_dim（agent 现在双输出英文 canonical + 中文 dimension），
        # 评测集 expected_anomaly_dims 用英文 key（traffic/conversion_rate/...），
        # 直接精确匹配比依赖 fuzzy_match_dimension 关键词表更确定、更快。
        # 没有 canonical_dim 的项（L2/L3 切片结果或未登记中文名）回退到原 dimension 字段。
        out = set()
        for a in self.get_scanned_anomalies():
            key = a.get("canonical_dim") or a.get("dimension", "")
            if key:
                out.add(key)
        return out

    def get_normal_dimensions(self) -> Set[str]:
        if self._arch == "v8":
            scan_state = self.result.get("scan_state") or {}
            # ✅ P0 FIX: 读 "all_normal"
            normals = (scan_state.get("all_normal")
                       or scan_state.get("normal_dimensions", []))
            dims = set()
            for n in normals:
                if isinstance(n, str):
                    dims.add(n)
                elif isinstance(n, dict):
                    # ✅ Task 2A FIX: normal 项也优先取 canonical_dim
                    key = n.get("canonical_dim") or n.get("dimension", "")
                    if key:
                        dims.add(key)
            return dims - {""}
        return self.get_scanned_dimensions() - self.get_anomaly_dimensions()

    # ── Step 2: Reason ──

    def get_reason_result(self) -> Dict:
        if self._arch == "v8":
            return self.result.get("reason_result") or {}
        elif self._arch == "v7":
            return self.result.get("causal_result") or {}
        return {}

    def get_root_causes(self) -> List[Dict]:
        return self.get_reason_result().get("root_causes", [])

    def get_root_cause_dimensions(self) -> Set[str]:
        # ✅ Task 2A FIX: 同 get_anomaly_dimensions，优先取 canonical_dim
        out = set()
        for rc in self.get_root_causes():
            key = rc.get("canonical_dim") or rc.get("dimension", "")
            if key:
                out.add(key)
        return out

    def get_needs_deep_rca(self) -> bool:
        if self._arch == "v8":
            return self.result.get("needs_deep_rca", False)
        return self.get_reason_result().get("needs_deep_rca", False)

    def get_confidence(self) -> str:
        return self.get_reason_result().get("confidence", "unknown")

    def get_suggested_data_types(self) -> List[str]:
        # ✅ v7.3 FIX: 完整 fallback 链，兼容多种 app 输出格式
        # 优先级:
        #   1) reason_result.suggested_data_type_ids (app_v9.6+ 直出 ID 列表)
        #   2) 顶层 suggested_data_types - dict 列表 → 取 .type / 关键词映射
        #                                  - string 列表 → 直接使用
        #   3) reason_result.suggested_data (结构化 dict 列表) → 关键词映射
        #   4) trace events 中 deep_rca_decision 兜底
        if self._arch != "v8":
            return self.get_reason_result().get("suggested_data_types", [])

        # 1) 直出 ID 列表
        reason = self.get_reason_result()
        ids = reason.get("suggested_data_type_ids")
        if ids and isinstance(ids, list):
            return [s for s in ids if isinstance(s, str)]

        # 2) 顶层 suggested_data_types
        top = self.result.get("suggested_data_types", [])
        if top and isinstance(top, list):
            if any(isinstance(x, dict) for x in top):
                # 优先读 dict 里显式的 .type 字段
                from_type = [
                    x.get("type") for x in top
                    if isinstance(x, dict) and x.get("type")
                    and x.get("type") != "unknown"
                ]
                if from_type:
                    return from_type
                return self._extract_types_from_suggested_data(top)
            if all(isinstance(x, str) for x in top):
                return top

        # 3) reason_result.suggested_data
        sd = reason.get("suggested_data") or []
        if sd and isinstance(sd, list) and any(isinstance(x, dict) for x in sd):
            from_type = [
                x.get("type") for x in sd
                if isinstance(x, dict) and x.get("type")
                and x.get("type") != "unknown"
            ]
            if from_type:
                return from_type
            return self._extract_types_from_suggested_data(sd)

        # 4) trace events 兜底
        for ev in self.events:
            if ev.get("event_type") == "deep_rca_decision":
                sd2 = (ev.get("output_data") or {}).get("suggested_data", [])
                if sd2:
                    return self._extract_types_from_suggested_data(sd2)
        return []

    @staticmethod
    def _extract_types_from_suggested_data(suggested_data: List[Dict]) -> List[str]:
        """从结构化 suggested_data 对象中提取对应的 data type 标签。

        suggested_data 格式: [{"description": "...", "reason": "...", "required_columns": [...]}, ...]
        通过关键词匹配映射到 SUGGESTED_DATA_TYPES 中的标准标签。
        """
        # 关键词 → 标准标签 映射
        keyword_to_type = {
            "广告": "ad_campaign", "投放": "ad_campaign", "营销": "ad_campaign",
            "推广": "ad_campaign", "ad": "ad_campaign", "campaign": "ad_campaign",
            "marketing": "ad_campaign", "流量渠道": "ad_campaign",
            "渠道明细": "ad_campaign", "渠道流量": "ad_campaign",
            "流量来源": "ad_campaign", "spend": "ad_campaign",
            "impression": "ad_campaign", "click": "ad_campaign",
            "roi": "ad_campaign", "roas": "ad_campaign",
            "cpc": "ad_campaign", "cpa": "ad_campaign", "cpm": "ad_campaign",
            "订单": "order_detail", "客户": "order_detail", "用户分群": "order_detail",
            "customer": "order_detail", "order": "order_detail",
            "漏斗": "order_detail", "转化漏斗": "order_detail",
            "用户行为": "order_detail",
            "竞品": "competitor_monitor", "竞争": "competitor_monitor",
            "competitor": "competitor_monitor", "市场基准": "competitor_monitor",
            "库存": "inventory_status", "缺货": "inventory_status",
            "inventory": "inventory_status", "stock": "inventory_status",
            "促销": "promotion_calendar", "活动日历": "promotion_calendar",
            "优惠": "promotion_calendar", "promotion": "promotion_calendar",
            "营销活动": "promotion_calendar",
            "商品": "product_performance", "sku": "product_performance",
            "product": "product_performance", "品类表现": "product_performance",
            "商品详情": "product_performance",
            "退款": "refund_after_sale", "售后": "refund_after_sale",
            "refund": "refund_after_sale",
        }
        extracted = []
        seen = set()
        for item in suggested_data:
            if not isinstance(item, dict):
                continue
            # ✅ v7.3: 兼容 reason/reasoning 双字段，并把 required_columns 也纳入匹配文本
            text = (
                item.get("description", "") + " "
                + (item.get("reason") or item.get("reasoning") or "") + " "
                + " ".join(item.get("required_columns", []) or [])
            ).lower()
            for kw, dtype in keyword_to_type.items():
                if kw in text and dtype not in seen:
                    extracted.append(dtype)
                    seen.add(dtype)
                    break
        return extracted

    def get_causal_chain(self) -> str:
        return self.get_reason_result().get("causal_chain", "")

    # ── Step 3: Deep RCA ──

    def get_deep_rca_result(self) -> Dict:
        # ✅ v7.3: 多源 fallback，兼容 app 未直出 deep_rca_result 的旧版本
        direct = self.result.get("deep_rca_result")
        if direct:
            return direct
        # Fallback 1: final_report 是 deep RCA 报告
        rpt = self.result.get("final_report") or self.result.get("report") or {}
        if isinstance(rpt, dict) and (
            rpt.get("is_deep_rca_report") or rpt.get("deep_rca_steps")
        ):
            return {
                "deep_root_causes": [
                    {"dimension": d if isinstance(d, str) else d.get("dimension", ""),
                     "name": d if isinstance(d, str) else d.get("name", ""),
                     "source": "fallback_from_report"}
                    for d in (rpt.get("confirmed_hypotheses", []) or [])
                ],
                "action_recommendations": [],
                "summary": rpt.get("summary", ""),
                "full_content": rpt.get("full_content", ""),
                "causal_chain": rpt.get("causal_chain", ""),
            }
        # Fallback 2: trace events 中的 deep_rca / reporter_deep
        for ev in self.events:
            etype = (ev.get("event_type") or "").lower()
            if "deep_rca_report" in etype or "reporter_deep" in etype:
                od = ev.get("output_data") or {}
                if od.get("confirmed_count") or od.get("summary"):
                    return {
                        "deep_root_causes": [],
                        "action_recommendations": [],
                        "summary": od.get("summary", ""),
                        "full_content": od.get("summary", ""),
                    }
        return {}

    def get_deep_root_causes(self) -> List[Dict]:
        return self.get_deep_rca_result().get("deep_root_causes", [])

    def get_action_recommendations(self) -> List[str]:
        return self.get_deep_rca_result().get("action_recommendations", [])

    # ── Report ──

    def get_report_text(self) -> str:
        report = self.result.get("report") or self.result.get("final_report") or {}
        return report.get("full_content", "") or report.get("content", "")

    def get_chart_paths(self) -> List[str]:
        return self.result.get("chart_paths", [])

    def get_steps(self) -> List[Dict]:
        return self.result.get("steps", [])


# ============================================================================
# 4. 各维度评分器
# ============================================================================

class AnalysisDepthValidator:
    """维度1: 分析深度正确性 (5%) — v6 替换原 RouteValidator

    v9 架构下 Router 仅做 VALID/INVALID 分类, 所有合法查询 route 都是 "complex".
    真正有评测价值的是 Commander 判定的 analysis_depth.

    软匹配规则:
      - 完全匹配 → 1.0
      - causal ↔ diagnostic 互相容差 → 0.7 (两者都需要扫描+推理,边界模糊)
      - descriptive ↔ diagnostic 容差 → 0.5 (都不需要完整因果链)
      - descriptive ↔ causal → 0.3 (深度差异大)
      - unknown / invalid → 0.0
    """

    DEPTH_COMPAT = {
        ("causal", "causal"):           1.0,
        ("diagnostic", "diagnostic"):   1.0,
        ("descriptive", "descriptive"): 1.0,
        ("causal", "diagnostic"):       0.7,
        ("diagnostic", "causal"):       0.7,
        ("diagnostic", "descriptive"):  0.5,
        ("descriptive", "diagnostic"):  0.5,
        ("causal", "descriptive"):      0.3,
        ("descriptive", "causal"):      0.3,
    }

    @staticmethod
    def score(question: Dict, trace: V8TraceAnalyzer) -> Dict:
        expected = question.get("expected_analysis_depth", "causal")
        actual = trace.analysis_depth

        # 先做 route 合法性前置检查
        route = trace.route
        if route in ("invalid", "error", "unknown"):
            return {
                "score": 0.0,
                "expected": expected,
                "actual": actual,
                "route": route,
                "detail": f"路由非法 (route={route}), 无法评测深度",
            }

        sc = AnalysisDepthValidator.DEPTH_COMPAT.get((expected, actual), 0.0)
        return {
            "score": round(sc, 4),
            "expected": expected,
            "actual": actual,
            "route": route,
            "detail": (f"深度匹配 expected={expected} actual={actual} → {sc:.0%}"
                       if sc > 0 else
                       f"深度不匹配 expected={expected} actual={actual}"),
        }


# 保留别名供旧代码路径引用 (向后兼容)
RouteValidator = AnalysisDepthValidator


class ScanQualityValidator:
    """维度2: 异常扫描质量 (15%) — Step 1 评测"""

    @staticmethod
    def score(question: Dict, trace: V8TraceAnalyzer) -> Dict:
        expected_step1 = question.get("expected_step1", {})
        expected_anomalies = expected_step1.get("expected_anomalies", [])
        expected_normal = expected_step1.get("expected_normal", [])

        expected_anomaly_dims = {a["dimension"] for a in expected_anomalies}
        expected_normal_dims = set(expected_normal)

        actual_anomaly_dims = trace.get_anomaly_dimensions()
        actual_normal_dims = trace.get_normal_dimensions()

        # Anomaly Recall (含模糊匹配)
        recall_hits = 0
        for ed in expected_anomaly_dims:
            if ed in actual_anomaly_dims:
                recall_hits += 1
            else:
                for ad in actual_anomaly_dims:
                    if fuzzy_match_dimension(ad, ed):
                        recall_hits += 1
                        break
        recall = safe_divide(recall_hits, len(expected_anomaly_dims), 1.0)

        # Anomaly Precision
        precision = safe_divide(recall_hits, len(actual_anomaly_dims), 1.0)

        # Normal Specificity
        is_no_anomaly_case = len(expected_anomaly_dims) == 0
        if is_no_anomaly_case:
            specificity = 1.0 if len(actual_anomaly_dims) == 0 else 0.0
        else:
            false_positives_in_normal = len(expected_normal_dims & actual_anomaly_dims)
            specificity = max(0.0, 1.0 - false_positives_in_normal * 0.25)

        # 变化率范围验证
        range_score = 1.0
        for ea in expected_anomalies:
            pct_range = ea.get("change_pct_range")
            if pct_range:
                dim = ea["dimension"]
                # ✅ Task 2A FIX: 在原 fuzzy 匹配前先做 canonical 精确匹配
                actual_entry = next(
                    (a for a in trace.get_scanned_anomalies()
                     if a.get("canonical_dim") == dim
                     or a.get("dimension") == dim
                     or fuzzy_match_dimension(a.get("dimension", ""), dim)),
                    None)
                if actual_entry and actual_entry.get("change_pct") is not None:
                    if not (pct_range[0] <= actual_entry["change_pct"] <= pct_range[1]):
                        range_score -= 0.2
        range_score = max(0.0, range_score)

        final = recall * 0.40 + precision * 0.30 + specificity * 0.20 + range_score * 0.10

        return {
            "score": round(max(0.0, min(1.0, final)), 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "specificity": round(specificity, 4),
            "range_score": round(range_score, 4),
            "expected_anomaly_dims": sorted(expected_anomaly_dims),
            "actual_anomaly_dims": sorted(actual_anomaly_dims),
            "is_no_anomaly_case": is_no_anomaly_case,
            "detail": (f"R={recall:.0%} P={precision:.0%} "
                       f"S={specificity:.0%} Rng={range_score:.0%}"),
        }


class ExecutionValidator:
    """维度3: 执行效率 (10%)"""

    @staticmethod
    def score(question: Dict, trace: V8TraceAnalyzer) -> Dict:
        steps = trace.get_steps()
        actual_time = trace.total_time_ms / 1000.0
        time_limit = question.get("evaluation_criteria", {}).get(
            "response_time_limit_sec", 120)

        if actual_time <= time_limit * 0.5:
            time_score = 1.0
        elif actual_time <= time_limit:
            time_score = max(0.3, 1.0 - (actual_time - time_limit * 0.5) / (time_limit * 0.5) * 0.7)
        else:
            time_score = max(0.0, 0.3 - (actual_time / time_limit - 1.0) * 0.3)

        total_steps = len(steps)
        success_count = sum(1 for s in steps
                            if s.get("result", {}).get("success", s.get("success", False)))
        exec_rate = safe_divide(success_count, total_steps, 1.0)

        max_steps = question.get("evaluation_criteria", {}).get("max_steps", 6)
        step_score = 1.0 if total_steps <= max_steps else max(
            0.3, 1.0 - (total_steps - max_steps) * 0.15)

        final = time_score * 0.40 + exec_rate * 0.35 + step_score * 0.25

        return {
            "score": round(max(0.0, min(1.0, final)), 4),
            "time_sec": round(actual_time, 2),
            "time_limit_sec": time_limit,
            "time_score": round(time_score, 4),
            "total_steps": total_steps,
            "success_steps": success_count,
            "exec_rate": round(exec_rate, 4),
            "step_score": round(step_score, 4),
            "detail": f"时间 {actual_time:.0f}s/{time_limit}s, 步骤 {success_count}/{total_steps}",
        }


class RootCauseValidator:
    """维度4: 根因正确性 (20%) — Step 2 评测"""

    @staticmethod
    def score(question: Dict, trace: V8TraceAnalyzer) -> Dict:
        expected_step2 = question.get("expected_step2", {})
        expected_root_causes = expected_step2.get("expected_root_causes", [])

        if not expected_root_causes:
            actual_rc = trace.get_root_causes()
            if len(actual_rc) == 0:
                return {"score": 1.0, "detail": "正确: 无根因 (无异常)", "hits": 0,
                        "expected_count": 0, "false_positives": 0}
            else:
                return {"score": 0.3, "detail": f"误报: 无异常但产出 {len(actual_rc)} 条根因",
                        "hits": 0, "expected_count": 0, "false_positives": len(actual_rc)}

        actual_rc_dims = trace.get_root_cause_dimensions()
        expected_rc_dims = set()
        for rc in expected_root_causes:
            if isinstance(rc, str):
                expected_rc_dims.add(rc)
            elif isinstance(rc, dict):
                expected_rc_dims.add(rc.get("dimension", rc.get("name", "")))

        hits = 0
        for ed in expected_rc_dims:
            if ed in actual_rc_dims:
                hits += 1
            elif any(fuzzy_match_dimension(ad, ed) or fuzzy_match_dimension(ed, ad)
                     for ad in actual_rc_dims):
                hits += 1

        accuracy = safe_divide(hits, len(expected_rc_dims))

        false_positives = 0
        for ad in actual_rc_dims:
            if ad not in expected_rc_dims and not any(
                    fuzzy_match_dimension(ad, ed) for ed in expected_rc_dims):
                false_positives += 1
        fp_penalty = min(false_positives * 0.15, 0.45)

        final = max(0.0, accuracy - fp_penalty)

        return {
            "score": round(min(1.0, final), 4),
            "accuracy": round(accuracy, 4),
            "hits": hits,
            "expected_count": len(expected_rc_dims),
            "false_positives": false_positives,
            "expected_rc_dims": sorted(expected_rc_dims),
            "actual_rc_dims": sorted(actual_rc_dims),
            "detail": f"根因 {hits}/{len(expected_rc_dims)}, 误报 {false_positives}",
        }


class DeepRCATriggerValidator:
    """维度5: 深度分析触发准确率 (15%)
    v5.1: suggested_data 使用 fuzzy_match_suggested_data 支持新旧别名匹配
    """

    @staticmethod
    def score(question: Dict, trace: V8TraceAnalyzer) -> Dict:
        expected_step2 = question.get("expected_step2", {})
        expected_needs_deep = expected_step2.get("needs_deep_rca", False)
        expected_suggested = set(expected_step2.get("expected_suggested_data_types", []))

        actual_needs_deep = trace.get_needs_deep_rca()
        # v7 FIX: suggested_data 可能返回 dict 对象而非字符串，过滤非字符串元素
        raw_suggested = trace.get_suggested_data_types()
        actual_suggested = set(
            s if isinstance(s, str) else s.get("type", s.get("description", str(s)))
            for s in raw_suggested
            if s is not None
        )

        trigger_correct = (actual_needs_deep == expected_needs_deep)
        trigger_score = 1.0 if trigger_correct else 0.0

        if expected_needs_deep and expected_suggested:
            # v5.1: 使用模糊匹配计算 recall/precision
            hits = 0
            matched_actual = set()
            for exp_s in expected_suggested:
                for act_s in actual_suggested:
                    if fuzzy_match_suggested_data(act_s, exp_s):
                        hits += 1
                        matched_actual.add(act_s)
                        break
            recall = safe_divide(hits, len(expected_suggested))
            precision = safe_divide(hits, len(actual_suggested), 1.0)
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
            suggested_score = f1
        elif not expected_needs_deep:
            suggested_score = 1.0 if len(actual_suggested) == 0 else max(
                0.0, 1.0 - len(actual_suggested) * 0.2)
            recall, precision = 0.0, 0.0
        else:
            suggested_score = 1.0 if actual_needs_deep else 0.0
            recall, precision = 0.0, 0.0

        final = trigger_score * 0.50 + suggested_score * 0.50

        return {
            "score": round(max(0.0, min(1.0, final)), 4),
            "trigger_correct": trigger_correct,
            "expected_needs_deep": expected_needs_deep,
            "actual_needs_deep": actual_needs_deep,
            "suggested_score": round(suggested_score, 4),
            "expected_suggested": sorted(expected_suggested),
            "actual_suggested": sorted(actual_suggested),
            "detail": (f"Trigger={'✅' if trigger_correct else '❌'} "
                       f"Suggested={suggested_score:.0%}"),
        }


class ReportValidator:
    """维度6: 报告完整性 (10%)"""

    @staticmethod
    def score(question: Dict, trace: V8TraceAnalyzer) -> Dict:
        report_text = trace.get_report_text()
        expected_step2 = question.get("expected_step2", {})

        if not report_text:
            return {"score": 0.0, "detail": "报告未生成", "report_exists": False}

        length_score = min(1.0, max(0.2, len(report_text) / 100))

        must_keywords = question.get("expected_report", {}).get("must_mention_keywords", [])
        kw_score = (safe_divide(
            sum(1 for kw in must_keywords if kw in report_text),
            len(must_keywords)) if must_keywords else 1.0)

        needs_deep = expected_step2.get("needs_deep_rca", False)
        if needs_deep:
            limit_kws = ["数据局限", "数据有限", "无法确认", "需要补充", "建议提供",
                         "进一步分析", "深入分析", "未验证", "有限的数据"]
            limitation_score = 1.0 if any(kw in report_text for kw in limit_kws) else 0.3
        else:
            limitation_score = 1.0

        final = length_score * 0.20 + kw_score * 0.40 + limitation_score * 0.40

        return {
            "score": round(max(0.0, min(1.0, final)), 4),
            "report_exists": True,
            "report_length": len(report_text),
            "kw_score": round(kw_score, 4),
            "limitation_score": round(limitation_score, 4),
            "detail": f"长度 {len(report_text)}, 关键词 {kw_score:.0%}, 局限标注 {limitation_score:.0%}",
        }


class ReasoningValidator:
    """维度7: 因果推理质量 (15%)"""

    @staticmethod
    def score(question: Dict, trace: V8TraceAnalyzer) -> Dict:
        expected_step2 = question.get("expected_step2", {})
        expected_root_causes = expected_step2.get("expected_root_causes", [])
        category = question.get("category", "")
        combined = f"{trace.get_report_text()} {trace.get_causal_chain()}"

        sub_scores = []

        # 因果链存在性
        causal_kws = ["导致", "引起", "因为", "根本原因", "连锁", "→", "驱动", "影响", "造成"]
        if expected_root_causes:
            sub_scores.append(("causal_chain", 1.0 if any(k in combined for k in causal_kws) else 0.2))
        else:
            sub_scores.append(("causal_chain", 1.0))

        # 方向正确性
        dir_map = {
            "down": ["下降", "降低", "减少", "下滑", "drop", "decline"],
            "up": ["上涨", "上升", "增长", "升高", "increase", "rise"],
        }
        dir_total, dir_correct = 0, 0
        for ea in question.get("expected_step1", {}).get("expected_anomalies", []):
            d = ea.get("direction", "")
            if d:
                dir_total += 1
                if any(kw in combined for kw in dir_map.get(d, [])):
                    dir_correct += 1
        sub_scores.append(("direction", safe_divide(dir_correct, dir_total, 1.0)))

        # 无异常 case 抗幻觉
        if category in ("no_anomaly", "threshold_boundary"):
            no_kws = ["未发现", "无显著", "正常范围", "正常波动", "无异常", "阈值内"]
            sub_scores.append(("anti_hallucination", 1.0 if any(k in combined for k in no_kws) else 0.0))
            if len(trace.get_root_causes()) > 0:
                sub_scores.append(("false_rc_penalty", 0.0))

        # 置信度
        exp_conf = expected_step2.get("confidence", "")
        act_conf = trace.get_confidence()
        if exp_conf and act_conf and act_conf != "unknown":
            level = {"high": 3, "medium": 2, "low": 1}
            diff = abs(level.get(exp_conf, 2) - level.get(act_conf, 2))
            sub_scores.append(("confidence", max(0.0, 1.0 - diff * 0.35)))

        if not sub_scores:
            return {"score": 0.5, "detail": "无推理评估项", "sub_scores": {}}

        avg = sum(s for _, s in sub_scores) / len(sub_scores)
        return {
            "score": round(max(0.0, min(1.0, avg)), 4),
            "sub_scores": {n: round(s, 4) for n, s in sub_scores},
            "detail": ", ".join(f"{n}={s:.2f}" for n, s in sub_scores),
        }


class DeepRCAValidator:
    """维度8: 深度分析质量 (10%) — 仅 B 组有效
    v7: 基于 expected_step3 的 expected_deep_findings + must_mention_keywords 评测
    """

    @staticmethod
    def score(question: Dict, trace: V8TraceAnalyzer,
              step3_ran: bool = False) -> Dict:
        needs_deep = question.get("expected_step2", {}).get("needs_deep_rca", False)

        if not needs_deep:
            return {"score": 1.0, "applicable": False, "detail": "A 组, 默认满分"}

        if not step3_ran:
            return {"score": 0.5, "applicable": True, "step3_ran": False,
                    "detail": "B 组未执行 Step 3 (--run-step3)"}

        deep_rca = trace.get_deep_rca_result()
        if not deep_rca:
            return {"score": 0.2, "applicable": True, "step3_ran": True,
                    "detail": "Step 3 无结果返回"}

        expected_step3 = question.get("expected_step3", {})

        # ── 子项 1: 报告关键词命中 (30%) ──
        must_kws = expected_step3.get("must_mention_keywords", [])
        report_text = ""
        # 从多处提取报告文本
        report_obj = trace.result.get("report") or trace.result.get("final_report") or {}
        if isinstance(report_obj, dict):
            report_text = report_obj.get("full_content", "") or report_obj.get("summary", "")
        elif isinstance(report_obj, str):
            report_text = report_obj
        # 也检查 deep_rca_result 中的文本
        deep_text = json.dumps(deep_rca, ensure_ascii=False) if deep_rca else ""
        combined_text = (report_text + " " + deep_text).lower()

        kw_hits = sum(1 for kw in must_kws if kw.lower() in combined_text) if must_kws else 0
        kw_score = kw_hits / len(must_kws) if must_kws else 1.0

        # ── 子项 2: 根因发现命中 (40%) ──
        expected_findings = expected_step3.get("expected_deep_findings", [])
        actual_deep_rcs = trace.get_deep_root_causes()

        # 也从 root_cause_summary 中匹配
        expected_summary = expected_step3.get("expected_root_cause_summary", "")

        finding_hits = 0
        finding_details = []
        for ef in expected_findings:
            finding_name = ef.get("finding", "")
            finding_desc = ef.get("description", "")
            # 检查 finding 是否在实际结果中被提到
            hit = False

            # 方法 1: 在 deep_root_causes 的 dimension/description 中搜索
            for drc in actual_deep_rcs:
                drc_text = (drc.get("description", "") + " " +
                            drc.get("dimension", "") + " " +
                            drc.get("reasoning", "")).lower()
                # 用 finding 描述中的关键词匹配
                desc_words = [w for w in finding_desc if len(w) > 1]
                if any(w in drc_text for w in desc_words):
                    hit = True
                    break

            # 方法 2: 在报告全文中搜索
            if not hit and finding_desc:
                # 抽取描述中的核心名词（2字以上）
                check_words = [w for w in finding_desc
                               if len(w) >= 2 and w not in ("的", "了", "为", "从", "到")]
                # 至少一半关键词命中
                word_hits = sum(1 for w in check_words if w in combined_text)
                if check_words and word_hits >= len(check_words) * 0.4:
                    hit = True

            finding_hits += int(hit)
            finding_details.append({"finding": finding_name, "hit": hit})

        finding_score = finding_hits / len(expected_findings) if expected_findings else 1.0

        # ── 子项 3: 补充数据利用 (20%) ──
        # v5.1 扩展补充数据利用检测关键词
        data_util = False
        for s in trace.get_steps():
            code = s.get("code", "") or ""
            supp_kws = [
                "supplementary", "补充", "traffic_customer",
                "competitor_pricing", "inventory_logistics",
                "marketing_detail", "sku_promo",
                "order_detail", "ad_campaign", "product_performance",
                "competitor_monitor", "refund_after_sale",
                "inventory_status", "promotion_calendar",
                "订单明细", "广告投放", "商品明细", "竞品监控",
                "退款售后", "库存报表", "促销日历",
                # v7 新增: 补充数据特征列名
                "是否新客", "竞品售价", "竞品促销", "平均发货时长",
                "是否缺货", "上架状态", "review_score", "ROI",
                "退款原因", "活动名称", "优惠描述",
            ]
            if any(kw in code for kw in supp_kws):
                data_util = True
                break

        util_score = 1.0 if data_util else 0.2

        # ── 子项 4: 行动建议 (10%) ──
        has_actions = len(trace.get_action_recommendations()) >= 1
        act_score = 1.0 if has_actions else 0.3

        final = kw_score * 0.30 + finding_score * 0.40 + util_score * 0.20 + act_score * 0.10

        return {
            "score": round(max(0.0, min(1.0, final)), 4),
            "applicable": True,
            "step3_ran": True,
            "keyword_score": round(kw_score, 4),
            "keyword_hits": f"{kw_hits}/{len(must_kws)}",
            "finding_score": round(finding_score, 4),
            "finding_hits": f"{finding_hits}/{len(expected_findings)}",
            "finding_details": finding_details,
            "data_utilization": data_util,
            "has_actions": has_actions,
            "detail": (f"关键词={kw_hits}/{len(must_kws)} "
                       f"发现={finding_hits}/{len(expected_findings)} "
                       f"数据={'✅' if data_util else '❌'} "
                       f"行动={'✅' if has_actions else '❌'}"),
        }


# ============================================================================
# 5. 综合评分器
# ============================================================================

class V8ComplexEvaluator:
    """v8 三步走综合评分器 — 8 维度"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or SCORING_WEIGHTS

    def evaluate(self, question: Dict, agent_result: Dict,
                 step3_ran: bool = False) -> Dict:
        trace = V8TraceAnalyzer(agent_result)

        scores = {
            "analysis_depth":   AnalysisDepthValidator.score(question, trace),
            "scan_quality":     ScanQualityValidator.score(question, trace),
            "execution":        ExecutionValidator.score(question, trace),
            "root_cause":       RootCauseValidator.score(question, trace),
            "deep_rca_trigger": DeepRCATriggerValidator.score(question, trace),
            "report":           ReportValidator.score(question, trace),
            "reasoning":        ReasoningValidator.score(question, trace),
            "deep_rca":         DeepRCAValidator.score(question, trace, step3_ran),
        }

        weighted_total = sum(
            scores[dim]["score"] * self.weights.get(dim, 0)
            for dim in scores
        )

        return {
            "question_id": question["id"],
            "question": question["question"],
            "category": question.get("category", ""),
            "group": "B" if question.get("expected_step2", {}).get(
                "needs_deep_rca", False) else "A",
            "weighted_score": round(weighted_total, 4),
            "dimension_scores": {dim: scores[dim]["score"] for dim in scores},
            "dimension_details": scores,
            "total_time_ms": agent_result.get("total_time_ms", 0),
            "steps_count": len(agent_result.get("steps", [])),
            "arch_detected": trace.arch,
            "error": agent_result.get("error"),
        }


# ============================================================================
# 5.1  Layer C — 误差归因分析 (v5.1 新增)
# ============================================================================

class ErrorAttributionAnalyzer:
    """
    Layer C: 误差归因分析。

    对每道失分题进行归因，判断失分的根本来源是 Step 1 (扫描) 还是 Step 2 (推理)。
    帮助定位优化方向: 如果大部分错误来自 scan_cascading，应优先优化 Step 1；
    如果大部分来自 reasoning_error，应优先优化 Step 2。

    归因标签定义:
      - scan_cascading:  Step 1 扫描失准 (scan_quality < 阈值) 导致下游 Step 2 连锁失分
      - reasoning_error: Step 1 扫描正常 (scan_quality ≥ 阈值) 但 Step 2 推理/触发失分
      - trigger_error:   Step 1+2 的根因判定都 OK, 但 needs_deep_rca / suggested_data 触发错误
      - compound_error:  Step 1 和 Step 2 同时有问题, 无法归因到单一环节
      - pass:            整体得分 ≥ 通过线, 不需要归因
    """

    # 可配置阈值
    SCAN_GOOD_THRESHOLD = 0.70    # scan_quality ≥ 此值视为"扫描基本正确"
    ROOT_CAUSE_FAIL_THRESHOLD = 0.60   # root_cause < 此值视为"根因判定失败"
    TRIGGER_FAIL_THRESHOLD = 0.60      # deep_rca_trigger < 此值视为"触发判定失败"
    REASONING_FAIL_THRESHOLD = 0.60    # reasoning < 此值视为"推理失败"
    PASS_THRESHOLD = 0.60              # weighted_score ≥ 此值视为通过

    def __init__(self, scan_good_threshold: float = None,
                 pass_threshold: float = None):
        if scan_good_threshold is not None:
            self.SCAN_GOOD_THRESHOLD = scan_good_threshold
        if pass_threshold is not None:
            self.PASS_THRESHOLD = pass_threshold

    def attribute_single(self, eval_result: Dict) -> Dict:
        """
        对单题进行归因。

        输入: V8ComplexEvaluator.evaluate() 的返回值
        输出: {
            "label": "scan_cascading" | "reasoning_error" | "trigger_error" | "compound_error" | "pass",
            "scan_quality": float,
            "root_cause": float,
            "deep_rca_trigger": float,
            "reasoning": float,
            "weighted_score": float,
            "explanation": str,  # 人类可读的归因解释
        }
        """
        ws = eval_result.get("weighted_score", 0)
        ds = eval_result.get("dimension_scores", {})

        scan_s = ds.get("scan_quality", 0)
        rc_s = ds.get("root_cause", 0)
        trigger_s = ds.get("deep_rca_trigger", 0)
        reasoning_s = ds.get("reasoning", 0)

        base = {
            "scan_quality": round(scan_s, 4),
            "root_cause": round(rc_s, 4),
            "deep_rca_trigger": round(trigger_s, 4),
            "reasoning": round(reasoning_s, 4),
            "weighted_score": round(ws, 4),
        }

        # 通过 → 无需归因
        if ws >= self.PASS_THRESHOLD:
            return {**base, "label": "pass", "explanation": "整体通过, 无需归因"}

        scan_ok = scan_s >= self.SCAN_GOOD_THRESHOLD
        rc_fail = rc_s < self.ROOT_CAUSE_FAIL_THRESHOLD
        trigger_fail = trigger_s < self.TRIGGER_FAIL_THRESHOLD
        reasoning_fail = reasoning_s < self.REASONING_FAIL_THRESHOLD

        # Step 2 维度的综合状态
        step2_any_fail = rc_fail or trigger_fail or reasoning_fail

        # ── 归因判定 ──

        if not scan_ok and step2_any_fail:
            # Step 1 差 + Step 2 差 → 需要判断是否是级联
            # 如果扫描漏报了期望异常，导致根因也缺失，则归因为级联
            scan_detail = eval_result.get("dimension_details", {}).get("scan_quality", {})
            recall = scan_detail.get("recall", 1.0)
            if recall < 0.5:
                return {
                    **base,
                    "label": "scan_cascading",
                    "explanation": (
                        f"Step 1 扫描 recall={recall:.0%} 过低, "
                        f"导致 Step 2 连锁失分 (root_cause={rc_s:.0%}, reasoning={reasoning_s:.0%}). "
                        f"优先修复 Step 1 的异常检出能力."
                    ),
                }
            else:
                return {
                    **base,
                    "label": "compound_error",
                    "explanation": (
                        f"Step 1 (scan={scan_s:.0%}) 和 Step 2 "
                        f"(rc={rc_s:.0%}, trigger={trigger_s:.0%}, reasoning={reasoning_s:.0%}) "
                        f"同时有问题, recall={recall:.0%} 尚可但扫描分低, 需逐步排查."
                    ),
                }

        if not scan_ok and not step2_any_fail:
            # Step 1 差但 Step 2 勉强 OK → 扫描精度问题 (误报多但根因凑巧对了)
            return {
                **base,
                "label": "scan_cascading",
                "explanation": (
                    f"Step 1 扫描分偏低 (scan={scan_s:.0%}) 拖累总分, "
                    f"但 Step 2 推理结果尚可. 需优化 Step 1 precision/specificity."
                ),
            }

        if scan_ok and not step2_any_fail:
            # 两步都 OK 但综合分低 → 多半是 report/execution 拖后腿
            return {
                **base,
                "label": "pass",  # 不归因到 Step 1/2
                "explanation": (
                    f"Step 1+2 核心维度正常, 总分偏低可能源于 "
                    f"report ({ds.get('report', 0):.0%}) 或 "
                    f"execution ({ds.get('execution', 0):.0%})."
                ),
            }

        if scan_ok and step2_any_fail:
            # Step 1 OK 但 Step 2 有问题 → 精确定位 Step 2 哪里出错
            if trigger_fail and not rc_fail:
                return {
                    **base,
                    "label": "trigger_error",
                    "explanation": (
                        f"Step 1 扫描正确 (scan={scan_s:.0%}), "
                        f"根因判定也 OK (rc={rc_s:.0%}), "
                        f"但 needs_deep_rca/suggested_data 触发判定失败 (trigger={trigger_s:.0%}). "
                        f"需优化 Reasoner v2 的 deep_rca 决策逻辑."
                    ),
                }
            else:
                return {
                    **base,
                    "label": "reasoning_error",
                    "explanation": (
                        f"Step 1 扫描正确 (scan={scan_s:.0%}), "
                        f"但 Step 2 推理失分: root_cause={rc_s:.0%}, reasoning={reasoning_s:.0%}. "
                        f"异常数据已给到 Reasoner, 是推理/因果分析环节的问题."
                    ),
                }

        # fallback
        return {**base, "label": "compound_error",
                "explanation": "无法精确归因, 需人工审查"}

    def attribute_all(self, eval_results: List[Dict]) -> Dict:
        """
        批量归因 + 汇总统计。

        输入: List[V8ComplexEvaluator.evaluate() 返回值]
        输出: {
            "per_question": {case_id: attribution_dict, ...},
            "summary": {
                "total": int,
                "pass_count": int,
                "fail_count": int,
                "attribution_counts": {"scan_cascading": N, "reasoning_error": N, ...},
                "attribution_rates": {"scan_cascading": 0.xx, ...},
                "primary_bottleneck": str,  # 最大的失分来源
                "optimization_recommendation": str,  # 优化建议
            }
        }
        """
        per_q = {}
        counts = {"scan_cascading": 0, "reasoning_error": 0,
                  "trigger_error": 0, "compound_error": 0, "pass": 0}

        for r in eval_results:
            qid = r.get("question_id", "?")
            attr = self.attribute_single(r)
            per_q[qid] = attr
            counts[attr["label"]] = counts.get(attr["label"], 0) + 1

        total = len(eval_results)
        fail_count = total - counts["pass"]

        # 在失败题中计算各类占比
        fail_labels = {k: v for k, v in counts.items() if k != "pass"}
        rates = {k: round(safe_divide(v, fail_count), 4) for k, v in fail_labels.items()}

        # 判断主要瓶颈
        if fail_count == 0:
            primary = "none"
            recommendation = "所有题目通过, 无需优化"
        else:
            primary = max(fail_labels, key=fail_labels.get)
            recommendations = {
                "scan_cascading": (
                    "主要瓶颈在 Step 1 扫描. 建议: "
                    "(1) 检查 Commander 的 AnalysisFrame 是否覆盖了所有关键指标; "
                    "(2) 调整阈值策略, 当前可能过高导致漏报; "
                    "(3) 检查 Arbiter 是否在边界 case 上给出了正确裁决."
                ),
                "reasoning_error": (
                    "主要瓶颈在 Step 2 推理. 建议: "
                    "(1) 优化 CAUSAL_REASONING_PROMPT, 增强因果链推导能力; "
                    "(2) 检查多异常 case 的排序逻辑, 确保主因排在前面; "
                    "(3) 考虑为 Step 2 做隔离评测 (Layer B) 以精确定位问题."
                ),
                "trigger_error": (
                    "主要瓶颈在 needs_deep_rca 触发判定. 建议: "
                    "(1) 检查后置校验规则 (confidence + arbiter_widen 逻辑); "
                    "(2) 优化 suggested_data 的 LLM 生成 prompt; "
                    "(3) 考虑增加 B 组 case 的覆盖."
                ),
                "compound_error": (
                    "Step 1 和 Step 2 混合失分. 建议: "
                    "(1) 优先按 scan_cascading 方向修复 Step 1; "
                    "(2) 修复后重跑评测, 观察 Step 2 得分是否连带提升; "
                    "(3) 如 Step 2 仍低, 再做 Layer B 隔离评测."
                ),
            }
            recommendation = recommendations.get(primary, "需人工审查")

        return {
            "per_question": per_q,
            "summary": {
                "total": total,
                "pass_count": counts["pass"],
                "fail_count": fail_count,
                "attribution_counts": counts,
                "attribution_rates": rates,
                "primary_bottleneck": primary,
                "optimization_recommendation": recommendation,
            },
        }


# ============================================================================
# 6. MockAgent v9 (适配统一管线 + analysis_depth)
# ============================================================================

class MockAgentV8:
    """模拟 v9 统一管线输出, 用于管线验证

    v6: route 统一为 "complex", 新增 analysis_depth 输出
    """

    @staticmethod
    def run(question: Dict, df: Any) -> Dict:
        """v6: df 是 pd.DataFrame (单平台单表) 或 dict (多表,向后兼容)"""
        expected_step1 = question.get("expected_step1", {})
        expected_step2 = question.get("expected_step2", {})
        expected_depth = question.get("expected_analysis_depth", "causal")

        # Step 1 mock
        anomalies = []
        for ea in expected_step1.get("expected_anomalies", []):
            pct_range = ea.get("change_pct_range", [-20, -10])
            anomalies.append({
                "dimension": ea["dimension"],
                "direction": ea.get("direction", "down"),
                "change_pct": round(np.random.uniform(pct_range[0], pct_range[1]), 2),
                "significant": True,
            })

        scan_state = {
            "anomalies": anomalies,
            "normal_dimensions": expected_step1.get("expected_normal", []),
            "scan_time_ms": np.random.randint(3000, 8000),
        }

        # Step 2 mock
        needs_deep = expected_step2.get("needs_deep_rca", False)
        confidence = expected_step2.get("confidence", "high")
        expected_rc = expected_step2.get("expected_root_causes", [])
        suggested_types = expected_step2.get("expected_suggested_data_types", [])

        root_causes = []
        for rc in expected_rc:
            dim = rc if isinstance(rc, str) else rc.get("dimension", rc.get("name", ""))
            root_causes.append({
                "dimension": dim,
                "description": f"{dim} 异常导致下游变化",
                "impact": "significant",
            })

        chain = " → ".join(rc["dimension"] for rc in root_causes) + " → GMV" if root_causes else "无异常"

        reason_result = {
            "root_causes": root_causes,
            "confidence": confidence,
            "causal_chain": chain,
            "needs_deep_rca": needs_deep,
            "suggested_data_types": suggested_types,
        }

        # Report mock
        report_kw = question.get("expected_report", {}).get("must_mention_keywords", [])
        text = f"分析报告: {'、'.join(report_kw) if report_kw else '所有指标'} 分析完成。"
        if root_causes:
            text += f" 根本原因: {chain}。"
            for rc in root_causes:
                text += f" {rc['dimension']} 异常导致下游指标变化。"
        else:
            text += " 未发现显著异常，各维度在正常波动范围内。"
        if needs_deep:
            text += f" 注意: 由于数据有限，建议补充 {'、'.join(suggested_types)} 进行深入分析。"

        # v6: route 统一为 complex, 补 analysis_depth
        events = [
            {"event_type": "intent_classification", "iteration": 0,
             "output_data": {"route": "complex"}, "success": True},
            {"event_type": "commander_plan", "iteration": 0,
             "output_data": {"analysis_depth": expected_depth}, "success": True},
        ]

        return {
            "route": "complex",
            "analysis_depth": expected_depth,   # v6
            "total_time_ms": 5000,
            "steps": [
                {"type": "commander_plan", "success": True, "result": {"success": True}},
                {"type": "scan_loop", "success": True, "result": {"success": True}},
                {"type": "reason_engine", "success": True, "result": {"success": True}},
            ],
            "scan_state": scan_state,
            "reason_result": reason_result,
            "needs_deep_rca": needs_deep,
            "suggested_data_types": suggested_types,
            "report": {"success": True, "full_content": text, "content": text},
            "final_report": {"success": True, "full_content": text},
            "deep_rca_result": None,
            "chart_paths": [],
            "error": None,
            "trace_log": {
                "session_id": f"mock_v9_{question['id']}",
                "events": events,
                "statistics": {"total_duration_ms": 5000},
            },
        }


# ============================================================================
# 7. 评测运行器 — v5.1: 多文件加载 + Layer C 集成
# ============================================================================

class EvalRunner:
    """加载评测集 → 逐题运行 → 评分 → Layer C 归因 → 按 case 隔离输出 → 汇总"""

    def __init__(self, eval_file: str, data_dir: str,
                 agent_module: str = None, output_dir: str = None,
                 step3_data_dir: str = None, run_step3: bool = False,
                 inter_question_delay: float = 0):
        self.eval_file = eval_file
        self.data_dir = data_dir
        self.step3_data_dir = step3_data_dir
        self.run_step3 = run_step3
        self.inter_question_delay = inter_question_delay

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(eval_file) or ".", "eval_results_v7")
        os.makedirs(self.output_dir, exist_ok=True)

        self.evaluator = V8ComplexEvaluator()
        self.attribution_analyzer = ErrorAttributionAnalyzer()  # v5.1

        with open(eval_file, "r", encoding="utf-8") as f:
            self.eval_set = json.load(f)
        self.questions = self.eval_set.get("questions", [])

        self.orchestrator = self._load_agent(agent_module)

    def _load_agent(self, module_name: str = None):
        if not module_name:
            return None
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        module_path = Path(module_name)
        if module_path.exists():
            sys.path.insert(0, str(module_path.parent))
            module_name = module_path.stem
        try:
            mod = importlib.import_module(module_name)
            return {
                "orchestrator": getattr(mod, "AgentOrchestrator")(),
                "DataSourceType": getattr(mod, "DataSourceType"),
            }
        except Exception as e:
            print(f"\n  ⚠️  加载 Agent 模块 '{module_name}' 失败: {e}")
            print(f"  ⚠️  将使用 MockAgent\n")
            return None

    def _case_output_dir(self, case_id: str) -> str:
        case_dir = os.path.join(self.output_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        return case_dir

    def _collect_artifacts(self, case_id: str, agent_result: Dict, case_dir: str):
        collected = []
        for path in agent_result.get("chart_paths", []):
            if os.path.exists(path):
                dest = os.path.join(case_dir, os.path.basename(path))
                if os.path.abspath(path) != os.path.abspath(dest):
                    shutil.copy2(path, dest)
                collected.append(dest)
        for pattern in [f"*{case_id}*.png", f"*{case_id}*.jpg", f"*{case_id}*.svg"]:
            for fp in glob.glob(pattern):
                dest = os.path.join(case_dir, os.path.basename(fp))
                if os.path.abspath(fp) != os.path.abspath(dest):
                    shutil.copy2(fp, dest)
                collected.append(dest)
        return collected

    # ── v6: 单文件加载 (v3 单平台单表 MVP) + 向后兼容多文件 ──

    def _load_data(self, question: Dict) -> Any:
        """
        v6 优先支持 v3 单表格式, 向后兼容旧多表格式:

        方式 A (v3 新版, 优先): data_dir/C01/店铺经营概况.csv
            → 返回 pd.DataFrame

        方式 B (向后兼容, 旧多表): data_dir/C01/trade_summary.csv + traffic_summary.csv
            → 返回 Dict[str, pd.DataFrame]

        方式 C (更老,单文件扁平): data_dir/C01_data.csv
            → 返回 pd.DataFrame
        """
        case_id = question["id"]

        # 方式 A: v3 单平台单表 (优先)
        data_files = question.get("data_files", [])
        if data_files and len(data_files) == 1:
            fname = data_files[0]
            # 优先 case 子目录
            path = os.path.join(self.data_dir, case_id, fname)
            if not os.path.exists(path):
                path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                return pd.read_csv(path)

        # 方式 C: 老版单文件
        single_file = question.get("data_file", "")
        if single_file:
            path = os.path.join(self.data_dir, single_file)
            if os.path.exists(path):
                return pd.read_csv(path)

        # 方式 B: 旧多表 (向后兼容)
        if data_files and len(data_files) > 1:
            tables = {}
            for fname in data_files:
                path = os.path.join(self.data_dir, case_id, fname)
                if not os.path.exists(path):
                    path = os.path.join(self.data_dir, fname)
                if os.path.exists(path):
                    key = Path(fname).stem
                    tables[key] = pd.read_csv(path)
            if tables:
                return tables

        # 方式 D: 按 case_id 子目录自动探测
        case_subdir = os.path.join(self.data_dir, case_id)
        if os.path.isdir(case_subdir):
            # 优先找 店铺经营概况.csv
            preferred = os.path.join(case_subdir, "店铺经营概况.csv")
            if os.path.exists(preferred):
                return pd.read_csv(preferred)

            # 否则多 CSV 聚合 (向后兼容)
            tables = {}
            for csv_path in sorted(Path(case_subdir).glob("*.csv")):
                if "supplementary" in str(csv_path):
                    continue
                key = csv_path.stem
                tables[key] = pd.read_csv(str(csv_path))
            if len(tables) == 1:
                return list(tables.values())[0]
            if tables:
                return tables

        return None

    def _run_agent(self, question: Dict) -> Dict:
        data = self._load_data(question)

        if data is None:
            return {
                "route": "error",
                "analysis_depth": "unknown",
                "total_time_ms": 0, "steps": [],
                "scan_state": None, "reason_result": None,
                "needs_deep_rca": False, "suggested_data_types": [],
                "report": None, "final_report": None,
                "deep_rca_result": None, "chart_paths": [],
                "error": f"数据文件不存在 (case={question['id']}, data_dir={self.data_dir})",
                "trace_log": None,
            }

        if self.orchestrator:
            DST = self.orchestrator["DataSourceType"]
            orch = self.orchestrator["orchestrator"]

            # v6: 正常路径 — 单表 DataFrame 直接走 process()
            if isinstance(data, pd.DataFrame):
                return orch.process(
                    user_query=question["question"],
                    data_source=DST.CSV,
                    uploaded_df=data)

            # 多表 fallback: 合并或使用 process_multi (向后兼容)
            if isinstance(data, dict):
                if hasattr(orch, 'process_multi'):
                    return orch.process_multi(
                        user_query=question["question"],
                        data_source=DST.CSV,
                        uploaded_dfs=data)
                merged = self._merge_tables(data)
                return orch.process(
                    user_query=question["question"],
                    data_source=DST.CSV,
                    uploaded_df=merged)

        return MockAgentV8.run(question, data)

    @staticmethod
    def _merge_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        v5.1: 将多张报表合并为单表, 用于不支持多表输入的旧版 Agent.
        合并策略: 按日期+平台+品类 outer merge.
        """
        dfs = list(tables.values())
        if len(dfs) == 0:
            return pd.DataFrame()
        if len(dfs) == 1:
            return dfs[0]

        # 寻找公共 merge key
        common_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            common_cols &= set(df.columns)

        merge_keys = []
        for candidate in ["stat_date", "platform", "shop_name", "category_name"]:
            if candidate in common_cols:
                merge_keys.append(candidate)

        if not merge_keys:
            # 无公共列, 直接横向拼接 (fallback)
            return pd.concat(dfs, axis=1)

        result = dfs[0]
        for df in dfs[1:]:
            # 去掉重复列 (除了 merge key)
            overlap = set(result.columns) & set(df.columns) - set(merge_keys)
            df_clean = df.drop(columns=[c for c in overlap if c in df.columns], errors='ignore')
            result = pd.merge(result, df_clean, on=merge_keys, how="outer")

        return result

    # ── v6.1 FIX: Step 3 执行支持 ──

    def _load_step3_data(self, question: Dict) -> List[pd.DataFrame]:
        """加载 Step 3 补充数据。

        v7.1 改动:
          - 所有搜索路径都排除主数据文件 (店铺经营概况.csv 等)
          - 优先搜索 supplementary/ 子目录
          - 返回 List[pd.DataFrame]

        搜索路径优先级:
          1. step3_data_dir/CASE_ID/supplementary/ (v7.1 新增)
          2. step3_data_dir/CASE_ID/ 下的非主数据 CSV
          3. data_dir/CASE_ID/supplementary/
          4. data_dir/CASE_ID/ 下排除主数据的额外 CSV
        """
        case_id = question["id"]
        found_dfs: List[pd.DataFrame] = []

        # 主数据文件名集合 — 所有路径都需要排除
        main_files = set(question.get("data_files", []))
        main_files.add("店铺经营概况.csv")

        def _is_supplementary(csv_path: Path) -> bool:
            """判断文件是否为补充数据（非主数据）"""
            return csv_path.name not in main_files

        def _load_from_dir(dir_path: str, label: str) -> List[pd.DataFrame]:
            """从指定目录加载所有补充数据 CSV"""
            dfs = []
            target = Path(dir_path)
            if not target.is_dir():
                return dfs

            # 优先检查 supplementary/ 子目录
            supp_sub = target / "supplementary"
            if supp_sub.is_dir():
                for csv_path in sorted(supp_sub.glob("*.csv")):
                    print(f"  📂 Step3 加载 ({label}/supplementary): {csv_path.name}")
                    dfs.append(pd.read_csv(str(csv_path)))
                if dfs:
                    return dfs

            # 否则从目录根加载，排除主数据文件
            for csv_path in sorted(target.glob("*.csv")):
                if _is_supplementary(csv_path):
                    print(f"  📂 Step3 加载 ({label}): {csv_path.name}")
                    dfs.append(pd.read_csv(str(csv_path)))
            return dfs

        # 路径 1: step3_data_dir/CASE_ID/
        if self.step3_data_dir:
            found_dfs = _load_from_dir(
                os.path.join(self.step3_data_dir, case_id), "step3_data_dir")
            if found_dfs:
                return found_dfs

        # 路径 2: data_dir/CASE_ID/
        found_dfs = _load_from_dir(
            os.path.join(self.data_dir, case_id), "data_dir")

        return found_dfs

    def _try_run_step3(self, question: Dict, agent_result: Dict) -> Tuple[Optional[Dict], bool]:
        """尝试执行 Step 3 深度根因分析。

        v7 改动:
          - 支持多个补充数据文件 (List[DataFrame])
          - 逐个文件调用 process_deep_rca，累积结果
          - 当 Agent API 只接受单个 supplementary_df 时，多文件场景依次传入

        Returns:
            (step3_result, step3_ran): 结果字典和是否成功执行标志
        """
        case_id = question["id"]
        print(f"  🔬 Step 3: 尝试执行深度根因分析 ({case_id})...")

        # 加载补充数据 (v7: 返回 List[DataFrame])
        supp_dfs = self._load_step3_data(question)
        if supp_dfs:
            for i, sdf in enumerate(supp_dfs):
                print(f"  📊 补充数据[{i}]: {sdf.shape[0]} 行 × {sdf.shape[1]} 列  列={list(sdf.columns[:6])}...")
        else:
            print(f"  ⚠️  未找到补充数据，Step 3 将仅基于原始数据执行")

        # 加载原始数据
        original_df = self._load_data(question)
        if isinstance(original_df, dict):
            original_df = self._merge_tables(original_df)

        try:
            orch = self.orchestrator["orchestrator"]
            prior_state = agent_result.get("_intermediate_state", {})

            # v7: 多文件合并策略 — 将所有补充文件 concat 传入
            # 不同 schema 的 df concat 会产生 NaN 列，但:
            #   - _identify_supplementary_type 基于列名检测，两种 type 都能被识别
            #   - _build_deep_rca_hypotheses 基于 schema 生成假设，能覆盖所有补充维度
            #   - ReAct 的 PythonAgent 可以通过 dropna 分别处理不同数据
            supplementary_df = None
            if supp_dfs:
                if len(supp_dfs) == 1:
                    supplementary_df = supp_dfs[0]
                else:
                    # 多文件: 逐个调用 process_deep_rca，累积结果
                    # 先用第一个文件跑完整流程，再用后续文件追加验证
                    all_step3_results = []
                    current_prior = prior_state
                    for i, sdf in enumerate(supp_dfs):
                        print(f"  🔬 Step 3 轮次 {i+1}/{len(supp_dfs)}...")
                        t0 = time.time()
                        partial_result = orch.process_deep_rca(
                            user_query=question["question"],
                            prior_state=current_prior,
                            uploaded_df=original_df,
                            supplementary_df=sdf,
                        )
                        elapsed = time.time() - t0
                        print(f"  🔬 Step 3 轮次 {i+1} 耗时: {elapsed:.1f}s")
                        all_step3_results.append(partial_result)

                    # 合并多轮结果: 取最后一轮的报告，累积所有步骤
                    merged_result = all_step3_results[-1].copy()
                    all_steps = []
                    total_ms = 0
                    all_charts = []
                    for r in all_step3_results:
                        all_steps.extend(r.get("steps", []))
                        total_ms += r.get("total_time_ms", 0)
                        all_charts.extend(r.get("chart_paths", []))
                    merged_result["steps"] = all_steps
                    merged_result["total_time_ms"] = total_ms
                    merged_result["chart_paths"] = all_charts
                    return merged_result, True

            t0 = time.time()
            step3_result = orch.process_deep_rca(
                user_query=question["question"],
                prior_state=prior_state,
                uploaded_df=original_df,
                supplementary_df=supplementary_df,
            )
            elapsed = time.time() - t0
            print(f"  🔬 Step 3 耗时: {elapsed:.1f}s")
            return step3_result, True

        except Exception as e:
            print(f"  ❌ Step 3 执行失败: {e}")
            traceback.print_exc()
            return None, False

    def run_single(self, question: Dict) -> Dict:
        qid = question["id"]
        is_mock = not self.orchestrator
        group = "B" if question.get("expected_step2", {}).get("needs_deep_rca", False) else "A"
        case_dir = self._case_output_dir(qid)

        tag = "⚠️ MOCK" if is_mock else "🤖 Agent"
        print(f"\n{'=' * 60}")
        print(f"  [{tag}] {qid} ({group}组): {question['question'][:50]}...")
        print(f"  类别: {question.get('category')}  输出: {case_dir}")
        print(f"{'=' * 60}")

        try:
            t0 = time.time()
            agent_result = self._run_agent(question)
            elapsed = time.time() - t0
            if not is_mock:
                print(f"  Agent {elapsed:.1f}s")

            collected = self._collect_artifacts(qid, agent_result, case_dir)
            if collected:
                print(f"  📁 收集 {len(collected)} 个文件")

            step3_ran = False

            # ✅ FIX: 当 --run-step3 启用且 Agent 判定 needs_deep_rca=True 时，
            # 模拟用户确认并加载补充数据，执行 Step 3 深度根因分析
            if (self.run_step3
                    and not is_mock
                    and self.orchestrator
                    and agent_result.get("needs_deep_rca")
                    and agent_result.get("_intermediate_state")):
                step3_result, step3_ran = self._try_run_step3(
                    question, agent_result)
                if step3_ran and step3_result:
                    # 合并 Step 3 结果到 agent_result
                    agent_result["deep_rca_result"] = step3_result.get("deep_rca_result")
                    agent_result["steps"] = (
                        agent_result.get("steps", [])
                        + step3_result.get("steps", []))
                    agent_result["total_time_ms"] = (
                        agent_result.get("total_time_ms", 0)
                        + step3_result.get("total_time_ms", 0))
                    # 如果 Step 3 生成了新报告，覆盖
                    if step3_result.get("report") or step3_result.get("final_report"):
                        agent_result["report"] = (
                            step3_result.get("report")
                            or step3_result.get("final_report"))
                    if step3_result.get("chart_paths"):
                        agent_result["chart_paths"] = (
                            agent_result.get("chart_paths", [])
                            + step3_result.get("chart_paths", []))
                    print(f"  ✅ Step 3 完成")

            eval_result = self.evaluator.evaluate(question, agent_result, step3_ran)
            eval_result["is_mock"] = is_mock

            # v5.1: Layer C 逐题归因
            attribution = self.attribution_analyzer.attribute_single(eval_result)
            eval_result["error_attribution"] = attribution

            self._print_score_summary(eval_result)

            detail_path = os.path.join(case_dir, "eval_detail.json")
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump({
                    "eval_result": eval_result,
                    "agent_result": {
                        "route": agent_result.get("route"),
                        "total_time_ms": agent_result.get("total_time_ms"),
                        "error": agent_result.get("error"),
                        "steps": agent_result.get("steps", []),
                        "scan_state": agent_result.get("scan_state"),
                        "reason_result": agent_result.get("reason_result"),
                        "needs_deep_rca": agent_result.get("needs_deep_rca"),
                        "suggested_data_types": agent_result.get("suggested_data_types"),
                        "chart_paths": agent_result.get("chart_paths", []),
                        "report": agent_result.get("report") or agent_result.get("final_report"),
                        "final_report": agent_result.get("final_report"),
                        "deep_rca_result": agent_result.get("deep_rca_result"),
                        "trace_log": agent_result.get("trace_log"),
                    },
                    "collected_artifacts": collected,
                }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

            return eval_result

        except Exception as e:
            print(f"  ❌ 异常: {e}")
            traceback.print_exc()
            return {
                "question_id": qid, "question": question["question"],
                "category": question.get("category", ""), "group": group,
                "weighted_score": 0.0,
                "dimension_scores": {d: 0.0 for d in SCORING_WEIGHTS},
                "error": str(e),
                "error_attribution": {
                    "label": "compound_error",
                    "explanation": f"评测执行异常: {e}",
                },
            }

    def run_all(self, filter_ids: List[str] = None) -> Dict:
        questions = self.questions
        if filter_ids:
            questions = [q for q in questions if q["id"] in filter_ids]
        if not questions:
            print("⚠️  没有匹配的评测题目")
            return {}

        a_n = sum(1 for q in questions if not q.get("expected_step2", {}).get("needs_deep_rca", False))
        b_n = len(questions) - a_n

        print(f"\n{'#' * 60}")
        print(f"  Complex 评测 v7.0 — {len(questions)} 题 (A:{a_n} B:{b_n})")
        print(f"  评测集: {self.eval_file}")
        print(f"  Agent: {'🤖 真实' if self.orchestrator else '⚠️ Mock'}")
        print(f"  输出: {self.output_dir}")
        print(f"  ✨ Layer C 误差归因: 已启用")
        print(f"{'#' * 60}")

        results = []
        for idx, q in enumerate(questions):
            if idx > 0 and self.inter_question_delay > 0:
                time.sleep(self.inter_question_delay)
            results.append(self.run_single(q))

        # v5.1: Layer C 批量归因
        attribution_result = self.attribution_analyzer.attribute_all(results)

        summary = self._generate_summary(results, attribution_result)

        with open(os.path.join(self.output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        self._print_final_report(summary)

        txt_path = os.path.join(self.output_dir, "eval_report.txt")
        self._save_text_report(summary, txt_path)
        return summary

    def _print_score_summary(self, r: Dict):
        ds = r.get("dimension_scores", {})
        ws = r.get("weighted_score", 0)
        attr = r.get("error_attribution", {})
        attr_label = attr.get("label", "?")

        print(f"\n  📊 {'管线验证' if r.get('is_mock') else '综合'}: {ws:.2%} ({r.get('group','?')}组)"
              f"  归因: {attr_label}")
        for dim, label in [("analysis_depth","深度判定"), ("scan_quality","扫描"), ("execution","执行"),
                           ("root_cause","根因"), ("deep_rca_trigger","触发"),
                           ("report","报告"), ("reasoning","推理"), ("deep_rca","深度RCA")]:
            s = ds.get(dim, 0)
            bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
            print(f"    {label}({SCORING_WEIGHTS.get(dim,0):>3.0%}): {bar} {s:.0%}")

    def _generate_summary(self, results: List[Dict],
                          attribution_result: Dict = None) -> Dict:
        valid = [r for r in results if "weighted_score" in r]
        scores = [r["weighted_score"] for r in valid]
        a_res = [r for r in valid if r.get("group") == "A"]
        b_res = [r for r in valid if r.get("group") == "B"]

        cat_stats = {}
        for r in valid:
            cat = r.get("category", "unknown")
            cat_stats.setdefault(cat, {"scores": [], "count": 0})
            cat_stats[cat]["scores"].append(r["weighted_score"])
            cat_stats[cat]["count"] += 1
        for c in cat_stats:
            s = cat_stats[c]["scores"]
            cat_stats[c]["avg_score"] = round(np.mean(s), 4) if s else 0

        dim_avgs = {}
        for dim in SCORING_WEIGHTS:
            ds = [r.get("dimension_scores", {}).get(dim, 0) for r in valid]
            dim_avgs[dim] = round(np.mean(ds), 4) if ds else 0

        pass_n = sum(1 for s in scores if s >= 0.6)
        weakest = min(dim_avgs, key=dim_avgs.get) if dim_avgs else None

        summary = {
            "metadata": {
                "eval_file": self.eval_file, "eval_time": datetime.now().isoformat(),
                "eval_version": "v6.0", "agent_target": "v9",
                "total_questions": len(results),
                "agent_mode": "real" if self.orchestrator else "mock",
                "scoring_weights": SCORING_WEIGHTS,
            },
            "overall": {
                "avg_score": round(np.mean(scores), 4) if scores else 0,
                "max_score": round(max(scores), 4) if scores else 0,
                "min_score": round(min(scores), 4) if scores else 0,
                "pass_rate": round(safe_divide(pass_n, len(results)), 4),
                "pass_count": pass_n, "total": len(results),
                "group_a_avg": round(float(np.mean([r["weighted_score"] for r in a_res])), 4) if a_res else 0,
                "group_b_avg": round(float(np.mean([r["weighted_score"] for r in b_res])), 4) if b_res else 0,
                "group_a_count": len(a_res), "group_b_count": len(b_res),
                "weakest_dimension": weakest,
                "weakest_dimension_score": dim_avgs.get(weakest, 0) if weakest else 0,
            },
            "dimension_averages": dim_avgs,
            "category_stats": cat_stats,
            "per_question": [
                {"id": r.get("question_id"), "category": r.get("category"),
                 "group": r.get("group"), "weighted_score": r.get("weighted_score"),
                 "dimension_scores": r.get("dimension_scores"),
                 "error_attribution": r.get("error_attribution", {}).get("label", "?"),
                 "error": r.get("error"),
                 "detail_file": f"{r.get('question_id')}/eval_detail.json"}
                for r in results
            ],
        }

        # v5.1: Layer C 归因汇总
        if attribution_result:
            summary["error_attribution"] = attribution_result["summary"]

        return summary

    def _print_final_report(self, summary: Dict):
        o = summary["overall"]
        da = summary["dimension_averages"]
        is_mock = summary.get("metadata", {}).get("agent_mode") != "real"

        print(f"\n\n{'#' * 60}")
        print(f"  评测汇总 v6.0 (8 维度 + Layer C 归因, 适配 app_v9)")
        print(f"{'#' * 60}")
        if is_mock:
            print(f"\n  ⚠️  MockAgent 模式 — 分数仅验证管线")

        print(f"\n  总题: {o['total']}  通过率: {o['pass_rate']:.0%} ({o['pass_count']}/{o['total']})")
        print(f"  均分: {o['avg_score']:.2%}  A组: {o['group_a_avg']:.2%}({o['group_a_count']})  B组: {o['group_b_avg']:.2%}({o['group_b_count']})")
        if o.get("weakest_dimension"):
            print(f"  ⚠️ 短板: {o['weakest_dimension']} ({o['weakest_dimension_score']:.2%})")

        print(f"\n  各维度:")
        for dim, label in [("analysis_depth","深度判定"), ("scan_quality","扫描"), ("execution","执行"),
                           ("root_cause","根因"), ("deep_rca_trigger","触发"),
                           ("report","报告"), ("reasoning","推理"), ("deep_rca","深度RCA")]:
            avg = da.get(dim, 0)
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            flag = " ⚠️" if avg < 0.5 else ""
            print(f"    {label:6s}({SCORING_WEIGHTS.get(dim,0):>3.0%}): {bar} {avg:.2%}{flag}")

        # v5.1: Layer C 归因统计
        attr = summary.get("error_attribution", {})
        if attr:
            print(f"\n  {'─' * 50}")
            print(f"  🔍 Layer C — 误差归因分析")
            print(f"  {'─' * 50}")
            counts = attr.get("attribution_counts", {})
            rates = attr.get("attribution_rates", {})
            fail_n = attr.get("fail_count", 0)
            print(f"  通过: {attr.get('pass_count', 0)}  未通过: {fail_n}")

            if fail_n > 0:
                label_names = {
                    "scan_cascading": "Step1级联",
                    "reasoning_error": "Step2推理",
                    "trigger_error":   "触发判定",
                    "compound_error":  "混合错误",
                }
                for label, name in label_names.items():
                    n = counts.get(label, 0)
                    rate = rates.get(label, 0)
                    if n > 0:
                        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
                        print(f"    {name:8s}: {bar} {n}题 ({rate:.0%})")

                bottleneck = attr.get("primary_bottleneck", "?")
                bn_name = label_names.get(bottleneck, bottleneck)
                print(f"\n  🎯 主要瓶颈: {bn_name}")
                rec = attr.get("optimization_recommendation", "")
                if rec:
                    # 按句号分行, 更易读
                    for line in rec.split("; "):
                        print(f"     {line.strip()}")

        print(f"\n  逐题:")
        print(f"    {'ID':5s} {'组':2s} {'类别':16s} {'综合':6s} {'归因':12s} "
              f"{'深度':4s} {'扫描':4s} {'执行':4s} {'根因':4s} {'触发':4s} {'报告':4s} {'推理':4s} {'RCA':4s}")
        print(f"    {'-' * 100}")
        for pq in summary["per_question"]:
            ds = pq.get("dimension_scores", {})
            attr_label = pq.get("error_attribution", "?")
            # 归因标签颜色标记
            attr_icon = {
                "pass": "  ✅",
                "scan_cascading": "🔴S1",
                "reasoning_error": "🟡S2",
                "trigger_error": "🟠TG",
                "compound_error": "⚪MX",
            }.get(attr_label, "  ? ")
            print(f"    {pq['id']:5s} {pq.get('group','?'):2s} {pq.get('category',''):16s} "
                  f"{pq['weighted_score']:5.0%} {attr_icon:>4s}         "
                  + " ".join(f"{ds.get(d,0):4.0%}" for d in SCORING_WEIGHTS))

        print(f"\n  输出: {self.output_dir}/")
        print(f"    ├── eval_summary.json + eval_report.txt")
        print(f"    ├── C01/eval_detail.json (+图表)")
        print(f"    └── ...")

    def _save_text_report(self, summary: Dict, path: str):
        lines = ["=" * 60, "Complex 评测报告 v7.0 (app_v9 统一管线 + Layer C 归因)", "=" * 60]
        o = summary["overall"]
        lines.append(f"总题: {o['total']}  通过率: {o['pass_rate']:.0%}  均分: {o['avg_score']:.2%}")
        lines.append(f"A组: {o['group_a_avg']:.2%}({o['group_a_count']})  B组: {o['group_b_avg']:.2%}({o['group_b_count']})")
        lines.append("")
        lines.append("维度均分:")
        for dim in SCORING_WEIGHTS:
            lines.append(f"  {dim}: {summary['dimension_averages'].get(dim,0):.2%}")

        # v5.1: 归因统计
        attr = summary.get("error_attribution", {})
        if attr:
            lines.append("")
            lines.append("Layer C 误差归因:")
            lines.append(f"  通过: {attr.get('pass_count', 0)}  未通过: {attr.get('fail_count', 0)}")
            for label, n in attr.get("attribution_counts", {}).items():
                if label != "pass" and n > 0:
                    lines.append(f"  {label}: {n}题")
            lines.append(f"  主要瓶颈: {attr.get('primary_bottleneck', '?')}")
            rec = attr.get("optimization_recommendation", "")
            if rec:
                lines.append(f"  建议: {rec}")

        lines.append("")
        lines.append("逐题:")
        for pq in summary["per_question"]:
            ds = pq.get("dimension_scores", {})
            attr_label = pq.get("error_attribution", "?")
            lines.append(f"  {pq['id']} [{pq.get('group','?')}] {pq['weighted_score']:.2%} 归因={attr_label}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ============================================================================
# 8. CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Complex 评测 v7.0 (app_v9 统一管线 + Layer C 归因)")
    parser.add_argument("--eval-file", required=True, help="评测集 JSON")
    parser.add_argument("--data-dir", required=True, help="数据目录 (支持单CSV或多报表子目录)")
    parser.add_argument("--agent-module", default=None, help="Agent 模块")
    parser.add_argument("--question", default=None, nargs="*", help="指定 case ID")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--step3-data-dir", default=None, help="Step 3 补充数据目录")
    parser.add_argument("--run-step3", action="store_true", help="启用 Step 3")
    parser.add_argument("--delay", type=float, default=0, help="题间延迟秒数")
    args = parser.parse_args()

    runner = EvalRunner(
        eval_file=args.eval_file, data_dir=args.data_dir,
        agent_module=args.agent_module, output_dir=args.output_dir,
        step3_data_dir=args.step3_data_dir, run_step3=args.run_step3,
        inter_question_delay=args.delay)

    summary = runner.run_all(filter_ids=args.question)
    mode = summary.get("metadata", {}).get("agent_mode", "mock")
    if mode != "real":
        sys.exit(1)
    sys.exit(0 if summary.get("overall", {}).get("avg_score", 0) >= 0.3 else 1)


if __name__ == "__main__":
    main()
