"""
eval_v9_csv_only_fixed.py - CSV-Only 评测系统 v9.5 (bidirectional CN/EN + fuzzy stopwords)
=============================================================
修复内容 (v9.1):
  FIX-1: 新增 CN_EN_FIELD_MAP，解决中文列名无法匹配英文 field_name 的问题
  FIX-2: _fuzzy_field_match 增加中英文翻译映射层
  FIX-3: _find_column 增加中英文翻译映射层 (FIX-14 进一步增强)
  FIX-4: _validate_comparison 支持多行 DataFrame（行标签展开 + metric/value 透视）
  FIX-5: extract_single_value 对 DataFrame 改用 iloc[0]（agent 通常将答案排在首行）
  FIX-10: CN_EN_FIELD_MAP 增加 "环比变化率%" 等百分号变体
  FIX-11: extract_single_value 中 count_keywords 路径增加 metric/value 透视优先检查

修复内容 (v9.2):
  FIX-12: 新增 answer 字段精确提取 — extract_single_value 和 _validate_comparison
          优先从 agent 显式输出的 answer dict 提取数值，原有逻辑保留为 fallback
  fixed: FIX-12 改为 answer 和 flat dict 分开存放，匹配时先查 answer 再 fallback，避免 _rowN 抢先命中
  FIX-13: _fuzzy_field_match synonyms 补充电商指标近义词（cvr↔conversion, gmv↔revenue,
          traffic↔visitors, spend↔cost, marketing↔ad, comp↔competitor, total↔sum,
          share↔proportion, price↔unit）。修复列顺序不同导致 type-inference fallback
          选错列的问题（如 weighted_avg_cvr 无法匹配 weighted_conversion_rate）
  FIX-14: _find_column 在 CN/EN 映射与类型推断 fallback 之间新增第2.9层——
          调用 _fuzzy_field_match 做近义词模糊匹配，使 _find_column 与
          _fuzzy_field_match 具备同等的 synonym 匹配能力，彻底修复
          grouped_values 场景下 value_col 因 synonym 缺失而 fallback 选错列的问题

新增内容 (v9.3):
  FIX-15: _validate_grouped_values 新增 answer-first 短路 — 当 agent 输出
          answer 为 list[dict] 时，直接按 group_field 构建 map 精确比对，
          跳过 _find_column 瀑布链。answer 缺失或命中率不足时自动 fallback 到原逻辑。
  FIX-16: _validate_ranked_list 新增 answer-first 短路 — 从 answer list 提取排序，
          跳过 _find_column。
  FIX-17: _validate_cross_dimension 新增 answer-first 短路 — 从 answer list
          构建复合 key map 精确比对。

修复内容 (v9.4):
  FIX-18a: _validate_comparison 支持 answer 为 list[dict] — 展开为复合键
           (row_label)_(numeric_col) 填入 answer_dict，并增加 value-based 反向映射
           让 expected field_name 直接出现在 answer_dict 中。
           修复 answer 为 list 时 answer_dict 始终为空、answer-first 短路完全失效的问题。
  FIX-18b: _validate_comparison 中 fields 提前统一计算，并传入
           _flatten_multirow_for_comparison，修复原 expected.get("fields", [])
           传空列表导致 value-based fallback 失效的问题。

修复内容 (v9.5):
  FIX-19: _cn_en_match 双向化 — 原函数仅支持 expected=英文/actual=中文 方向，
          当 expected 为中文 field_name（如 "环比变化率"）而 answer key 为英文（如
          "change_pct"）时匹配失败，导致 extract_single_value 和 FIX-15 answer-first
          均无法提取值。新增方向3和方向4，覆盖 expected=中文/actual=英文 的场景。
  FIX-20: _fuzzy_field_match 新增修饰词停用词集（best/worst/top/bottom/first/last/
          max/min 等）— 当 token 重叠仅包含停用词时跳过该候选，修复 "best_channel"
          误匹配 "best_roi"、"best_category" 误匹配 "best_roi" 等 false positive。
"""

import os
import sys
import json
import time
import argparse
import inspect
import importlib.util
import traceback
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============================================================================
# FIX-1: 中英文字段名映射表
# ============================================================================

# 中文列名 → 可能对应的英文 field_name（支持多对多）
CN_EN_FIELD_MAP = {
    # 维度
    "渠道": ["channel"],
    "品类": ["category"],
    "日期": ["date"],
    "月份": ["month", "period"],
    # GMV 相关
    "总gmv": ["total_gmv", "gmv"],
    "gmv变化": ["gmv_change", "change"],
    # 环比 / 增长
    # ★ FIX-10: 增加 "环比变化率%" 变体 (agent 常用 % 而非 (%))
    "环比变化率(%)": ["change_pct", "mom_pct", "growth_rate_pct", "change_percent"],
    "环比变化率%": ["change_pct", "mom_pct", "growth_rate_pct", "change_percent"],
    "环比变化率": ["change_pct", "mom_pct", "growth_rate_pct", "change_percent"],
    "增长率(%)": ["growth_rate_pct", "growth_pct", "change_percent"],
    "增长率%": ["growth_rate_pct", "growth_pct", "change_percent"],
    "增长率": ["growth_rate_pct", "growth_pct", "change_percent"],
    # 加权 / 客单价
    "有效客单价": ["effective_avg_price", "avg_price", "gmv_per_conversion"],
    "加权平均转化率": ["avg_cvr", "weighted_cvr"],
    # 流量 / 营销
    "总流量": ["total_traffic", "traffic"],
    "总营销费用": ["total_marketing", "marketing_spend"],
    "总有效转化数": ["total_effective_conversions", "effective_conversions"],
    # 占比
    "gmv占比(%)": ["gmv_share_pct", "share_pct"],
    "gmv占比%": ["gmv_share_pct", "share_pct"],
    "gmv占比": ["gmv_share_pct", "share_pct"],
    # 月份标签 → 字段名
    "7月": ["jul_gmv", "jul"],
    "8月": ["aug_gmv", "aug"],
    "9月": ["sep_gmv", "sep"],
    "10月": ["oct_gmv", "oct"],
    "上半年": ["h1_monthly_avg", "h1"],
    "第三季度": ["q3_monthly_avg", "q3"],
    "环比变化": ["change", "gmv_change"],
    "变化": ["change", "gmv_change"],
    # ★ FIX-10: agent 使用 "9月GMV" / "10月GMV" 作为列名的场景
    "9月gmv": ["sep_gmv", "sep"],
    "10月gmv": ["oct_gmv", "oct"],
}

# 反向映射: 英文 field_name → 中文列名 set
_EN_CN_FIELD_MAP: Dict[str, set] = {}
for cn, en_list in CN_EN_FIELD_MAP.items():
    for en in en_list:
        _EN_CN_FIELD_MAP.setdefault(en.lower(), set()).add(cn.lower())


def _cn_en_match(expected_field: str, actual_key: str) -> bool:
    """
    检查 expected_field 与 actual_key 是否通过中英文映射匹配。
    ★ FIX-19: 双向支持 — 无论哪边是中文、哪边是英文都能匹配。
    """
    ef = expected_field.lower()
    ak = actual_key.lower()

    # 方向1: actual_key 是中文 → 查 CN_EN_FIELD_MAP 看是否包含 expected_field(英文)
    cn_entry = CN_EN_FIELD_MAP.get(ak)
    if cn_entry and ef in [x.lower() for x in cn_entry]:
        return True

    # 方向2: expected_field 是英文 → 查 _EN_CN_FIELD_MAP 看是否包含 actual_key(中文)
    cn_set = _EN_CN_FIELD_MAP.get(ef, set())
    if ak in cn_set:
        return True

    # ★ FIX-19: 方向3: expected_field 是中文 → 查 CN_EN_FIELD_MAP 看是否包含 actual_key(英文)
    cn_entry_ef = CN_EN_FIELD_MAP.get(ef)
    if cn_entry_ef and ak in [x.lower() for x in cn_entry_ef]:
        return True

    # ★ FIX-19: 方向4: actual_key 是英文 → 查 _EN_CN_FIELD_MAP 看是否包含 expected_field(中文)
    cn_set_ak = _EN_CN_FIELD_MAP.get(ak, set())
    if ef in cn_set_ak:
        return True

    return False


# ============================================================================
# JSON 编码器
# ============================================================================

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            try:
                return super().default(obj)
            except TypeError:
                return str(obj)


# ============================================================================
# FIX-2: 数据提取器 — _fuzzy_field_match 增加中英文映射
# ============================================================================

def _fuzzy_field_match(expected_field: str, actual_keys: list) -> Optional[str]:
    """模糊匹配字段名：精确 → 子串 → 中英文映射 → 词重叠"""
    f_lower = expected_field.lower()
    actual_lower_map = {k.lower(): k for k in actual_keys}  # lower → original

    # 第一轮：精确匹配
    if f_lower in actual_lower_map:
        return actual_lower_map[f_lower]

    # 第二轮：子串匹配
    for key_lower, key_orig in actual_lower_map.items():
        if f_lower in key_lower or key_lower in f_lower:
            return key_orig

    # ★ FIX-2: 第2.5轮：中英文映射匹配
    for key_lower, key_orig in actual_lower_map.items():
        if _cn_en_match(f_lower, key_lower):
            return key_orig

    # 第三轮：分词 + 近义词重叠
    # ★ FIX-13: 补充电商指标近义词，解决 cvr↔conversion 等匹配失败
    synonyms = {
        "pct": {"rate", "percent", "percentage", "ratio"},
        "rate": {"pct", "percent", "percentage", "ratio", "cvr", "conversion"},
        "percent": {"pct", "rate", "percentage", "ratio"},
        "growth": {"increase", "change", "delta"},
        "avg": {"average", "mean", "ma", "weighted"},
        "ma": {"avg", "average", "mean"},
        "mean": {"avg", "average", "ma"},
        "cnt": {"count", "num", "number"},
        "diff": {"difference", "change", "delta"},
        "rolling": {"moving", "ma"},
        # ★ FIX-13: 电商指标近义词
        "cvr": {"conversion", "rate", "convert"},
        "conversion": {"cvr", "convert", "rate"},
        "gmv": {"revenue", "sales"},
        "revenue": {"gmv", "sales"},
        "sales": {"gmv", "revenue"},
        "traffic": {"visitors", "visits", "uv"},
        "visitors": {"traffic", "visits", "uv"},
        "spend": {"cost", "expense", "budget"},
        "cost": {"spend", "expense", "budget"},
        "marketing": {"ad", "advertising", "promotion"},
        "ad": {"marketing", "advertising", "promotion"},
        "comp": {"competitor", "market", "competing"},
        "competitor": {"comp", "market", "competing"},
        "total": {"sum", "all", "overall"},
        "sum": {"total", "all", "overall"},
        "share": {"proportion", "ratio", "fraction"},
        "price": {"unit", "aov"},
    }
    f_tokens = set(f_lower.replace("-", "_").split("_"))
    f_expanded = set(f_tokens)
    for t in f_tokens:
        f_expanded.update(synonyms.get(t, set()))

    # ★ FIX-20: 修饰词停用词 — 不携带领域语义，仅靠这些词重叠不应构成匹配
    _modifier_stopwords = {
        "best", "worst", "top", "bottom", "first", "last",
        "max", "min", "most", "least", "new", "old",
        "all", "each", "per", "by", "of", "the",
    }

    best_key, best_score = None, 0.0
    for key_lower, key_orig in actual_lower_map.items():
        k_tokens = set(key_lower.replace("-", "_").split("_"))
        raw_overlap = k_tokens & f_expanded
        overlap = len(raw_overlap)
        denom = min(len(f_tokens), len(k_tokens))
        score = overlap / denom if denom > 0 else 0

        # ★ FIX-20: 如果重叠词全是修饰词（如 "best"），不构成有效匹配
        meaningful_overlap = raw_overlap - _modifier_stopwords
        if overlap > 0 and len(meaningful_overlap) == 0:
            continue

        if score > best_score:
            best_score = score
            best_key = key_orig
    if best_score >= 0.5:
        return best_key
    return None


class ResultExtractor:
    """从 Agent 输出中提取结构化数据"""

    @staticmethod
    def extract_dataframe(final_result: Any) -> Optional[pd.DataFrame]:
        if final_result is None:
            return None
        if isinstance(final_result, pd.DataFrame):
            return final_result
        if isinstance(final_result, dict):
            if 'data' in final_result:
                data = final_result['data']
                if isinstance(data, pd.DataFrame):
                    return data
                elif isinstance(data, list):
                    return pd.DataFrame(data)
            if 'columns' in final_result and 'rows' in final_result:
                return pd.DataFrame(final_result['rows'], columns=final_result['columns'])
            # ★ FIX-6: Handle {"rows": [list_of_dicts]} without "columns" key
            if 'rows' in final_result:
                rows = final_result['rows']
                if isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict):
                    return pd.DataFrame(rows)
            try:
                return pd.DataFrame([final_result])
            except:
                pass
        if isinstance(final_result, list):
            if len(final_result) > 0 and isinstance(final_result[0], dict):
                return pd.DataFrame(final_result)
            return pd.DataFrame({'value': final_result})
        if isinstance(final_result, (int, float, str)):
            return pd.DataFrame({'value': [final_result]})
        return None

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
            return result
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _try_metric_value_pivot(df: pd.DataFrame, field_name: str) -> Optional[float]:
        """
        ★ FIX-9: 检测 metric/value 透视格式并从中提取值。
        适用于: [{"metric": "slope", "value": -12349}, {"metric": "r_squared", "value": 0.03}]
        """
        if df is None or df.empty or not field_name or len(df.columns) > 4:
            return None
        # 找 label 列和 value 列
        label_candidates = ["metric", "指标", "item", "name", "key"]
        value_candidates = ["value", "值", "val", "result"]
        label_col = value_col = None
        cols_lower = {c.lower(): c for c in df.columns}
        for lc in label_candidates:
            if lc in cols_lower:
                label_col = cols_lower[lc]
                break
        for vc in value_candidates:
            if vc in cols_lower:
                value_col = cols_lower[vc]
                break
        if not label_col or not value_col:
            return None
        # 在 label 列中模糊匹配 field_name
        metric_names = [str(v).strip() for v in df[label_col].dropna()]
        matched_metric = _fuzzy_field_match(field_name, metric_names)
        if matched_metric:
            row = df[df[label_col].astype(str).str.strip() == matched_metric]
            if len(row) > 0:
                return ResultExtractor._safe_float(row.iloc[0][value_col])
        return None

    # ★ FIX-5 revised + FIX-11: 增加 metric/value 透视优先于 row_count 回退
    # ★ FIX-12: 优先从 answer 字段精确提取
    @staticmethod
    def extract_single_value(final_result: Any, field_name: str = None) -> Optional[float]:
        if final_result is None:
            return None
        if isinstance(final_result, (int, float)):
            return ResultExtractor._safe_float(final_result)

        # ★ FIX-12: answer 字段短路 — 精确匹配，无需模糊
        if isinstance(final_result, dict) and 'answer' in final_result:
            ans = final_result['answer']
            if isinstance(ans, dict):
                if field_name and field_name in ans:
                    return ResultExtractor._safe_float(ans[field_name])
                # field_name 模糊匹配 answer 的 key
                if field_name:
                    matched = _fuzzy_field_match(field_name, list(ans.keys()))
                    if matched:
                        return ResultExtractor._safe_float(ans[matched])
                # 只有一个 key 时直接取
                if len(ans) == 1:
                    # 只有 1 个数值型 value 时直接取（不依赖 field_name 匹配）
                    numeric_vals = {k: v for k, v in ans.items() if isinstance(v, (int, float))}
                    if len(numeric_vals) == 1:
                        return ResultExtractor._safe_float(list(numeric_vals.values())[0])

        if isinstance(final_result, pd.DataFrame):
            if len(final_result) > 0:
                if field_name and field_name in final_result.columns:
                    col_series = final_result[field_name].dropna()
                    if len(col_series) > 0:
                        return ResultExtractor._safe_float(col_series.iloc[-1])
                    return ResultExtractor._safe_float(final_result.iloc[-1][field_name])
                if field_name:
                    matched = _fuzzy_field_match(field_name, list(final_result.columns))
                    if matched:
                        col_series = final_result[matched].dropna()
                        if len(col_series) > 0:
                            return ResultExtractor._safe_float(col_series.iloc[-1])
                        return ResultExtractor._safe_float(final_result.iloc[-1][matched])
                    # ★ FIX-9: metric/value 透视格式检测
                    pivot_result = ResultExtractor._try_metric_value_pivot(
                        final_result, field_name)
                    if pivot_result is not None:
                        return pivot_result
                return ResultExtractor._safe_float(final_result.iloc[0, 0])
        if isinstance(final_result, dict):
            if field_name and field_name in final_result:
                return ResultExtractor._safe_float(final_result[field_name])

            # ★ FIX-6b: 处理 {"rows": [...]} 格式
            if 'rows' in final_result:
                rows = final_result['rows']
                if isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict):
                    df_from_rows = pd.DataFrame(rows)
                    result = ResultExtractor.extract_single_value(df_from_rows, field_name)
                    if result is not None:
                        return result
                    # ★ FIX-11: 如果 DataFrame 提取失败，不要直接 return None
                    # 继续往下走，但也不要让 row_count 误匹配

            data = final_result.get('data')
            is_detail_list = (isinstance(data, list)
                              and len(data) > 1
                              and isinstance(data[0], dict))

            if is_detail_list:
                if field_name:
                    columns = list(data[0].keys())
                    matched_col = _fuzzy_field_match(field_name, columns)
                    if matched_col:
                        return ResultExtractor._safe_float(data[-1][matched_col])
                    for rec in reversed(data):
                        if field_name in rec:
                            return ResultExtractor._safe_float(rec[field_name])

                    # ★ FIX-11: 在使用 row_count 之前，先尝试 metric/value 透视
                    pivot_df = pd.DataFrame(data)
                    pivot_result = ResultExtractor._try_metric_value_pivot(
                        pivot_df, field_name)
                    if pivot_result is not None:
                        return pivot_result

                    count_keywords = {"count", "cnt", "num", "number", "total", "条"}
                    f_tokens = set(field_name.lower().replace("-", "_").split("_"))
                    if f_tokens & count_keywords and 'row_count' in final_result:
                        return ResultExtractor._safe_float(final_result['row_count'])
                    if 'row_count' in final_result:
                        return ResultExtractor._safe_float(final_result['row_count'])
                else:
                    if 'row_count' in final_result:
                        return ResultExtractor._safe_float(final_result['row_count'])
                    return float(len(data))

            if data is not None:
                result = ResultExtractor.extract_single_value(data, field_name)
                if result is not None:
                    return result

            for k, v in final_result.items():
                if k in ('data', 'columns', 'executed_code', 'preview',
                         'correction_history', 'task_id', 'error',
                         'error_type', 'chart_path', 'chart_type',
                         'rows', 'total_retries'):  # ★ FIX-11: 排除 rows 和 total_retries
                    continue
                # ★ FIX-11: 对 row_count 做特殊处理 — 仅当 field_name 明确要求 count 时才返回
                if k == 'row_count':
                    if field_name:
                        count_keywords = {"count", "cnt", "num", "number", "total", "条",
                                          "row_count", "行数", "记录数"}
                        f_lower = field_name.lower()
                        # 只有 field_name 本身就是 count 类词时才用 row_count
                        # 排除 "outlier_count" 这种复合词（outlier 才是关键语义）
                        f_tokens = set(f_lower.replace("-", "_").split("_"))
                        non_count_tokens = f_tokens - count_keywords
                        if non_count_tokens:
                            # 复合词如 "outlier_count"，有非 count 成分 → 跳过 row_count
                            continue
                    if isinstance(v, (int, float)):
                        return float(v)
                    continue
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                    return float(v[0])

        if isinstance(final_result, list) and final_result:
            if isinstance(final_result[0], (int, float)):
                return ResultExtractor._safe_float(final_result[0])
            if isinstance(final_result[0], dict):
                if field_name:
                    for rec in [final_result[0], final_result[-1]]:
                        if field_name in rec:
                            return ResultExtractor._safe_float(rec[field_name])
                    columns = list(final_result[0].keys())
                    matched_col = _fuzzy_field_match(field_name, columns)
                    if matched_col:
                        return ResultExtractor._safe_float(final_result[0][matched_col])
                for val in final_result[0].values():
                    result = ResultExtractor._safe_float(val)
                    if result is not None:
                        return result
        return None


# ============================================================================
# 答案验证器
# ============================================================================

class AnswerValidator:
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def validate(self, question: Dict, final_result: Any) -> Dict:
        expected = question.get("expected_answer", {})
        answer_type = expected.get("type", "unknown")
        criteria = question.get("evaluation_criteria", {})
        tolerance = criteria.get("value_tolerance", self.tolerance)

        validators = {
            "single_value": self._validate_single_value,
            "grouped_values": self._validate_grouped_values,
            "time_series": self._validate_time_series,
            "ranked_list": self._validate_ranked_list,
            "comparison": self._validate_comparison,
            "cross_dimension": self._validate_cross_dimension,
            "derived_calculation": self._validate_grouped_values,
            "filtered_combinations": self._validate_filtered,
            "extremes": self._validate_extremes,
        }

        validator = validators.get(answer_type, self._validate_default)
        return validator(expected, final_result, tolerance, criteria)

    def _validate_single_value(self, expected, actual, tolerance, criteria):
        result = {"score": 0.0, "passed": False, "details": "", "checks": [], "answer_type": "single_value"}
        expected_value = expected.get("value")
        if expected_value is None:
            result["details"] = "期望值为空"
            result["score"] = 0.5
            return result

        field_name = expected.get("field")

        # ★ FIX-7: 多候选值策略
        candidates = set()
        # ★ FIX-21: answer dict 数值直提 — 绕过 extract_single_value 的复杂路径
        if isinstance(actual, dict) and isinstance(actual.get('answer'), dict):
            ans = actual['answer']
            numeric_vals = {k: v for k, v in ans.items()
                            if isinstance(v, (int, float)) and not np.isnan(v)}
            if field_name:
                # 先精确/模糊匹配 field_name
                if field_name in ans:
                    v = ResultExtractor._safe_float(ans[field_name])
                    if v is not None:
                        candidates.add(v)
                else:
                    matched = _fuzzy_field_match(field_name, list(ans.keys()))
                    if matched:
                        v = ResultExtractor._safe_float(ans[matched])
                        if v is not None:
                            candidates.add(v)
            # 不论 field_name 有没有命中，只要 answer 只有 1 个数值就加入候选
            if len(numeric_vals) == 1:
                candidates.add(float(list(numeric_vals.values())[0]))

        # 候选 1: 默认提取
        default_val = ResultExtractor.extract_single_value(actual, field_name)
        if default_val is not None:
            candidates.add(default_val)

        # 候选 2/3: 从 DataFrame 的首行和末行分别提取
        actual_df = ResultExtractor.extract_dataframe(actual)
        if actual_df is not None and not actual_df.empty and field_name:
            matched_col = _fuzzy_field_match(field_name, list(actual_df.columns))
            if matched_col and matched_col in actual_df.columns:
                col_series = actual_df[matched_col].dropna()
                if len(col_series) > 0:
                    first_v = ResultExtractor._safe_float(col_series.iloc[0])
                    last_v = ResultExtractor._safe_float(col_series.iloc[-1])
                    if first_v is not None:
                        candidates.add(first_v)
                    if last_v is not None:
                        candidates.add(last_v)

            # ★ FIX-11: 额外尝试 metric/value 透视提取
            pivot_val = ResultExtractor._try_metric_value_pivot(actual_df, field_name)
            if pivot_val is not None:
                candidates.add(pivot_val)

        if not candidates:
            result["checks"].append("❌ 无法提取实际值")
            return result

        # 从候选值中选择与期望值误差最小的
        actual_value = min(candidates, key=lambda v: self._calculate_error(expected_value, v))

        error = self._calculate_error(expected_value, actual_value)
        if error <= tolerance:
            result["score"] = 1.0
            result["passed"] = True
            result["checks"].append(f"✅ 数值匹配: {actual_value:.2f} (期望: {expected_value:.2f}, 误差: {error:.4f})")
        elif error <= tolerance * 2:
            result["score"] = 0.7
            result["checks"].append(f"⚠️ 数值接近: {actual_value:.2f} (期望: {expected_value:.2f}, 误差: {error:.4f})")
        else:
            result["score"] = 0.3
            result["checks"].append(f"❌ 数值不匹配: {actual_value:.2f} (期望: {expected_value:.2f}, 误差: {error:.4f})")
        result["details"] = f"实际: {actual_value:.2f}, 期望: {expected_value:.2f}, 误差: {error:.2%}"
        result["actual_value"] = actual_value
        result["expected_value"] = expected_value
        return result

    def _validate_grouped_values(self, expected, actual, tolerance, criteria):
        result = {"score": 0.0, "passed": False, "details": "", "checks": [], "answer_type": "grouped_values"}
        expected_values = expected.get("values", [])
        if not expected_values:
            result["score"] = 0.5
            return result
        actual_df = ResultExtractor.extract_dataframe(actual)
        if actual_df is None or actual_df.empty:
            result["checks"].append("❌ 无法提取结果数据")
            return result
        group_field = expected.get("group_field", "")
        value_field = expected.get("value_field", "")
        expected_row_count = expected.get("row_count", len(expected_values))
        if len(actual_df) == expected_row_count:
            result["checks"].append(f"✅ 行数正确: {len(actual_df)}")
            row_score = 1.0
        elif len(actual_df) >= expected_row_count * 0.8:
            result["checks"].append(f"⚠️ 行数接近: {len(actual_df)} (期望: {expected_row_count})")
            row_score = 0.7
        else:
            result["checks"].append(f"❌ 行数不匹配: {len(actual_df)} (期望: {expected_row_count})")
            row_score = 0.3
        expected_map = {}
        for item in expected_values:
            key = item.get(group_field, str(list(item.values())[0]))
            value = item.get(value_field, list(item.values())[-1])
            expected_map[str(key).lower()] = float(value) if isinstance(value, (int, float)) else value

        # ★ FIX-15: answer-first 短路 — 从 agent 显式输出的 answer list[dict] 精确比对
        answer_list = None
        if isinstance(actual, dict) and 'answer' in actual:
            ans = actual['answer']
            if isinstance(ans, list) and len(ans) > 0 and isinstance(ans[0], dict):
                answer_list = ans
        if answer_list is not None:
            answer_map = {}
            for item in answer_list:
                # 在 answer 的 key 中查找 group_field 和 value_field
                a_key = item.get(group_field) or item.get(value_field.split("_")[0] if value_field else "", None)
                a_val = item.get(value_field)
                # fallback: 模糊匹配 key
                if a_key is None and group_field:
                    mk = _fuzzy_field_match(group_field, list(item.keys()))
                    if mk:
                        a_key = item[mk]
                if a_val is None and value_field:
                    mv = _fuzzy_field_match(value_field, list(item.keys()))
                    if mv:
                        a_val = item[mv]
                if a_key is not None and a_val is not None:
                    answer_map[str(a_key).lower()] = a_val

            # 命中率足够时（>=50% expected keys 能在 answer_map 中找到），使用 answer_map
            if len(answer_map) >= len(expected_map) * 0.5:
                matched = 0
                total = len(expected_map)
                for exp_key, exp_val in expected_map.items():
                    if exp_key in answer_map:
                        act_val = answer_map[exp_key]
                        if isinstance(exp_val, (int, float)):
                            try:
                                error = self._calculate_error(exp_val, float(act_val))
                                if error <= tolerance * 2:
                                    matched += 1
                                    result["checks"].append(
                                        f"✅ {exp_key}: {act_val} ≈ {exp_val} (answer-first)")
                                else:
                                    result["checks"].append(
                                        f"❌ {exp_key}: {act_val} ≠ {exp_val} (误差: {error:.4f})")
                            except (TypeError, ValueError):
                                result["checks"].append(f"⚠️ {exp_key}: 类型转换失败")
                        else:
                            if str(act_val).lower() == str(exp_val).lower():
                                matched += 1
                    else:
                        result["checks"].append(f"⚠️ {exp_key}: answer 中未找到")
                value_score = matched / total if total > 0 else 0
                if value_score >= 0.8:
                    result["checks"].insert(0, f"✅ answer-first 匹配: {matched}/{total}")
                elif value_score >= 0.5:
                    result["checks"].insert(0, f"⚠️ answer-first 部分匹配: {matched}/{total}")
                else:
                    result["checks"].insert(0, f"❌ answer-first 不匹配: {matched}/{total}")
                result["score"] = row_score * 0.3 + value_score * 0.7
                result["passed"] = result["score"] >= 0.8
                result["details"] = (f"answer-first | 行数: {len(actual_df)}/{expected_row_count}, "
                                     f"值匹配: {matched}/{total}")
                return result
        # ★ FIX-15 END — fallback: 使用 _find_column 从 result_df 提取（原逻辑）

        matched = 0
        total = len(expected_map)
        _sample_exp_val = list(expected_map.values())[0] if expected_map else None
        _expect_numeric = isinstance(_sample_exp_val, (int, float))

        group_col = self._find_column(actual_df, group_field, role="group",
                                      exclude_col=value_field)
        value_col = self._find_column(actual_df, value_field, role="value",
                                      exclude_col=group_field,
                                      expect_numeric=_expect_numeric)

        for _, row in actual_df.iterrows():
            if group_col and group_col in actual_df.columns:
                actual_key = str(row[group_col]).lower()
            else:
                actual_key = str(row.iloc[0]).lower()

            actual_value = None
            if value_col and value_col in actual_df.columns:
                if _expect_numeric:
                    actual_value = float(row[value_col]) if pd.notna(row[value_col]) else None
                else:
                    actual_value = str(row[value_col]).strip() if pd.notna(row[value_col]) else None
            if actual_value is None and len(row) > 1:
                try:
                    actual_value = float(row.iloc[-1]) if pd.notna(row.iloc[-1]) else None
                except (TypeError, ValueError):
                    actual_value = str(row.iloc[-1]).strip() if pd.notna(row.iloc[-1]) else None

            if actual_key in expected_map and actual_value is not None:
                exp_val = expected_map[actual_key]
                if isinstance(exp_val, (int, float)):
                    error = self._calculate_error(exp_val, actual_value)
                    if error <= tolerance * 2:
                        matched += 1
                else:
                    if str(actual_value).lower() == str(exp_val).lower():
                        matched += 1
        value_score = matched / total if total > 0 else 0
        if value_score >= 0.8:
            result["checks"].append(f"✅ 数值匹配: {matched}/{total}")
        elif value_score >= 0.5:
            result["checks"].append(f"⚠️ 部分匹配: {matched}/{total}")
        else:
            result["checks"].append(f"❌ 数值不匹配: {matched}/{total}")
        result["score"] = row_score * 0.3 + value_score * 0.7
        result["passed"] = result["score"] >= 0.8
        result["details"] = f"行数: {len(actual_df)}/{expected_row_count}, 值匹配: {matched}/{total}"
        return result

    def _validate_time_series(self, expected, actual, tolerance, criteria):
        result = self._validate_grouped_values(expected, actual, tolerance, criteria)
        result["answer_type"] = "time_series"
        if criteria.get("chart_generated"):
            result["checks"].append("ℹ️ 需要生成图表")
            result["requires_chart"] = True
        return result

    def _validate_ranked_list(self, expected, actual, tolerance, criteria):
        result = {"score": 0.0, "passed": False, "details": "", "checks": [], "answer_type": "ranked_list"}
        expected_values = expected.get("values", [])
        if not expected_values:
            result["score"] = 0.5
            return result
        actual_df = ResultExtractor.extract_dataframe(actual)
        if actual_df is None or actual_df.empty:
            result["checks"].append("❌ 无法提取结果数据")
            return result
        group_field = expected.get("group_field", "")
        expected_order = [str(item.get(group_field, list(item.values())[0])).lower() for item in expected_values]

        # ★ FIX-16: answer-first 短路 — 从 answer list 提取排序
        answer_list = None
        if isinstance(actual, dict) and 'answer' in actual:
            ans = actual['answer']
            if isinstance(ans, list) and len(ans) > 0 and isinstance(ans[0], dict):
                answer_list = ans
        if answer_list is not None:
            actual_order_from_answer = []
            for item in answer_list:
                a_key = item.get(group_field)
                if a_key is None and group_field:
                    mk = _fuzzy_field_match(group_field, list(item.keys()))
                    if mk:
                        a_key = item[mk]
                if a_key is not None:
                    actual_order_from_answer.append(str(a_key).lower())
            if len(actual_order_from_answer) >= len(expected_order) * 0.8:
                require_order = criteria.get("require_correct_order", True)
                if require_order:
                    order_correct = actual_order_from_answer[:len(expected_order)] == expected_order
                    if order_correct:
                        result["checks"].append(f"✅ answer-first 顺序正确: {actual_order_from_answer[:len(expected_order)]}")
                        result["score"] = 1.0
                        result["passed"] = True
                    else:
                        set_match = set(actual_order_from_answer[:len(expected_order)]) == set(expected_order)
                        if set_match:
                            result["checks"].append("⚠️ answer-first 元素正确但顺序错误")
                            result["score"] = 0.6
                        else:
                            result["checks"].append("❌ answer-first 顺序不匹配")
                            result["score"] = 0.3
                else:
                    if set(actual_order_from_answer[:len(expected_order)]) == set(expected_order):
                        result["score"] = 1.0
                        result["passed"] = True
                        result["checks"].append("✅ answer-first 元素匹配（不要求顺序）")
                    else:
                        result["score"] = 0.3
                        result["checks"].append("❌ answer-first 元素不匹配")
                result["details"] = f"answer-first | 期望: {expected_order}, 实际: {actual_order_from_answer[:len(expected_order)]}"
                return result
        # ★ FIX-16 END — fallback: 使用 _find_column 从 result_df 提取（原逻辑）

        actual_order = []
        group_col = self._find_column(actual_df, group_field, role="group")
        for _, row in actual_df.iterrows():
            if group_col and group_col in actual_df.columns:
                actual_order.append(str(row[group_col]).lower())
            else:
                actual_order.append(str(row.iloc[0]).lower())
        require_order = criteria.get("require_correct_order", True)
        if require_order:
            order_correct = actual_order[:len(expected_order)] == expected_order
            if order_correct:
                result["checks"].append(f"✅ 顺序正确: {actual_order[:len(expected_order)]}")
                result["score"] = 1.0
                result["passed"] = True
            else:
                set_match = set(actual_order[:len(expected_order)]) == set(expected_order)
                if set_match:
                    result["checks"].append("⚠️ 元素正确但顺序错误")
                    result["score"] = 0.6
                else:
                    result["checks"].append("❌ 顺序不匹配")
                    result["score"] = 0.3
        else:
            if set(actual_order[:len(expected_order)]) == set(expected_order):
                result["score"] = 1.0
                result["passed"] = True
        result["details"] = f"期望: {expected_order}, 实际: {actual_order[:len(expected_order)]}"
        return result

    @staticmethod
    def _fuzzy_field_match(expected_field: str, actual_keys: list) -> Optional[str]:
        return _fuzzy_field_match(expected_field, actual_keys)

    @staticmethod
    def _compare_values(expected_val, actual_val, tolerance: float) -> Optional[bool]:
        try:
            exp_f = float(expected_val)
            act_f = float(actual_val)
            if exp_f == 0:
                return abs(act_f) < tolerance
            return abs(act_f - exp_f) / abs(exp_f) <= tolerance * 2
        except (TypeError, ValueError):
            pass
        try:
            exp_str = str(expected_val).strip()[:10]
            act_str = str(actual_val).strip()[:10]
            if len(exp_str) == 10 and "-" in exp_str:
                return exp_str == act_str
        except:
            pass
        return str(expected_val).strip().lower() == str(actual_val).strip().lower()

    def _validate_comparison(self, expected, actual, tolerance, criteria):
        result = {"score": 0.0, "passed": False, "details": "", "checks": [], "answer_type": "comparison"}
        expected_values = expected.get("values", {})
        if not expected_values:
            result["score"] = 0.5
            return result

        actual_df = ResultExtractor.extract_dataframe(actual)
        actual_dict = {}

        # ★ FIX-18b: 统一 fields 取值，避免传空列表导致 value-based fallback 失效
        fields = expected.get("fields", list(expected_values.keys()))

        if actual_df is not None and not actual_df.empty:
            if len(actual_df) == 1:
                for col in actual_df.columns:
                    actual_dict[col.lower()] = actual_df.iloc[0][col]
            else:
                actual_dict = self._flatten_multirow_for_comparison(
                    actual_df, expected_values, fields
                )
        elif isinstance(actual, dict):
            actual_dict = {k.lower(): v for k, v in actual.items()}

        # ★ FIX-12 revised + FIX-18a: answer 字段单独保存，匹配时优先查 answer
        # FIX-18a: 支持 answer 为 list[dict] — 展开为复合键 (label_value)_(numeric_col)
        answer_dict = {}
        if isinstance(actual, dict) and 'answer' in actual:
            ans = actual['answer']
            if isinstance(ans, dict):
                answer_dict = {k.lower(): v for k, v in ans.items()}
            elif isinstance(ans, list) and len(ans) > 0 and isinstance(ans[0], dict):
                # list[dict] 场景：按 label_col + numeric_col 展开为复合键
                ans_df = pd.DataFrame(ans)
                # 找 label 列 (字符串) 和 numeric 列
                label_col = None
                num_cols = []
                for c in ans_df.columns:
                    if ans_df[c].dtype == object or pd.api.types.is_string_dtype(ans_df[c]):
                        if label_col is None:
                            label_col = c
                    else:
                        try:
                            ans_df[c] = pd.to_numeric(ans_df[c])
                            num_cols.append(c)
                        except (TypeError, ValueError):
                            pass
                if label_col and num_cols:
                    for _, row in ans_df.iterrows():
                        row_label = str(row[label_col]).strip().lower()
                        for nc in num_cols:
                            val = row[nc]
                            if pd.notna(val):
                                nc_lower = nc.lower()
                                composite_key = f"{row_label}_{nc_lower}"
                                answer_dict[composite_key] = val
                    # 额外：用 expected_values 做 value-based 反向映射
                    # 让 expected field_name 直接出现在 answer_dict 中
                    for f in fields:
                        f_lower = f.lower()
                        if f_lower in answer_dict:
                            continue
                        exp_val = expected_values.get(f)
                        if exp_val is None:
                            continue
                        for _, row in ans_df.iterrows():
                            for nc in num_cols:
                                cell_val = row[nc]
                                if pd.notna(cell_val):
                                    try:
                                        if abs(float(cell_val) - float(exp_val)) / max(abs(float(exp_val)), 1e-10) < 0.05:
                                            answer_dict[f_lower] = cell_val
                                    except (TypeError, ValueError):
                                        pass

        # fields 已在上方统一计算（FIX-18b）
        matched = 0
        answer_keys = list(answer_dict.keys())
        actual_keys = list(actual_dict.keys())
        for f in fields:
            expected_val = expected_values.get(f)
            if expected_val is None:
                continue

            # 优先从 answer 匹配（key 干净，无 _rowN 干扰）
            matched_key = self._fuzzy_field_match(f, answer_keys) if answer_keys else None
            if matched_key is not None:
                actual_val = answer_dict[matched_key]
                source = f"answer.{matched_key}"
            else:
                # fallback: 从 flatten 后的 actual_dict 匹配
                matched_key = self._fuzzy_field_match(f, actual_keys)
                if matched_key is None:
                    result["checks"].append(f"❌ {f}: 未找到匹配字段 (实际列: {actual_keys})")
                    continue
                actual_val = actual_dict[matched_key]
                source = matched_key

            match_result = self._compare_values(expected_val, actual_val, tolerance)
            if match_result is True:
                matched += 1
                result["checks"].append(f"✅ {f}: {actual_val} ≈ {expected_val} (匹配列: {source})")
            elif match_result is False:
                result["checks"].append(f"❌ {f}: {actual_val} ≠ {expected_val} (匹配列: {source})")
            else:
                result["checks"].append(f"⚠️ {f}: 无法比较 {actual_val} vs {expected_val}")

        result["score"] = matched / len(fields) if fields else 0.5
        result["passed"] = result["score"] >= 0.8
        result["details"] = f"字段匹配: {matched}/{len(fields)}"
        return result

    @staticmethod
    def _flatten_multirow_for_comparison(
        df: pd.DataFrame,
        expected_values: Dict,
        expected_fields: List[str],
    ) -> Dict[str, Any]:
        result_dict: Dict[str, Any] = {}
        cols_lower = {c.lower(): c for c in df.columns}

        metric_col = None
        value_col = None
        metric_candidates = ["metric", "指标", "item", "name"]
        value_candidates = ["value", "值", "val"]
        for mc in metric_candidates:
            if mc in cols_lower:
                metric_col = cols_lower[mc]
                break
        for vc in value_candidates:
            if vc in cols_lower:
                value_col = cols_lower[vc]
                break

        if metric_col and value_col and len(df.columns) <= 3:
            for _, row in df.iterrows():
                key = str(row[metric_col]).strip().lower()
                val = row[value_col]
                result_dict[key] = val
            if result_dict:
                return result_dict

        label_col = None
        num_cols = []
        for c in df.columns:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                if label_col is None:
                    label_col = c
            elif df[c].dtype in ['int64', 'float64', 'int32', 'float32']:
                num_cols.append(c)

        if label_col and num_cols:
            for _, row in df.iterrows():
                row_label = str(row[label_col]).strip().lower()

                for nc in num_cols:
                    val = row[nc]
                    if pd.isna(val):
                        continue
                    nc_lower = nc.lower()
                    composite_key = f"{row_label}_{nc_lower}"
                    result_dict[composite_key] = val
                    result_dict[nc_lower] = val

                cn_entry = CN_EN_FIELD_MAP.get(row_label)
                if cn_entry:
                    for en_field in cn_entry:
                        en_lower = en_field.lower()
                        if en_lower in [f.lower() for f in expected_fields]:
                            exp_val = expected_values.get(en_field)
                            if exp_val is not None:
                                for nc in num_cols:
                                    cell_val = row[nc]
                                    if pd.notna(cell_val):
                                        try:
                                            if abs(float(cell_val) - float(exp_val)) / max(abs(float(exp_val)), 1e-10) < 0.05:
                                                result_dict[en_lower] = cell_val
                                        except (TypeError, ValueError):
                                            pass
                            else:
                                for nc in num_cols:
                                    cell_val = row[nc]
                                    if pd.notna(cell_val):
                                        result_dict[en_lower] = cell_val
                                        break

            for f in expected_fields:
                f_lower = f.lower()
                if f_lower in result_dict:
                    continue
                exp_val = expected_values.get(f)
                if exp_val is None:
                    continue
                for _, row in df.iterrows():
                    for nc in num_cols:
                        cell_val = row[nc]
                        if pd.notna(cell_val):
                            try:
                                if abs(float(cell_val) - float(exp_val)) / max(abs(float(exp_val)), 1e-10) < 0.05:
                                    result_dict[f_lower] = cell_val
                            except (TypeError, ValueError):
                                pass

        if not result_dict:
            for col in df.columns:
                for i, val in enumerate(df[col]):
                    if pd.notna(val):
                        result_dict[f"{col.lower()}_row{i}"] = val
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    result_dict[col.lower()] = non_null.iloc[-1]

        return result_dict

    def _validate_cross_dimension(self, expected, actual, tolerance, criteria):
        result = {
            "score": 0.0, "passed": False, "details": "", "checks": [],
            "answer_type": "cross_dimension"
        }
        expected_values = expected.get("values", [])
        if not expected_values:
            result["score"] = 0.5
            result["checks"].append("ℹ️ 期望值为空，跳过验证")
            return result
        actual_df = ResultExtractor.extract_dataframe(actual)
        if actual_df is None or actual_df.empty:
            result["checks"].append("❌ 无法提取结果数据")
            return result

        dimensions = expected.get("dimensions", [])
        value_field = expected.get("value_field", "")
        expected_row_count = expected.get("row_count", len(expected_values))

        if len(actual_df) == expected_row_count:
            result["checks"].append(f"✅ 行数正确: {len(actual_df)}")
            row_score = 1.0
        elif len(actual_df) >= expected_row_count * 0.8:
            result["checks"].append(f"⚠️ 行数接近: {len(actual_df)} (期望: {expected_row_count})")
            row_score = 0.7
        else:
            result["checks"].append(f"❌ 行数不匹配: {len(actual_df)} (期望: {expected_row_count})")
            row_score = 0.3

        expected_map = {}
        for item in expected_values:
            if dimensions:
                key = tuple(str(item.get(d, "")).lower() for d in dimensions)
            else:
                vals = list(item.values())
                key = tuple(str(v).lower() for v in vals[:-1]) if len(vals) > 1 else (str(vals[0]).lower(),)
            try:
                val = float(item.get(value_field, list(item.values())[-1]))
                expected_map[key] = val
            except (TypeError, ValueError):
                pass

        if not expected_map:
            result["checks"].append("⚠️ 无法解析期望值映射")
            result["score"] = 0.3
            return result

        # ★ FIX-17: answer-first 短路 — 从 answer list[dict] 构建复合 key map
        answer_list = None
        if isinstance(actual, dict) and 'answer' in actual:
            ans = actual['answer']
            if isinstance(ans, list) and len(ans) > 0 and isinstance(ans[0], dict):
                answer_list = ans
        if answer_list is not None and dimensions:
            answer_map = {}
            for item in answer_list:
                # 构建复合 key
                key_parts = []
                all_found = True
                for d in dimensions:
                    d_val = item.get(d)
                    if d_val is None:
                        mk = _fuzzy_field_match(d, list(item.keys()))
                        if mk:
                            d_val = item[mk]
                    if d_val is not None:
                        key_parts.append(str(d_val).lower())
                    else:
                        all_found = False
                        break
                if not all_found:
                    continue
                # 提取 value
                a_val = item.get(value_field)
                if a_val is None and value_field:
                    mv = _fuzzy_field_match(value_field, list(item.keys()))
                    if mv:
                        a_val = item[mv]
                if a_val is not None:
                    try:
                        answer_map[tuple(key_parts)] = float(a_val)
                    except (TypeError, ValueError):
                        pass

            # 命中率足够时使用 answer_map
            if len(answer_map) >= len(expected_map) * 0.5:
                matched = 0
                total = len(expected_map)
                for exp_key, exp_val in expected_map.items():
                    if exp_key in answer_map:
                        act_val = answer_map[exp_key]
                        error = self._calculate_error(exp_val, act_val)
                        if error <= tolerance * 2:
                            matched += 1
                            result["checks"].append(
                                f"✅ {exp_key}: {act_val} ≈ {exp_val} (answer-first)")
                        else:
                            result["checks"].append(
                                f"❌ {exp_key}: {act_val} ≠ {exp_val} (误差: {error:.4f})")
                    else:
                        result["checks"].append(f"⚠️ {exp_key}: answer 中未找到")
                value_score = matched / total if total > 0 else 0.0
                if value_score >= 0.8:
                    result["checks"].insert(0, f"✅ answer-first 匹配: {matched}/{total}")
                elif value_score >= 0.5:
                    result["checks"].insert(0, f"⚠️ answer-first 部分匹配: {matched}/{total}")
                else:
                    result["checks"].insert(0, f"❌ answer-first 不匹配: {matched}/{total}")
                result["score"] = row_score * 0.3 + value_score * 0.7
                result["passed"] = result["score"] >= 0.8
                result["details"] = (f"answer-first | 行数: {len(actual_df)}/{expected_row_count}, "
                                     f"值匹配: {matched}/{total} (复合key)")
                return result
        # ★ FIX-17 END — fallback: 使用 _find_column 从 result_df 提取（原逻辑）

        actual_dim_cols = []
        if dimensions:
            for dim in dimensions:
                matched_col = self._find_column(actual_df, dim, role="group")
                actual_dim_cols.append(matched_col or "")
        else:
            for col in actual_df.columns:
                if actual_df[col].dtype == object:
                    actual_dim_cols.append(col)
                if len(actual_dim_cols) >= 2:
                    break

        actual_val_col = self._find_column(actual_df, value_field, role="value")
        if actual_val_col is None:
            actual_val_col = actual_df.columns[-1]

        matched = 0
        total = len(expected_map)
        unmatched_keys = list(expected_map.keys())

        for _, row in actual_df.iterrows():
            actual_key_parts = []
            for col in actual_dim_cols:
                if col and col in actual_df.columns:
                    actual_key_parts.append(str(row[col]).lower())
                else:
                    actual_key_parts.append("")
            actual_key = tuple(actual_key_parts)

            if actual_key in expected_map:
                try:
                    actual_val = float(row[actual_val_col]) if pd.notna(row[actual_val_col]) else None
                except (TypeError, ValueError):
                    actual_val = None
                if actual_val is not None:
                    exp_val = expected_map[actual_key]
                    denom = abs(exp_val) if abs(exp_val) > 1e-10 else 1.0
                    error = abs(actual_val - exp_val) / denom
                    if error <= tolerance * 2:
                        matched += 1
                        if actual_key in unmatched_keys:
                            unmatched_keys.remove(actual_key)

        value_score = matched / total if total > 0 else 0.0
        if value_score >= 0.8:
            result["checks"].append(f"✅ 数值匹配: {matched}/{total}")
        elif value_score >= 0.5:
            result["checks"].append(f"⚠️ 部分匹配: {matched}/{total}")
        else:
            result["checks"].append(f"❌ 数值不匹配: {matched}/{total}")
            if unmatched_keys:
                result["checks"].append(f"   未匹配的 key 示例: {unmatched_keys[:3]}")

        result["checks"].append(f"ℹ️ 维度列: {actual_dim_cols} | 数值列: {actual_val_col}")
        result["score"] = row_score * 0.3 + value_score * 0.7
        result["passed"] = result["score"] >= 0.8
        result["details"] = (
            f"行数: {len(actual_df)}/{expected_row_count}, "
            f"值匹配: {matched}/{total} (复合key匹配)"
        )
        return result

    def _validate_filtered(self, expected, actual, tolerance, criteria):
        result = {"score": 0.0, "passed": False, "details": "", "checks": [], "answer_type": "filtered_combinations"}
        expected_row_count = expected.get("row_count", 0)
        allow_empty = criteria.get("allow_empty_result", True)
        actual_df = ResultExtractor.extract_dataframe(actual)
        actual_count = len(actual_df) if actual_df is not None else 0
        if expected_row_count == 0:
            if actual_count == 0 or (actual_df is not None and actual_df.empty):
                result["score"] = 1.0
                result["passed"] = True
                result["checks"].append("✅ 正确返回空结果")
            elif allow_empty:
                result["score"] = 0.7
                result["checks"].append(f"⚠️ 返回了 {actual_count} 行（期望为空）")
            else:
                result["score"] = 0.3
                result["checks"].append("❌ 应返回空结果")
        else:
            if actual_count == expected_row_count:
                result["checks"].append(f"✅ 行数正确: {actual_count}")
                result["score"] = 0.8
                result["passed"] = True
            elif actual_count > 0:
                result["checks"].append(f"⚠️ 行数不匹配: {actual_count} (期望: {expected_row_count})")
                result["score"] = 0.5
            else:
                result["checks"].append(f"❌ 无结果 (期望: {expected_row_count})")
                result["score"] = 0.2
        result["details"] = f"结果行数: {actual_count}, 期望: {expected_row_count}"
        return result

    def _validate_extremes(self, expected, actual, tolerance, criteria):
        result = {"score": 0.0, "passed": False, "details": "", "checks": [], "answer_type": "extremes"}
        actual_df = ResultExtractor.extract_dataframe(actual)
        if actual_df is None or actual_df.empty:
            result["checks"].append("❌ 无法提取结果数据")
            return result
        expected_count = expected.get("row_count", 16)
        if len(actual_df) >= expected_count:
            result["checks"].append(f"✅ 完整数据: {len(actual_df)} 行")
            result["score"] = 0.8
            result["passed"] = True
        else:
            result["checks"].append(f"⚠️ 数据不完整: {len(actual_df)}/{expected_count}")
            result["score"] = 0.5
        result["details"] = f"包含 {len(actual_df)} 行数据"
        return result

    def _validate_default(self, expected, actual, tolerance, criteria):
        result = {"score": 0.5, "passed": False, "details": "未知类型", "checks": [], "answer_type": "unknown"}
        if actual is not None:
            actual_df = ResultExtractor.extract_dataframe(actual)
            if actual_df is not None and len(actual_df) > 0:
                result["score"] = 0.8
                result["passed"] = True
                result["checks"].append(f"✅ 有结果返回: {len(actual_df)} 行")
        return result

    # ★ FIX-3 + FIX-10 + FIX-14: _find_column 增加中英文翻译映射层 + 百分号标准化 + 近义词模糊匹配
    @staticmethod
    def _find_column(df: "pd.DataFrame", field_name: str,
                     role: str = "value",
                     exclude_col: str = None,
                     expect_numeric: bool = True) -> Optional[str]:
        if not field_name or df is None or df.empty:
            return None

        exclude_lower = exclude_col.lower() if exclude_col else ""
        columns = list(df.columns)

        def _is_excluded(col_name: str) -> bool:
            if not exclude_col:
                return False
            cl = col_name.lower()
            if cl == exclude_lower:
                return True
            if _cn_en_match(exclude_col, col_name):
                return True
            return False

        # ★ FIX-10: 标准化百分号变体用于匹配
        def _normalize_pct(s: str) -> str:
            """将 '(%)' / '%' / '（%）' 统一为 '%' """
            return s.replace("(%)", "%").replace("（%）", "%").replace("( % )", "%")

        fn_lower = field_name.lower()
        fn_normalized = _normalize_pct(fn_lower)

        # 第1层: 精确匹配
        for col in columns:
            if col.lower() == fn_lower and not _is_excluded(col):
                return col

        # 第1.5层: 百分号标准化后精确匹配
        for col in columns:
            if _normalize_pct(col.lower()) == fn_normalized and not _is_excluded(col):
                return col

        # 第2层: 子串匹配
        for col in columns:
            if _is_excluded(col):
                continue
            cl = col.lower()
            if fn_lower in cl or cl in fn_lower:
                return col

        # ★ FIX-3: 第2.5层: 中英文映射匹配
        for col in columns:
            if _is_excluded(col):
                continue
            if _cn_en_match(field_name, col):
                return col

        # ★ FIX-10: 第2.7层: 百分号标准化后的中英文映射匹配
        for col in columns:
            if _is_excluded(col):
                continue
            col_normalized = _normalize_pct(col.lower())
            if col_normalized != col.lower():
                # 尝试用标准化后的列名再走一遍 CN_EN 映射
                if _cn_en_match(field_name, col_normalized):
                    return col

        # ★ FIX-14: 第2.9层: 近义词模糊匹配 (复用 _fuzzy_field_match 的 synonym 逻辑)
        # 解决如 weighted_avg_cvr ↔ weighted_conversion_rate 等跨命名风格匹配
        non_excluded_cols = [c for c in columns if not _is_excluded(c)]
        if non_excluded_cols:
            fuzzy_matched = _fuzzy_field_match(field_name, non_excluded_cols)
            if fuzzy_matched is not None:
                return fuzzy_matched

        # 第3层: 类型推断 fallback
        if role == "group":
            for col in columns:
                if _is_excluded(col):
                    continue
                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    return col
        elif role == "value":
            if expect_numeric:
                for col in columns:
                    if _is_excluded(col):
                        continue
                    if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        return col
            else:
                found_first_obj = False
                for col in columns:
                    if _is_excluded(col):
                        continue
                    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                        if found_first_obj:
                            return col
                        found_first_obj = True

        return None

    def _calculate_error(self, expected: float, actual: float) -> float:
        if abs(expected) < 1e-10:
            return 0.0 if abs(actual) < 1e-10 else 1.0
        return abs(actual - expected) / abs(expected)


# ============================================================================
# 图表验证器
# ============================================================================

class ChartValidator:
    def validate(self, chart_paths: List[str], expected: Dict,
                 should_have_chart: bool = True, chart_optional: bool = False) -> Dict:
        result = {"has_chart": False, "valid": False, "score": 0.0, "details": "",
                  "checks": [], "unwanted_chart": False}
        has_chart = bool(chart_paths) and any(p and os.path.exists(p) for p in chart_paths)
        result["has_chart"] = has_chart

        if chart_optional:
            if not has_chart:
                result["score"] = 1.0
                result["valid"] = True
                result["checks"].append("✅ 图表可选，未生成不扣分")
                return result
            elif not should_have_chart:
                result["score"] = 1.0
                result["valid"] = True
                result["has_chart"] = True
                result["checks"].append("✅ 图表可选，额外生成不扣分")
                return result

        if not should_have_chart and has_chart:
            result["unwanted_chart"] = True
            result["score"] = 0.5
            result["checks"].append("⚠️ 不需要图表但生成了图表（误生成）")
            return result
        if not should_have_chart and not has_chart:
            result["score"] = 1.0
            result["valid"] = True
            result["checks"].append("✅ 正确：不需要图表且未生成")
            return result
        if should_have_chart and not has_chart:
            result["score"] = 0.0
            result["checks"].append("❌ 需要图表但未生成")
            return result

        chart_path = next((p for p in chart_paths if p and os.path.exists(p)), None)
        if not chart_path:
            result["checks"].append("❌ 图表文件不存在")
            return result

        result["checks"].append("✅ 图表文件存在")
        score = 0.5
        try:
            file_size = os.path.getsize(chart_path)
            if file_size < 1000:
                result["checks"].append(f"⚠️ 文件过小: {file_size} bytes")
                score -= 0.2
            else:
                result["checks"].append(f"✅ 文件大小正常: {file_size} bytes")
                score += 0.2
            if HAS_PIL and chart_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(chart_path) as img:
                    w, h = img.size
                    result["checks"].append(f"✅ 图片尺寸: {w}x{h}")
                    if w >= 400 and h >= 300:
                        score += 0.3
            result["valid"] = score >= 0.5
            result["score"] = min(1.0, max(0.0, score))
        except Exception as e:
            result["checks"].append(f"❌ 验证异常: {str(e)}")
        return result


# ============================================================================
# Schema 分析器
# ============================================================================

@dataclass
class SchemaInfo:
    table_name: str
    row_count: int
    metrics: List[str]
    dimensions: List[str]
    date_fields: List[str]


def analyze_schema_df(df: pd.DataFrame, source_name: str = "DataFrame") -> Optional[SchemaInfo]:
    try:
        metrics, dimensions, date_fields = [], [], []
        for col in df.columns:
            dtype = df[col].dtype
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['date', 'time']):
                date_fields.append(col)
            elif dtype in ['int64', 'float64', 'int32', 'float32']:
                metrics.append(col)
            else:
                dimensions.append(col)
        return SchemaInfo(source_name, len(df), metrics, dimensions, date_fields)
    except Exception as e:
        print(f"❌ DataFrame Schema 分析失败: {e}")
        return None


# ============================================================================
# 重试追踪器
# ============================================================================

@dataclass
class RetryRecord:
    attempt: int
    reason: str
    error_message: str = ""
    code_snippet: str = ""
    timestamp_ms: int = 0
    succeeded: bool = False


@dataclass
class RetryTrace:
    question_id: str
    total_attempts: int = 1
    retry_count: int = 0
    retries: List[RetryRecord] = field(default_factory=list)
    final_succeeded: bool = False
    retry_reasons: List[str] = field(default_factory=list)

    def add_retry(self, record: RetryRecord):
        self.retries.append(record)
        self.retry_count = len(self.retries)
        self.total_attempts = self.retry_count + 1
        if record.reason and record.reason not in self.retry_reasons:
            self.retry_reasons.append(record.reason)

    def to_dict(self) -> Dict:
        return {
            "question_id": self.question_id,
            "total_attempts": self.total_attempts,
            "retry_count": self.retry_count,
            "final_succeeded": self.final_succeeded,
            "retry_reasons": self.retry_reasons,
            "retries": [
                {
                    "attempt": r.attempt,
                    "reason": r.reason,
                    "error_message": r.error_message[:200],
                    "code_snippet": r.code_snippet[:200],
                    "timestamp_ms": r.timestamp_ms,
                    "succeeded": r.succeeded,
                }
                for r in self.retries
            ],
        }


class RetryTracker:
    RETRY_EVENT_TYPES = {
        "retry", "code_retry", "execution_retry", "auto_retry",
        "error_recovery", "rerun", "re_execute", "fallback_retry",
    }
    RETRY_STEP_KEYWORDS = {
        "retry", "attempt", "rerun", "re_execute", "fallback",
        "error_recovery", "second_try", "code_fix",
    }

    def extract_retry_trace(self, question_id, trace_log, exec_steps, final_succeeded):
        rt = RetryTrace(question_id=question_id, final_succeeded=final_succeeded)
        for event in trace_log.get("events", []):
            etype = event.get("event_type", "").lower()
            if self._is_retry_event(etype):
                meta = event.get("metadata", {})
                output = event.get("output_data", {})
                reason = (meta.get("retry_reason") or meta.get("error_type")
                          or output.get("retry_reason") or etype)
                error_msg = (meta.get("error_message") or meta.get("error")
                             or output.get("error", ""))
                code_snippet = (meta.get("code", "") or meta.get("failed_code", "")
                                or output.get("code", ""))
                record = RetryRecord(
                    attempt=len(rt.retries) + 2,
                    reason=str(reason), error_message=str(error_msg),
                    code_snippet=str(code_snippet)[:200],
                    timestamp_ms=event.get("timestamp_ms", 0),
                    succeeded=bool(output.get("succeeded", False)),
                )
                rt.add_retry(record)
        for step in trace_log.get("steps", []):
            step_name = step.get("step_name", "").lower()
            step_type = step.get("type", "").lower()
            is_retry_step = (
                any(kw in step_name for kw in self.RETRY_STEP_KEYWORDS)
                or any(kw in step_type for kw in self.RETRY_STEP_KEYWORDS)
                or step.get("is_retry", False)
            )
            if is_retry_step:
                attempt_num = step.get("attempt", len(rt.retries) + 2)
                if any(r.attempt == attempt_num for r in rt.retries):
                    continue
                reason = step.get("retry_reason", step.get("error_type", step_name))
                record = RetryRecord(
                    attempt=attempt_num, reason=str(reason),
                    error_message=str(step.get("error", ""))[:200],
                    code_snippet=str(step.get("code", ""))[:200],
                    timestamp_ms=step.get("timestamp_ms", 0),
                    succeeded=bool(step.get("succeeded", False)),
                )
                rt.add_retry(record)
        self._detect_implicit_retries(rt, exec_steps)
        return rt

    def _is_retry_event(self, event_type):
        return any(kw in event_type.lower() for kw in self.RETRY_EVENT_TYPES)

    def _detect_implicit_retries(self, rt, exec_steps):
        if len(exec_steps) < 2:
            return
        for i in range(1, len(exec_steps)):
            prev, curr = exec_steps[i - 1], exec_steps[i]
            prev_has_error = bool(prev.get("error") or prev.get("has_error"))
            curr_has_error = bool(curr.get("error") or curr.get("has_error"))
            same_type = prev.get("type", "") == curr.get("type", "")
            is_explicit_retry = curr.get("is_retry", False) or curr.get("retry", False)
            if (prev_has_error and same_type and not curr_has_error) or is_explicit_retry:
                attempt_num = i + 1
                if any(r.attempt == attempt_num for r in rt.retries):
                    continue
                reason = prev.get("error_type", "execution_error")
                error_msg = str(prev.get("error", ""))
                code_snippet = str(prev.get("code") or prev.get("sql", ""))[:200]
                record = RetryRecord(
                    attempt=attempt_num, reason=reason,
                    error_message=error_msg[:200], code_snippet=code_snippet,
                    timestamp_ms=curr.get("timestamp_ms", 0),
                    succeeded=not curr_has_error,
                )
                rt.add_retry(record)


def generate_retry_stats(retry_traces):
    total_questions = len(retry_traces)
    questions_with_retry = [rt for rt in retry_traces if rt.retry_count > 0]
    total_retries = sum(rt.retry_count for rt in retry_traces)
    reason_counter = {}
    for rt in retry_traces:
        for reason in rt.retry_reasons:
            reason_counter[reason] = reason_counter.get(reason, 0) + 1
    retry_success_count = sum(1 for rt in questions_with_retry if rt.final_succeeded)
    max_retries_question = max(retry_traces, key=lambda rt: rt.retry_count) if retry_traces else None
    return {
        "total_questions": total_questions,
        "questions_with_retry": len(questions_with_retry),
        "retry_rate_pct": len(questions_with_retry) / total_questions * 100 if total_questions > 0 else 0,
        "total_retries": total_retries,
        "avg_retries_per_question": total_retries / total_questions if total_questions > 0 else 0,
        "avg_retries_when_retried": (total_retries / len(questions_with_retry) if questions_with_retry else 0),
        "retry_recovery_rate_pct": (retry_success_count / len(questions_with_retry) * 100 if questions_with_retry else 0),
        "reason_distribution": dict(sorted(reason_counter.items(), key=lambda x: -x[1])),
        "max_retries": {
            "question_id": max_retries_question.question_id if max_retries_question else None,
            "count": max_retries_question.retry_count if max_retries_question else 0,
        },
    }


# ============================================================================
# Trace 分析器
# ============================================================================

class TraceAnalyzer:
    def __init__(self, trace_log, exec_result_steps=None):
        self.trace = trace_log or {}
        self.exec_steps = exec_result_steps or []

    def has_error(self):
        return self.trace.get("has_error", False) or bool(self.trace.get("errors"))

    def get_error_info(self):
        errors = self.trace.get("errors", [])
        return errors[0] if errors else None

    def get_detected_intent(self):
        return self.trace.get("detected_intent", "unknown")

    def get_route(self):
        return self.trace.get("route", "unknown")

    def get_output_type(self):
        for event in self.trace.get("events", []):
            if event.get("event_type") == "intent_classification":
                return event.get("output_data", {}).get("output_type", "TABLE")
        return "TABLE"

    def get_chart_type(self):
        for event in self.trace.get("events", []):
            if event.get("event_type") == "intent_classification":
                return event.get("output_data", {}).get("chart_type")
        return None

    def should_have_chart(self):
        return self.get_output_type() == "CHART" or self.get_chart_type() is not None

    def extract_codes(self):
        codes = []
        seen = set()
        for step in self.trace.get("steps", []):
            if "code" in step:
                code_str = step.get("code", "")
                if code_str and code_str not in seen:
                    seen.add(code_str)
                    codes.append({"step": step.get("step_name", "trace_step"),
                                  "code": code_str, "type": step.get("type", "unknown"),
                                  "source": "trace"})
        for step in self.exec_steps:
            code_str = step.get("code") or step.get("sql", "")
            if code_str and code_str not in seen:
                seen.add(code_str)
                codes.append({"step": step.get("type", "exec_step"), "code": code_str,
                              "type": step.get("type", "unknown"), "source": "exec_result"})
        return codes

    def extract_sql(self):
        sqls = []
        for step in self.trace.get("steps", []):
            if "sql" in step:
                sqls.append(step["sql"])
        for step in self.exec_steps:
            if step.get("sql"):
                sqls.append(step["sql"])
        return list(dict.fromkeys(sqls))

    def get_execution_type(self):
        for step in self.exec_steps:
            if step.get("type") == "python_execution":
                return "python"
            if step.get("type") == "sql_execution":
                return "sql"
        return "unknown"

    def get_skill_usage(self):
        # 优先从 skill_loaded 事件中提取（新版 trace）
        all_loaded = []
        skill_mode = "unknown"
        for event in self.trace.get("events", []):
            if event.get("event_type") == "skill_loaded":
                out = event.get("output_data", {})
                loaded = out.get("loaded_skills", [])
                all_loaded.extend(loaded)
                meta = event.get("metadata", {})
                skill_mode = meta.get("mode", skill_mode)
        if all_loaded:
            return {
                "mode": skill_mode,
                "loaded_skills": list(dict.fromkeys(all_loaded)),
                "skill_load_count": len(all_loaded),
            }
        # 兼容旧版：从 intent_classification 的 metadata 中提取
        for event in self.trace.get("events", []):
            if event.get("event_type") == "intent_classification":
                meta = event.get("metadata", {})
                skill_info = meta.get("skill_usage", {})
                if skill_info:
                    return skill_info
        return {"mode": "unknown", "loaded_skills": [], "skills_available": None}

    def get_execution_summary(self):
        skill = self.get_skill_usage()
        return {
            "intent": self.get_detected_intent(),
            "route": self.get_route(),
            "output_type": self.get_output_type(),
            "chart_type": self.get_chart_type(),
            "should_have_chart": self.should_have_chart(),
            "has_error": self.has_error(),
            "execution_type": self.get_execution_type(),
            "step_count": len(self.exec_steps),
            "code_count": len(self.extract_codes()),
            "sql_count": len(self.extract_sql()),
            "skill_mode": skill.get("mode", "unknown"),
            "loaded_skills": skill.get("loaded_skills", []),
            "skill_load_count": skill.get("skill_load_count", 0),
            "skills_available": skill.get("skills_available"),
        }


def format_codes(codes, qid, question):
    output = f"问题 {qid}: {question}\n"
    output += "=" * 80 + "\n\n"
    for i, code_info in enumerate(codes, 1):
        output += f"步骤 {i}: {code_info['step']} ({code_info.get('type', 'unknown')}) [来源: {code_info.get('source', '?')}]\n"
        output += "-" * 80 + "\n"
        output += code_info['code'] + "\n"
        output += "-" * 80 + "\n\n"
    return output


# ============================================================================
# 综合评估器
# ============================================================================

class Evaluator:
    def __init__(self, df=None, tolerance=0.01):
        self.tolerance = tolerance
        self.df = df
        self.answer_validator = AnswerValidator(tolerance)
        self.chart_validator = ChartValidator()

    def evaluate(self, question, analyzer, final_state):
        result = {
            "score": 0.0, "passed": False,
            "passed_criteria": [], "failed_criteria": [],
            "details": {}, "checks": [], "issues": []
        }
        has_error = analyzer.has_error() or final_state.get("error")
        if has_error:
            error_info = str(analyzer.get_error_info() or final_state.get("error", ""))
            result["failed_criteria"].append("execution_error")
            result["details"]["error"] = error_info
            result["checks"].append(f"❌ 执行错误: {error_info[:100]}")
            return result
        result["checks"].append("✅ 执行无错误")

        final_result = final_state.get("final_result")
        answer_result = self.answer_validator.validate(question, final_result)
        result["details"]["answer_validation"] = answer_result
        result["checks"].extend(answer_result.get("checks", []))
        answer_score = answer_result.get("score", 0)

        criteria = question.get("evaluation_criteria", {})
        chart_paths = final_state.get("chart_paths", [])
        should_have_chart = criteria.get("chart_generated", False)
        chart_optional = criteria.get("chart_optional", False) or "chart_generated" not in criteria
        expected_answer = question.get("expected_answer", {})
        chart_result = self.chart_validator.validate(
            chart_paths, expected_answer.get("chart_validation", {}),
            should_have_chart=should_have_chart, chart_optional=chart_optional
        )
        result["details"]["chart_validation"] = chart_result
        result["checks"].extend(chart_result.get("checks", []))
        if chart_result.get("unwanted_chart"):
            result["issues"].append({
                "type": "unwanted_chart", "severity": "warning",
                "message": "不应生成图表但生成了图表",
                "trace_output_type": analyzer.get_output_type()
            })
            result["failed_criteria"].append("unwanted_chart")
        chart_score = chart_result.get("score", 1.0)

        time_limit = criteria.get("response_time_limit_sec", 30)
        actual_time = final_state.get("total_time_ms", 0) / 1000
        if actual_time <= time_limit:
            result["passed_criteria"].append("response_time")
            result["checks"].append(f"✅ 响应时间: {actual_time:.1f}s")
            time_score = 1.0
        elif actual_time <= time_limit * 1.5:
            result["checks"].append(f"⚠️ 响应时间略超: {actual_time:.1f}s")
            time_score = 0.7
        else:
            result["failed_criteria"].append("response_time_exceeded")
            result["checks"].append(f"❌ 响应时间超限: {actual_time:.1f}s")
            time_score = 0.4

        if chart_optional:
            if chart_result.get("has_chart"):
                result["score"] = answer_score * 0.75 + chart_score * 0.15 + time_score * 0.1
            else:
                result["score"] = answer_score * 0.85 + time_score * 0.15
        elif should_have_chart:
            result["score"] = answer_score * 0.6 + chart_score * 0.3 + time_score * 0.1
        else:
            if chart_result.get("unwanted_chart"):
                result["score"] = answer_score * 0.7 + chart_score * 0.2 + time_score * 0.1
            else:
                result["score"] = answer_score * 0.85 + time_score * 0.15

        result["passed"] = result["score"] >= 0.8 and not result["issues"]
        if result["passed"]:
            result["passed_criteria"].append("overall_pass")
        return result

    def cleanup(self):
        pass


# ============================================================================
# Complex 前置能力组统计
# ============================================================================

def compute_prerequisite_stats(eval_data, results):
    metadata = eval_data.get("metadata", {})
    groups_def = metadata.get("complex_prerequisite_groups", {})
    if not groups_def:
        return {"groups": {}, "overall_ready": None, "summary": "评测集未定义前置能力组"}
    result_map = {r["question_id"]: r for r in results}
    groups_stats = {}
    all_meet = True
    for group_name, group_info in groups_def.items():
        qids = group_info.get("questions", [])
        required = group_info.get("required_pass_rate", 0.8)
        note = group_info.get("note", "")
        detail = []
        passed_count = 0
        for qid in qids:
            r = result_map.get(qid)
            if r is None:
                detail.append({"id": qid, "status": "missing", "score": 0})
            else:
                status = r.get("overall_status", "fail")
                score = r.get("criteria_score", 0)
                detail.append({"id": qid, "status": status, "score": score})
                if status == "pass":
                    passed_count += 1
        total = len(qids)
        pass_rate = passed_count / total if total > 0 else 0
        meets = pass_rate >= required
        if not meets:
            all_meet = False
        groups_stats[group_name] = {
            "questions": qids, "required_pass_rate": required, "note": note,
            "total": total, "passed": passed_count,
            "pass_rate": round(pass_rate, 4), "meets_threshold": meets, "detail": detail,
        }
    met_count = sum(1 for g in groups_stats.values() if g["meets_threshold"])
    total_groups = len(groups_stats)
    return {
        "groups": groups_stats, "overall_ready": all_meet,
        "summary": f"{met_count}/{total_groups} 组达标"
                   + (" → ✅ 可进入 Complex 评测" if all_meet
                      else " → ❌ 尚未满足 Complex 前置条件"),
    }


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(results, eval_data, schema, retry_stats=None, retry_traces=None,
                    prerequisite_stats=None):
    total = len(results)
    passed = sum(1 for r in results if r["overall_status"] == "pass")
    partial = sum(1 for r in results if r["overall_status"] == "partial")
    failed = sum(1 for r in results if r["overall_status"] == "fail")
    avg_score = sum(r.get("criteria_score", 0) for r in results) / total if total > 0 else 0

    issues_by_type = {}
    for r in results:
        for issue in r.get("criteria_checks", {}).get("issues", []):
            issue_type = issue.get("type", "unknown")
            issues_by_type.setdefault(issue_type, []).append(r["question_id"])

    report = f"""# Insight Agent 评测报告 v9.2 (CSV-Only, Fixed)

## 概览
- **评测时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **数据源**: CSV/Excel
- **问题总数**: {total}
- **通过率**: {passed}/{total} ({passed/total*100:.1f}%)
- **部分通过**: {partial}
- **失败**: {failed}
- **平均得分**: {avg_score:.1%}

"""

    if retry_stats:
        report += "## 🔄 重试统计\n\n"
        report += f"- **触发重试的题目数**: {retry_stats['questions_with_retry']}/{retry_stats['total_questions']} "
        report += f"({retry_stats['retry_rate_pct']:.1f}%)\n"
        report += f"- **总重试次数**: {retry_stats['total_retries']}\n"
        report += f"- **平均重试次数（全部题目）**: {retry_stats['avg_retries_per_question']:.2f}\n"
        report += f"- **平均重试次数（仅重试题）**: {retry_stats['avg_retries_when_retried']:.2f}\n"
        report += f"- **重试恢复率**: {retry_stats['retry_recovery_rate_pct']:.1f}%\n"
        if retry_stats.get("max_retries", {}).get("count", 0) > 0:
            report += f"- **最多重试**: {retry_stats['max_retries']['question_id']} "
            report += f"({retry_stats['max_retries']['count']} 次)\n"
        if retry_stats.get("reason_distribution"):
            report += "\n### 重试原因分布\n\n"
            report += "| 原因 | 次数 |\n|------|------|\n"
            for reason, count in retry_stats["reason_distribution"].items():
                report += f"| {reason} | {count} |\n"
        report += "\n"

    if retry_traces:
        retried = [rt for rt in retry_traces if rt.retry_count > 0]
        if retried:
            report += "### 重试明细\n\n"
            report += "| 题目ID | 重试次数 | 最终成功 | 重试原因 |\n"
            report += "|--------|----------|----------|----------|\n"
            for rt in sorted(retried, key=lambda x: -x.retry_count):
                reasons_str = ", ".join(rt.retry_reasons) if rt.retry_reasons else "-"
                success_emoji = "✅" if rt.final_succeeded else "❌"
                report += f"| {rt.question_id} | {rt.retry_count} | {success_emoji} | {reasons_str} |\n"
            report += "\n"

    if prerequisite_stats and prerequisite_stats.get("groups"):
        report += "## 🎯 Complex 前置能力组评估\n\n"
        report += f"**综合判定**: {prerequisite_stats['summary']}\n\n"
        report += "| 能力组 | 题数 | 通过 | 通过率 | 要求 | 达标 |\n"
        report += "|--------|------|------|--------|------|------|\n"
        for group_name, gs in prerequisite_stats["groups"].items():
            met_emoji = "✅" if gs["meets_threshold"] else "❌"
            report += (f"| {group_name} | {gs['total']} | {gs['passed']} "
                       f"| {gs['pass_rate']*100:.0f}% | ≥{gs['required_pass_rate']*100:.0f}% "
                       f"| {met_emoji} |\n")
        report += "\n"
        for group_name, gs in prerequisite_stats["groups"].items():
            met_emoji = "✅" if gs["meets_threshold"] else "❌"
            report += f"### {met_emoji} {group_name} ({gs['passed']}/{gs['total']})\n\n"
            if gs.get("note"):
                report += f"> {gs['note']}\n\n"
            for d in gs["detail"]:
                s_emoji = {"pass": "✅", "partial": "⚠️", "fail": "❌", "missing": "❓"}.get(d["status"], "❓")
                report += f"- {s_emoji} `{d['id']}` — {d['status']} ({d['score']:.0%})\n"
            report += "\n"

    if schema:
        report += "## 数据源信息\n\n"
        report += f"- **来源**: {schema.table_name}\n"
        report += f"- **行数**: {schema.row_count}\n"
        report += f"- **指标列**: {', '.join(schema.metrics)}\n"
        report += f"- **维度列**: {', '.join(schema.dimensions)}\n"
        report += f"- **日期列**: {', '.join(schema.date_fields)}\n\n"

    if issues_by_type:
        report += "## 发现的问题\n\n"
        for issue_type, qids in issues_by_type.items():
            report += f"- **{issue_type}**: {', '.join(qids)}\n"
        report += "\n"

    skill_stats = {"skill": 0, "fallback": 0, "keyword_fallback": 0, "unknown": 0}
    skill_file_freq = {}
    for r in results:
        su = r.get("skill_usage", {})
        mode = su.get("mode", "unknown")
        skill_stats[mode] = skill_stats.get(mode, 0) + 1
        for sf in su.get("loaded_skills", []):
            skill_file_freq[sf] = skill_file_freq.get(sf, 0) + 1
    has_skill_data = skill_stats.get("skill", 0) > 0 or skill_stats.get("fallback", 0) > 0
    if has_skill_data:
        report += "## Skill 使用统计\n\n### 路由模式分布\n\n"
        report += "| 模式 | 次数 | 占比 |\n|------|------|------|\n"
        for mode, cnt in sorted(skill_stats.items(), key=lambda x: -x[1]):
            if cnt > 0:
                report += f"| {mode} | {cnt} | {cnt/total*100:.0f}% |\n"
        report += "\n"
        if skill_file_freq:
            report += "### Skill 文件加载频率\n\n"
            report += "| Skill 文件 | 加载次数 | 占比 |\n|------------|----------|------|\n"
            for sf, cnt in sorted(skill_file_freq.items(), key=lambda x: -x[1]):
                report += f"| {sf} | {cnt} | {cnt/total*100:.0f}% |\n"
            report += "\n"

    retry_map = {}
    if retry_traces:
        retry_map = {rt.question_id: rt for rt in retry_traces}
    report += "## 逐题结果\n\n"
    report += "| ID | 难度 | 类别 | 状态 | 得分 | 耗时(ms) | 执行器 | 重试次数 |\n"
    report += "|----|------|------|------|------|----------|--------|----------|\n"
    for r in results:
        emoji = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(r["overall_status"], "❓")
        rt = retry_map.get(r["question_id"])
        retry_cnt = rt.retry_count if rt else 0
        retry_display = f"{retry_cnt}" if retry_cnt == 0 else f"🔄 {retry_cnt}"
        report += (f"| {r['question_id']} | {r['difficulty']} "
                   f"| {r['category'][:25]} | {emoji} "
                   f"| {r['criteria_score']:.0%} | {r['execution_time_ms']} "
                   f"| {r.get('execution_type', '?')} | {retry_display} |\n")
    report += f"\n---\n_生成时间: {datetime.now().isoformat()}_\n"
    return report


# ============================================================================
# 主执行函数
# ============================================================================

def load_agent_module(module_path):
    module_path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location("agent_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)
    return module


def run_evaluation(agent_module, questions_file, data_file="", data_source="csv",
                   output_dir="./eval_output", tolerance=0.01):
    with open(questions_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    questions = eval_data.get("questions", [])
    metadata = eval_data.get("metadata", {})
    tolerance = metadata.get("answer_tolerance", tolerance)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"eval_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    df_file = None
    if data_file:
        print(f"\n   📂 数据文件: {data_file}")
        if data_file.endswith(('.xlsx', '.xls')):
            df_file = pd.read_excel(data_file)
        else:
            df_file = pd.read_csv(data_file)
        print(f"   数据行数: {len(df_file)}")
    else:
        print("   ⚠️ 未指定 --data-file，评测可能无法正常运行")

    DataSourceType = getattr(agent_module, 'DataSourceType', None)
    print(f"\n{'='*60}")
    print(f"🚀 开始评测: {len(questions)} 个问题")
    print(f"   数据源: {data_source.upper()}")
    print(f"   容差: {tolerance:.1%}")
    print(f"{'='*60}")

    orchestrator = agent_module.AgentOrchestrator()
    evaluator = Evaluator(df=df_file, tolerance=tolerance)
    retry_tracker = RetryTracker()
    all_results = []
    all_retry_traces = []

    for q in questions:
        qid = q.get("id", "?")
        question_text = q.get("question", "")
        q_source = q.get("data_source", data_source)
        print(f"\n[{qid}] [{q_source.upper():8s}] {question_text[:50]}...")
        start = time.time()
        result = {
            "question_id": qid, "question": question_text,
            "difficulty": q.get("difficulty", 0), "category": q.get("category", ""),
            "question_data_source": q_source, "route": "", "execution_type": "unknown",
            "overall_status": "error", "execution_time_ms": 0,
            "criteria_score": 0, "criteria_checks": {}
        }
        try:
            if df_file is None:
                raise ValueError(f"题目 {qid} 需要数据文件，但未提供 --data-file")
            ds_enum = None
            if DataSourceType:
                if q_source == "csv":
                    ds_enum = DataSourceType.CSV
                elif q_source == "excel":
                    ds_enum = DataSourceType.EXCEL
            exec_result = orchestrator.process(question_text, data_source=ds_enum, uploaded_df=df_file)
            final_state = {
                "route": exec_result.get("route", ""),
                "total_time_ms": exec_result.get("total_time_ms", 0),
                "steps": exec_result.get("steps", []),
                "final_result": exec_result.get("final_result"),
                "final_report": exec_result.get("final_report"),
                "chart_paths": exec_result.get("chart_paths", []),
                "error": exec_result.get("error"),
                "trace_log": exec_result.get("trace_log", {})
            }
            trace_dict = exec_result.get("trace_log", {})
            exec_steps = exec_result.get("steps", [])
            result["execution_time_ms"] = int((time.time() - start) * 1000)
            result["route"] = final_state.get("route", "")
            analyzer = TraceAnalyzer(trace_dict, exec_steps)
            result["execution_type"] = analyzer.get_execution_type()
            summary = analyzer.get_execution_summary()
            print(f"  📋 Trace: output_type={summary['output_type']}, "
                  f"chart_type={summary['chart_type']}, exec_type={summary['execution_type']}")
            skill_mode = summary.get("skill_mode", "unknown")
            loaded_skills = summary.get("loaded_skills", [])
            skill_usage = analyzer.get_skill_usage()
            skill_load_count = skill_usage.get("skill_load_count", 0)
            if loaded_skills:
                skill_detail = f"mode={skill_mode}, loaded={loaded_skills}"
                if skill_load_count > 1:
                    skill_detail += f", load_events={skill_load_count}"
                print(f"  🧩 Skills: {skill_detail}")
            has_error = analyzer.has_error() or final_state.get("error")
            eval_result = evaluator.evaluate(q, analyzer, final_state)
            final_succeeded = eval_result.get("score", 0) >= 0.5 and not has_error
            retry_trace = retry_tracker.extract_retry_trace(
                question_id=qid, trace_log=trace_dict,
                exec_steps=exec_steps, final_succeeded=final_succeeded,
            )
            all_retry_traces.append(retry_trace)
            if retry_trace.retry_count > 0:
                print(f"  🔄 重试: {retry_trace.retry_count} 次, "
                      f"原因: {retry_trace.retry_reasons}, "
                      f"最终{'成功' if retry_trace.final_succeeded else '失败'}")
            with open(output_path / f"trace_{qid}.json", 'w', encoding='utf-8') as f:
                json.dump(trace_dict, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
            codes = analyzer.extract_codes()
            with open(output_path / f"codes_{qid}.txt", 'w', encoding='utf-8') as f:
                f.write(format_codes(codes, qid, question_text))
            chart_paths = final_state.get("chart_paths", [])
            saved_charts = []
            if chart_paths:
                charts_dir = output_path / "charts"
                charts_dir.mkdir(exist_ok=True)
                for i, chart_path in enumerate(chart_paths):
                    if chart_path and os.path.exists(chart_path):
                        ext = os.path.splitext(chart_path)[1] or '.png'
                        dest = charts_dir / f"chart_{qid}_{i}{ext}"
                        try:
                            shutil.copy(chart_path, dest)
                            saved_charts.append(str(dest))
                            print(f"  📊 保存图表: {dest.name}")
                        except Exception as e:
                            print(f"  ⚠️ 图表保存失败: {e}")
            result["saved_charts"] = saved_charts
            score = eval_result.get("score", 0)
            has_issues = bool(eval_result.get("issues"))
            if score >= 0.8 and not has_error and not has_issues:
                status = "pass"
            elif score >= 0.5:
                status = "partial"
            else:
                status = "fail"
            result.update({
                "overall_status": status, "criteria_score": score,
                "criteria_checks": eval_result,
                "detected_intent": analyzer.get_detected_intent(),
                "skill_usage": analyzer.get_skill_usage(),
                "retry_trace": retry_trace.to_dict(),
            })
            emoji = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(status, "❓")
            print(f"  {emoji} {status} | 路径: {result['route']} | 执行器: {result['execution_type']} "
                  f"| 评分: {score:.0%} | {result['execution_time_ms']}ms")
            for check in eval_result.get("checks", [])[:3]:
                print(f"     {check}")
            for issue in eval_result.get("issues", []):
                print(f"  🔴 问题: {issue.get('message', '')}")
            if has_error:
                error_info = analyzer.get_error_info() or final_state.get("error", "")
                print(f"  ⚠️ 错误: {str(error_info)[:80]}")
        except Exception as e:
            result["execution_time_ms"] = int((time.time() - start) * 1000)
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"  💥 异常: {str(e)[:80]}")
            traceback.print_exc()
            all_retry_traces.append(RetryTrace(question_id=qid, final_succeeded=False))
        all_results.append(result)

    evaluator.cleanup()
    retry_stats = generate_retry_stats(all_retry_traces)
    prerequisite_stats = compute_prerequisite_stats(eval_data, all_results)

    with open(output_path / "results.json", 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "eval_version": "v9.4-csv-only-answer-multirow-comparison-list",
            "data_source": data_source, "tolerance": tolerance,
            "summary": {
                "total": len(all_results),
                "passed": sum(1 for r in all_results if r["overall_status"] == "pass"),
                "partial": sum(1 for r in all_results if r["overall_status"] == "partial"),
                "failed": sum(1 for r in all_results if r["overall_status"] == "fail"),
                "avg_score": sum(r.get("criteria_score", 0) for r in all_results) / len(all_results) if all_results else 0,
                "by_execution_type": {
                    et: len([r for r in all_results if r.get("execution_type") == et])
                    for et in set(r.get("execution_type", "unknown") for r in all_results)
                },
                "issues": {
                    "unwanted_chart": [r["question_id"] for r in all_results
                                       if any(i.get("type") == "unwanted_chart"
                                              for i in r.get("criteria_checks", {}).get("issues", []))],
                },
            },
            "prerequisite_stats": prerequisite_stats,
            "retry_stats": retry_stats,
            "retry_traces": [rt.to_dict() for rt in all_retry_traces],
            "results": all_results
        }, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)

    with open(output_path / "retry_traces.json", 'w', encoding='utf-8') as f:
        json.dump({"stats": retry_stats, "traces": [rt.to_dict() for rt in all_retry_traces]},
                  f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)

    schema = analyze_schema_df(df_file, data_file) if df_file is not None else None
    report = generate_report(all_results, eval_data, schema, retry_stats, all_retry_traces,
                             prerequisite_stats=prerequisite_stats)
    with open(output_path / "report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    passed = sum(1 for r in all_results if r["overall_status"] == "pass")
    avg_score = sum(r.get("criteria_score", 0) for r in all_results) / len(all_results) if all_results else 0
    print(f"\n{'='*60}")
    print(f"✅ 评测完成")
    print(f"   通过率: {passed}/{len(all_results)} ({passed/len(all_results)*100:.0f}%)")
    print(f"   平均得分: {avg_score:.1%}")
    by_exec = {}
    for r in all_results:
        et = r.get("execution_type", "unknown")
        by_exec.setdefault(et, []).append(r)
    print(f"   执行器分布: { {k: len(v) for k, v in by_exec.items()} }")
    if prerequisite_stats.get("groups"):
        print(f"\n   🎯 Complex 前置能力组:")
        for gname, gs in prerequisite_stats["groups"].items():
            met_emoji = "✅" if gs["meets_threshold"] else "❌"
            print(f"      {met_emoji} {gname}: {gs['passed']}/{gs['total']} "
                  f"({gs['pass_rate']*100:.0f}%, 要求≥{gs['required_pass_rate']*100:.0f}%)")
        ready_emoji = "✅" if prerequisite_stats["overall_ready"] else "❌"
        print(f"      {'-'*40}")
        print(f"      {ready_emoji} {prerequisite_stats['summary']}")
    if retry_stats["questions_with_retry"] > 0:
        print(f"\n   🔄 重试汇总:")
        print(f"      触发重试: {retry_stats['questions_with_retry']}/{retry_stats['total_questions']} 题")
        print(f"      总重试次数: {retry_stats['total_retries']}")
        print(f"      重试恢复率: {retry_stats['retry_recovery_rate_pct']:.1f}%")
        if retry_stats.get("reason_distribution"):
            top_reasons = list(retry_stats["reason_distribution"].items())[:3]
            reasons_str = ", ".join(f"{r}({c})" for r, c in top_reasons)
            print(f"      主要原因: {reasons_str}")
    else:
        print(f"\n   🔄 重试: 无题目触发重试")
    unwanted = [r["question_id"] for r in all_results
                if any(i.get("type") == "unwanted_chart" for i in r.get("criteria_checks", {}).get("issues", []))]
    if unwanted:
        print(f"   ⚠️ 图表误生成: {unwanted}")
    print(f"   📝 输出: {output_path}")
    print(f"{'='*60}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Insight Agent 评测 v9.4 (CSV-Only, Answer-First Multi-Row, Comparison List Fix)")
    parser.add_argument("--app", default="app_v4_5_2_chart_logic_fixed.py", help="Agent 脚本路径")
    parser.add_argument("--questions", default="eval-csv-only.json", help="问题集 JSON")
    parser.add_argument("--data-file", default="", help="CSV/Excel 数据文件路径（必需）")
    parser.add_argument("--data-source", default="csv", choices=["csv", "excel"],
                        help="数据源类型（默认 csv）")
    parser.add_argument("--output", default="./eval_output", help="输出目录")
    parser.add_argument("--tolerance", type=float, default=0.01, help="数值容差")
    args = parser.parse_args()
    if not os.path.exists(args.app):
        print(f"❌ Agent 文件不存在: {args.app}")
        sys.exit(1)
    if not os.path.exists(args.questions):
        print(f"❌ 问题文件不存在: {args.questions}")
        sys.exit(1)
    if not args.data_file:
        print(f"❌ CSV-Only 模式需要指定 --data-file")
        sys.exit(1)
    if not os.path.exists(args.data_file):
        print(f"❌ 数据文件不存在: {args.data_file}")
        sys.exit(1)
    agent_module = load_agent_module(args.app)
    run_evaluation(agent_module, args.questions, data_file=args.data_file,
                   data_source=args.data_source, output_dir=args.output, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
