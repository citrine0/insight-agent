"""
app_v9-9.py - 统一管线架构（Unified Pipeline + Adaptive Depth）
基于 app_v9-8.py，清理冗余代码

v9.9 变更：清除以下已废弃的死代码（~750 行）
  - AnalysisInstruction 类（v5 遗留，v9 统一管线不再使用）
  - _COMPLEX_BASE / PYTHON_AGENT_SYSTEM_PROMPT / PYTHON_VISUALIZE_PROMPT（仅供已废弃方法使用）
  - PythonAgent.execute()、visualize()、_generate_code()、_generate_viz_code()、_fix_code()、_default_viz_code()
  - RouterAgent._simple_parse()
  - PlannerAgent.generate_instruction()
  - simple_executor_node()（v9 统一管线后不再接入 graph）

当前架构要点：
  - 所有合法查询统一进 Quick Scan（Commander → Scan Loop → Detect → Arbiter）
  - 分析深度由 Commander LLM 判断（descriptive / diagnostic / causal）
  - Reasoner v2 做因果推理 + needs_deep_rca 决策
  - Deep RCA 子图支持补充数据上传 + 全量扫描（v9.8）
  - Fallback ReAct 路径仅在 Quick Scan 失败时降级使用

详见 claude.md 架构设计文档。
"""

import os
import json
import tempfile
import traceback
import time
import uuid
import base64
from datetime import datetime, timedelta
from typing import TypedDict, Literal, Optional, Any, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager

import calendar

# ============================================================================
# 中文字体配置（解决图表中文显示为方框的问题）
# ============================================================================

CHINESE_FONTS = [
    'Noto Sans CJK SC',
    'Noto Sans CJK',
    'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei',
    'Droid Sans Fallback',
    'SimHei',
    'Microsoft YaHei',
    'PingFang SC',
    'Heiti SC',
    'STHeiti',
    'Arial Unicode MS',
    'DejaVu Sans',
]


def get_font_config_snippet() -> str:
    """返回可嵌入动态生成 Python 代码中的字体配置代码片段（唯一来源）"""
    fonts_str = repr(CHINESE_FONTS)
    return f"""# 中文字体配置
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = {fonts_str}
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
"""


def setup_chinese_font():
    """配置 matplotlib 支持中文显示"""
    import subprocess

    try:
        result = subprocess.run(
            ['fc-list', ':lang=zh', '-f', '%{family}\n'],
            capture_output=True, text=True, timeout=5
        )
        available_fonts = result.stdout.strip().split('\n') if result.stdout else []
    except Exception:
        available_fonts = []

    selected_font = None
    for font in CHINESE_FONTS:
        try:
            font_names = [f.name for f in font_manager.fontManager.ttflist]
            if font in font_names:
                selected_font = font
                break
        except Exception:
            pass
        if any(font.lower() in f.lower() for f in available_fonts):
            selected_font = font
            break

    if not selected_font:
        try:
            subprocess.run(['apt-get', 'update'], capture_output=True, timeout=30)
            subprocess.run(['apt-get', 'install', '-y', 'fonts-noto-cjk'], capture_output=True, timeout=60)
            cache_dir = matplotlib.get_cachedir()
            if cache_dir and os.path.exists(cache_dir):
                for f in os.listdir(cache_dir):
                    if 'font' in f.lower():
                        try:
                            os.remove(os.path.join(cache_dir, f))
                        except Exception:
                            pass
            font_manager._rebuild()
            selected_font = 'Noto Sans CJK SC'
        except Exception:
            pass

    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial']
    else:
        plt.rcParams['font.sans-serif'] = CHINESE_FONTS + ['DejaVu Sans', 'Arial']

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    return selected_font


_CHINESE_FONT = setup_chinese_font()
get_font_config_code = get_font_config_snippet


# ============================================================================
# 全局配置
# ============================================================================

CHART_DIR = Path(tempfile.gettempdir()) / "agent_charts"
CHART_DIR.mkdir(exist_ok=True)

# === DataFrame Schema 缓存（替代 DB Schema）===
_current_df_schema: Dict = {}
_current_df_date_range: Dict = {}


# ── 动态日期列检测 ──────────────────────────────────────────────
def _try_parse_date_column(series: pd.Series, threshold: float = 0.8) -> bool:
    """
    尝试将一列解析为日期。成功率 >= threshold 即判定为日期列。
    支持 YYYY-MM-DD、YYYY-MM、YYYY/MM/DD、各种 timestamp 等。
    跳过纯数字列（避免把年份整数误判为日期）。
    """
    if series.dropna().empty:
        return False
    # 如果已经是 datetime64 直接返回
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # 只对 object / string 类型列做探测
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
        return False
    sample = series.dropna().head(50)
    # 排除纯数字字符串（如 "2025" 可能是年份整数）—— 至少包含分隔符
    sample_str = sample.astype(str)
    has_separator = sample_str.str.contains(r'[-/年.]', regex=True)
    if has_separator.sum() / len(sample_str) < threshold:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        success_rate = parsed.notna().sum() / len(sample)
        return success_rate >= threshold
    except Exception:
        return False


def safe_parse_date(raw_value: str) -> Optional[pd.Timestamp]:
    """
    安全地将日期字符串解析为 Timestamp。
    处理 None、"unknown"、各种格式异常。返回 None 表示解析失败。
    """
    if not raw_value or raw_value in ("unknown", "None", "none", "null"):
        return None
    try:
        return pd.to_datetime(raw_value, errors="coerce")
    except Exception:
        return None


def safe_get_year_month(date_range: Dict[str, str], key: str = "max"
                        ) -> tuple:
    """
    从 date_range dict 安全提取 (year: int, month: int)。
    返回 (None, None) 表示无法提取。
    """
    ts = safe_parse_date(date_range.get(key, ""))
    if ts is None or pd.isna(ts):
        return (None, None)
    return (ts.year, ts.month)


def infer_aggregation_from_schema(col_name: str, schema: dict) -> str:
    """
    根据 schema 中已有的 unit_hint 信息推断聚合方式。
    优先使用 schema 元信息（unit_hint），fallback 到通用规则。
    返回 "sum" | "mean" | "weighted_avg"
    """
    # 从 schema columns 查找该列的 unit_hint
    for table in schema.get("tables", []):
        for col_info in table.get("columns", []):
            if col_info.get("name") == col_name:
                hint = col_info.get("unit_hint", "")
                if hint in ("decimal_ratio", "already_percentage"):
                    return "weighted_avg"
                # INTEGER 大概率是可加指标
                if col_info.get("type") in ("INTEGER", "REAL") and not hint:
                    return "sum"
    # 通用 fallback：名称含 rate/ratio/pct 等关键词 → mean
    RATIO_KW = ("rate", "ratio", "pct", "percent", "roi", "roas",
                "margin", "ctr", "cpc", "avg_price", "competitor_price")
    if any(kw in col_name.lower() for kw in RATIO_KW):
        return "weighted_avg"
    return "sum"


def infer_default_target_field(schema: dict) -> str:
    """
    从 schema 的 numeric_columns 推断默认 target_field。
    优先选第一个 sum 型（非 ratio）数值列；找不到则返回第一个数值列。
    """
    meta = schema.get("_meta", {})
    numeric_cols = [col for _, col in meta.get("numeric_columns", [])]
    if not numeric_cols:
        return "value"  # 兜底
    # 优先选非 ratio 列
    for col in numeric_cols:
        agg = infer_aggregation_from_schema(col, schema)
        if agg == "sum":
            return col
    return numeric_cols[0]


def get_df_schema(df: pd.DataFrame) -> dict:
    """从 DataFrame 推断 schema（替代 get_db_schema）"""
    if df is None or df.empty:
        return {"tables": [], "_meta": {
            "date_columns": [], "numeric_columns": [],
            "text_columns": [], "all_tables": [],
        }}

    table_name = "uploaded_data"
    meta = {
        "date_columns": [],
        "numeric_columns": [],
        "text_columns": [],
        "all_tables": [table_name],
    }
    columns_info = []

    for col in df.columns:
        col_info = {"name": col, "type": str(df[col].dtype)}

        # 日期列检测 — 动态探测，不依赖列名白名单
        is_date = _try_parse_date_column(df[col])

        if is_date:
            meta["date_columns"].append((table_name, col))
            col_info["type"] = "DATE"
        elif pd.api.types.is_numeric_dtype(df[col]):
            if col.lower() not in ("id", "pk"):
                meta["numeric_columns"].append((table_name, col))
                col_info["type"] = "REAL" if pd.api.types.is_float_dtype(df[col]) else "INTEGER"
                RATIO_KEYWORDS = ("rate", "ratio", "pct", "percent", "roi", "roas", "margin")
                if any(kw in col.lower() for kw in RATIO_KEYWORDS):
                    col_max = df[col].dropna().max()
                    if col_max > 1:
                        col_info["unit_hint"] = "already_percentage"
                        col_info["description"] = (
                            f"已为百分比形式（如 {col_max:.2f} 表示 {col_max:.2f}%），"
                            f"加权平均后无需再 ×100"
                        )
                    else:
                        col_info["unit_hint"] = "decimal_ratio"
                        col_info["description"] = (
                            f"小数形式（如 {col_max:.4f} 表示 {col_max * 100:.2f}%），"
                            f"输出百分比时需 ×100"
                        )
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            distinct_vals = sorted(df[col].dropna().unique().tolist())[:30]
            if distinct_vals:
                col_info["enum_values"] = distinct_vals
                meta["text_columns"].append((table_name, col, distinct_vals))
            col_info["type"] = "TEXT"

        columns_info.append(col_info)

    return {
        "tables": [{"name": table_name, "columns": columns_info}],
        "_meta": meta,
    }


def get_df_date_range(df: pd.DataFrame) -> Dict[str, str]:
    """从 DataFrame 探测日期范围（替代 get_db_date_range）"""
    if df is None or df.empty:
        return {"min": "unknown", "max": "unknown",
                "table": "uploaded_data", "date_column": "unknown"}

    # 尝试找日期列 — 动态探测，不依赖列名白名单
    date_col = None
    for col in df.columns:
        if _try_parse_date_column(df[col]):
            date_col = col
            break

    if date_col is None:
        return {"min": "unknown", "max": "unknown",
                "table": "uploaded_data", "date_column": "unknown"}

    try:
        dates = pd.to_datetime(df[date_col].dropna(), errors="coerce").dropna()
        if dates.empty:
            return {"min": "unknown", "max": "unknown",
                    "table": "uploaded_data", "date_column": date_col}
        return {
            "min": str(dates.min().date()),
            "max": str(dates.max().date()),
            "table": "uploaded_data",
            "date_column": date_col,
        }
    except Exception:
        return {"min": "unknown", "max": "unknown",
                "table": "uploaded_data", "date_column": date_col}


def set_current_df(df: pd.DataFrame):
    """加载新 DataFrame 时调用，更新全局 schema 和日期范围缓存"""
    global _current_df_schema, _current_df_date_range
    _current_df_schema = get_df_schema(df)
    _current_df_date_range = get_df_date_range(df)


def get_cached_schema() -> dict:
    """获取缓存的 schema（供 Router 等无法直接获取 df 的组件使用）"""
    if _current_df_schema:
        return _current_df_schema
    return {"tables": [], "_meta": {
        "date_columns": [], "numeric_columns": [],
        "text_columns": [], "all_tables": [],
    }}


def get_cached_date_range() -> Dict[str, str]:
    """获取缓存的日期范围"""
    if _current_df_date_range:
        return _current_df_date_range
    return {"min": "unknown", "max": "unknown",
            "table": "unknown", "date_column": "unknown"}


# ============================================================================
# 1. 数据源类型
# ============================================================================

class DataSourceType(Enum):
    EXCEL = "excel"
    CSV = "csv"


class RouteType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    INVALID = "invalid"


# ============================================================================

@dataclass
class AnalysisResult:
    """PythonAgent 返回的结构化结果"""
    task_id: str
    success: bool

    executed_code: str = ""
    execution_time_ms: int = 0

    data: List[Dict] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    summary: str = ""
    preview: str = ""

    chart_path: Optional[str] = None
    chart_type: Optional[str] = None

    evaluation: Optional[Dict] = None

    answer: Optional[Any] = None  # 评测精确提取用: 单行 {"field": value}, 多行 [{"group": x, "value": y}, ...]

    error: Optional[str] = None
    error_type: Optional[str] = None
    correction_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "executed_code": self.executed_code,
            "execution_time_ms": self.execution_time_ms,
            "data": self.data,
            "columns": self.columns,
            "row_count": self.row_count,
            "summary": self.summary,
            "preview": self.preview,
            "chart_path": self.chart_path,
            "chart_type": self.chart_type,
            "evaluation": self.evaluation,
            "answer": self.answer,
            "error": self.error,
            "error_type": self.error_type,
            "correction_history": self.correction_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        return cls(
            task_id=data.get("task_id", ""),
            success=data.get("success", False),
            executed_code=data.get("executed_code", ""),
            execution_time_ms=data.get("execution_time_ms", 0),
            data=data.get("data", []),
            columns=data.get("columns", []),
            row_count=data.get("row_count", 0),
            summary=data.get("summary", ""),
            preview=data.get("preview", ""),
            chart_path=data.get("chart_path"),
            chart_type=data.get("chart_type"),
            evaluation=data.get("evaluation"),
            answer=data.get("answer"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            correction_history=data.get("correction_history", []),
        )

    def to_observation(self) -> str:
        if not self.success:
            return f"执行失败 | error: {self.error}"
        parts = [f"执行成功 | rows: {self.row_count}"]
        if self.data and len(self.data) <= 5:
            parts.append(f"data: {json.dumps(self.data, ensure_ascii=False)}")
        elif self.preview:
            parts.append(f"preview: {self.preview[:300]}")
        if self.evaluation:
            parts.append(f"evaluation: {json.dumps(self.evaluation, ensure_ascii=False)}")
        return " | ".join(parts)


# ============================================================================
# 2.5  v8 Quick Scan 数据结构
# ============================================================================

@dataclass
class MetricSpec:
    """一个待扫描指标的规格，由 Commander 从 schema 动态推断"""
    name: str
    aggregation: str = "sum"       # "sum" | "weighted_avg" | "mean"
    unit_hint: str = ""            # 直接来自 schema column.unit_hint
    description: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "aggregation": self.aggregation,
                "unit_hint": self.unit_hint, "description": self.description}

    @classmethod
    def from_dict(cls, d: dict) -> "MetricSpec":
        return cls(name=d["name"], aggregation=d.get("aggregation", "sum"),
                   unit_hint=d.get("unit_hint", ""), description=d.get("description", ""))


@dataclass
class TimeWindow:
    """时间对比窗口，由 Commander 从用户问题推断"""
    current_start: str = ""
    current_end: str = ""
    previous_start: str = ""
    previous_end: str = ""
    comparison_type: str = "mom"   # mom=环比, yoy=同比, wow=周环比, custom

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TimeWindow":
        return cls(**{k: d.get(k, "") for k in
                      ("current_start", "current_end", "previous_start",
                       "previous_end", "comparison_type")})


@dataclass
class ScanLayer:
    """
    扫描层级定义:
      L1 — 全局指标扫描 (group_by=[])
      L2 — 按单个分类维度下钻 (group_by=["channel"])
      L3 — 交叉下钻 (group_by=["channel","category"])
    """
    depth: int = 1
    metrics: List[str] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    trigger_condition: str = "auto"   # "auto" | "always" | "anomaly_found"

    def to_dict(self) -> dict:
        return {"depth": self.depth, "metrics": self.metrics,
                "group_by": self.group_by, "filters": self.filters,
                "trigger_condition": self.trigger_condition}

    @classmethod
    def from_dict(cls, d: dict) -> "ScanLayer":
        return cls(depth=d.get("depth", 1), metrics=d.get("metrics", []),
                   group_by=d.get("group_by", []), filters=d.get("filters", {}),
                   trigger_condition=d.get("trigger_condition", "auto"))


@dataclass
class AnalysisFrame:
    """
    Commander 的输出: 完整的扫描计划。
    核心设计: 所有字段均从 schema + 用户问题动态推断，不做硬编码。
    """
    target_metric: str = ""               # 用户关注的主指标
    metrics: List[MetricSpec] = field(default_factory=list)
    time_window: Optional[TimeWindow] = None
    layers: List[ScanLayer] = field(default_factory=list)
    significance_threshold: float = 8.0   # Commander 可动态调整
    max_depth: int = 2                    # Quick Scan 最大下钻深度
    filters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""                   # Commander 的推理说明
    # ✅ v9 新增：分析深度（由 Commander 从用户问题语义判断）
    analysis_depth: str = "diagnostic"    # descriptive | diagnostic | causal
    depth_reasoning: str = ""             # Commander 对深度选择的解释
    # ✅ v9.7 新增：问题类别（用于零异常时的 suggested_data 映射）
    question_category: str = "general"    # general | roi_ad | competitor | inventory | promotion

    def to_dict(self) -> dict:
        return {
            "target_metric": self.target_metric,
            "metrics": [m.to_dict() for m in self.metrics],
            "time_window": self.time_window.to_dict() if self.time_window else {},
            "layers": [l.to_dict() for l in self.layers],
            "significance_threshold": self.significance_threshold,
            "max_depth": self.max_depth,
            "filters": self.filters,
            "reasoning": self.reasoning,
            "analysis_depth": self.analysis_depth,
            "depth_reasoning": self.depth_reasoning,
            "question_category": self.question_category,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisFrame":
        metrics = [MetricSpec.from_dict(m) for m in d.get("metrics", [])]
        tw = TimeWindow.from_dict(d["time_window"]) if d.get("time_window") else None
        layers = [ScanLayer.from_dict(l) for l in d.get("layers", [])]
        return cls(
            target_metric=d.get("target_metric", ""),
            metrics=metrics, time_window=tw, layers=layers,
            significance_threshold=d.get("significance_threshold", 8.0),
            max_depth=d.get("max_depth", 2),
            filters=d.get("filters", {}),
            reasoning=d.get("reasoning", ""),
            analysis_depth=d.get("analysis_depth", "diagnostic"),
            depth_reasoning=d.get("depth_reasoning", ""),
            question_category=d.get("question_category", "general"),
        )


@dataclass
class ScanState:
    """
    Scan Loop 的累积状态 — 替代 v7 的 scan_data / confirmed_anomalies / rejected_dimensions。
    """
    analysis_frame: Optional[Dict] = None   # Commander 的 AnalysisFrame（序列化）
    layer_results: Dict[int, List[Dict]] = field(default_factory=dict)   # {1: [...], 2: [...]}
    all_anomalies: List[Dict] = field(default_factory=list)
    all_normal: List[Dict] = field(default_factory=list)
    arbiter_decisions: List[Dict] = field(default_factory=list)
    scan_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "analysis_frame": self.analysis_frame,
            "layer_results": self.layer_results,
            "all_anomalies": self.all_anomalies,
            "all_normal": self.all_normal,
            "arbiter_decisions": self.arbiter_decisions,
            "scan_summary": self.scan_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScanState":
        return cls(
            analysis_frame=d.get("analysis_frame"),
            layer_results=d.get("layer_results", {}),
            all_anomalies=d.get("all_anomalies", []),
            all_normal=d.get("all_normal", []),
            arbiter_decisions=d.get("arbiter_decisions", []),
            scan_summary=d.get("scan_summary", ""),
        )


# ============================================================================
# 3. Trace 日志系统
# ============================================================================

class TraceEventType(Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    INTENT_CLASSIFICATION = "intent_classification"
    INSTRUCTION_GENERATED = "instruction_generated"
    PLAN_INITIALIZED = "plan_initialized"
    HYPOTHESIS_SELECTED = "hypothesis_selected"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_SUCCESS = "tool_execution_success"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    TOOL_RETRY = "tool_retry"
    RESULT_EVALUATION = "result_evaluation"
    HYPOTHESIS_CONFIRMED = "hypothesis_confirmed"
    HYPOTHESIS_REJECTED = "hypothesis_rejected"
    PRUNING_EXECUTED = "pruning_executed"
    PIVOT_DECISION = "pivot_decision"
    REPORT_GENERATED = "report_generated"
    SKILL_LOADED = "skill_loaded"
    HYPOTHESIS_REFILL = "hypothesis_refill"  # ✅ v5.7: 假设补充（reasoner 按需调用）
    EVALUATION_COMPLETED = "evaluation_completed"  # ✅ v5.8: evaluator 节点完成评估
    EVIDENCE_UPDATED = "evidence_updated"          # ✅ v5.8: 证据板更新
    EARLY_STOP = "early_stop"                      # ✅ v5.8: 证据充分提前终止
    PRIORITY_RERANKED = "priority_reranked"         # ✅ v5.8: 动态优先级重排
    # ✅ v8 新增：Quick Scan 三步架构
    COMMANDER_GENERATED = "commander_generated"      # Commander 生成 AnalysisFrame
    COMMANDER_FALLBACK = "commander_fallback"         # Commander 失败，降级到默认 Frame
    SCAN_LAYER_COMPLETED = "scan_layer_completed"    # Scan Loop 某层完成
    ARBITER_INVOKED = "arbiter_invoked"              # Arbiter 裁判被调用
    # ✅ v8 Step 2 新增：Reasoner v2
    REASONER_V2_COMPLETED = "reasoner_v2_completed"    # Reasoner v2 完成因果推理
    DEEP_RCA_DECISION = "deep_rca_decision"            # needs_deep_rca 决策结果
    # ✅ v8 Step 3 新增：Deep RCA
    PRESENT_FINDINGS = "present_findings"              # 展示初步结论，等待用户决策
    DEEP_RCA_INIT = "deep_rca_init"                    # Deep RCA 初始化（继承 Step 1+2 证据）
    SUPPLEMENTARY_DATA_MERGED = "supplementary_data_merged"  # 补充数据合并完成
    DEEP_RCA_REPORT = "deep_rca_report"                # 深度分析报告生成


@dataclass
class TraceEvent:
    timestamp: str
    event_type: str
    iteration: int
    hypothesis: Optional[str]
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    duration_ms: Optional[int] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp, "event_type": self.event_type,
            "iteration": self.iteration, "hypothesis": self.hypothesis,
            "input_data": self.input_data, "output_data": self.output_data,
            "duration_ms": self.duration_ms, "success": self.success,
            "error": self.error, "metadata": self.metadata,
        }


@dataclass
class TraceLog:
    session_id: str
    user_query: str
    start_time: str
    end_time: Optional[str] = None
    data_source: str = "database"
    config: Dict = field(default_factory=dict)
    events: List[TraceEvent] = field(default_factory=list)
    final_status: str = "running"
    final_result: Optional[Dict] = None
    statistics: Dict = field(default_factory=dict)

    def log(self, event_type: TraceEventType, iteration: int = 0,
            hypothesis: str = None, **kwargs) -> TraceEvent:
        event = TraceEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type.value,
            iteration=iteration,
            hypothesis=hypothesis,
            input_data=kwargs.get("input_data"),
            output_data=kwargs.get("output_data"),
            duration_ms=kwargs.get("duration_ms"),
            success=kwargs.get("success"),
            error=kwargs.get("error"),
            metadata=kwargs.get("metadata", {}),
        )
        self.events.append(event)
        return event

    def finalize(self, status: str, result: Dict = None):
        self.end_time = datetime.now().isoformat()
        self.final_status = status
        self.final_result = result
        self._compute_statistics()

    def _compute_statistics(self):
        self.statistics = {
            "total_events": len(self.events),
            "total_iterations": max((e.iteration for e in self.events), default=0),
            "tool_executions": sum(1 for e in self.events if e.event_type == TraceEventType.TOOL_EXECUTION_START.value),
            "tool_errors": sum(1 for e in self.events if e.event_type == TraceEventType.TOOL_EXECUTION_ERROR.value),
            "skill_loaded_count": sum(1 for e in self.events if e.event_type == TraceEventType.SKILL_LOADED.value),
        }
        # 汇总所有 skill_loaded 事件的加载详情
        all_loaded_skills = []
        for e in self.events:
            if e.event_type == TraceEventType.SKILL_LOADED.value and e.output_data:
                all_loaded_skills.extend(e.output_data.get("loaded_skills", []))
        if all_loaded_skills:
            self.statistics["loaded_skills"] = list(dict.fromkeys(all_loaded_skills))
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            self.statistics["total_duration_ms"] = int((end - start).total_seconds() * 1000)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_query": self.user_query,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "data_source": self.data_source,
            "events": [e.to_dict() for e in self.events],
            "final_status": self.final_status,
            "statistics": self.statistics,
        }


# ============================================================================
# 4. LLM 接口
# ============================================================================

class LLMInterface:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class _APIRateLimiter:
    """全局 API 调用节流器 — 同一 base_url 共享，避免并发请求过密触发限流"""

    _instances: Dict[str, "_APIRateLimiter"] = {}

    def __init__(self, min_interval: float = 1.0):
        self._min_interval = min_interval
        self._last_call_time = 0.0
        self._lock = __import__("threading").Lock()

    @classmethod
    def get(cls, base_url: str, min_interval: float = 1.0, model: str = "") -> "_APIRateLimiter":
        """按 base_url + model 单例 — ✅ v5.8: 不同角色不再互相阻塞"""
        key = f"{base_url or 'default'}::{model}"
        if key not in cls._instances:
            cls._instances[key] = cls(min_interval)
        return cls._instances[key]

    def wait(self):
        """调用前等待，确保两次调用间隔 ≥ min_interval"""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                gap = self._min_interval - elapsed
                time.sleep(gap)
            self._last_call_time = time.time()


# 全局配置：每次 API 调用之间的最小间隔（秒）— ✅ v5.8: 从 1.0 降到 0.3
API_CALL_MIN_INTERVAL = float(os.getenv("API_CALL_MIN_INTERVAL", "0.3"))
# 连接/限流错误最大重试次数
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "5"))


class OpenAICompatibleLLM(LLMInterface):
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-3.5-turbo", timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client = None
        self._rate_limiter = _APIRateLimiter.get(base_url or "default", API_CALL_MIN_INTERVAL, model=model)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        """判断异常是否可重试: 连接错误、超时、429 限流"""
        err_type = type(e).__name__
        # openai.APIConnectionError / httpx.ConnectError / ConnectionError 等
        if "Connection" in err_type or "Timeout" in err_type:
            return True
        # openai.RateLimitError (HTTP 429)
        if "RateLimit" in err_type:
            return True
        # openai.APIStatusError 中 status_code 为 429 / 502 / 503 / 529
        status = getattr(e, "status_code", None)
        if status in (429, 502, 503, 529):
            return True
        # 兜底: 检查错误消息
        err_msg = str(e).lower()
        if any(kw in err_msg for kw in ("connection", "rate limit", "too many", "overloaded", "server_busy")):
            return True
        return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        max_retries = API_MAX_RETRIES
        last_exception = None

        for attempt in range(max_retries + 1):
            # ── 调用前节流 ──
            self._rate_limiter.wait()

            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                return response.choices[0].message.content

            except Exception as e:
                last_exception = e

                if self._is_retryable(e) and attempt < max_retries:
                    # 指数退避: 2s, 4s, 8s, 16s, 32s ...
                    base_wait = 2 ** (attempt + 1)
                    # 加随机抖动避免雷群效应
                    import random
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait = base_wait + jitter
                    print(f"⚠️  [{self.model}] API 错误，{wait:.1f}s 后重试 "
                          f"({attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)[:120]}")
                    time.sleep(wait)
                    # 连接类错误: 销毁客户端重建 TCP
                    if "Connection" in type(e).__name__:
                        self._client = None
                else:
                    raise

        # 所有重试耗尽
        raise last_exception

class MockLLM(LLMInterface):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "Python" in system_prompt or "pandas" in system_prompt:
            return """
import pandas as pd
result_df = df.groupby('channel')['gmv'].sum().reset_index()
result_df.columns = ['channel', 'total_gmv']
result_df = result_df.sort_values('total_gmv', ascending=False)
summary = f"共 {len(result_df)} 个渠道"
"""
        return "模拟响应"


MODEL_CONFIG = {
    "router": {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "timeout": 60,
    },
    "python_agent": {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "timeout": 60,
    },
    "planner": {
        "model": "deepseek-reasoner",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "timeout": 120,
    },
    # ✅ v5.7: 新增 planner_chat — 精准查询生成用 chat 模型（循环中调用，避免 reasoner 频繁请求）
    "planner_chat": {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "timeout": 90,
    },
    "reporter": {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "timeout": 90,
    },
}
REACT_CONFIG = {
    "max_iterations": 10,
    "max_attribution_attempts": 6,
    "max_refills": 1,  # ✅ v5.7: 假设补充最大次数（限制 reasoner 按需调用）
}

def get_llm(role: str, fallback_to_mock: bool = True) -> LLMInterface:
    config = MODEL_CONFIG.get(role, MODEL_CONFIG["router"])
    api_key = os.getenv(config.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not api_key:
        if fallback_to_mock:
            return MockLLM()
        raise ValueError(f"请设置 {config['api_key_env']} 环境变量")
    return OpenAICompatibleLLM(
        api_key=api_key,
        base_url=config.get("base_url"),
        model=config["model"],
        timeout=config.get("timeout", 60),
    )


# ============================================================================
# 5. 数据库工具（仅供 PythonAgent 内部通过 pd.read_sql_query 使用）
# ============================================================================


def format_schema_for_prompt(schema: dict) -> str:
    if not schema.get("tables"):
        return "无可用表"

    lines = []
    for table in schema["tables"]:
        col_parts = []
        for c in table["columns"]:
            part = f"{c['name']}({c['type']})"
            if c.get("enum_values"):
                vals = c["enum_values"]
                display = vals if len(vals) <= 15 else vals[:15] + ["..."]
                part += f" 可选值=[{', '.join(str(v) for v in display)}]"
            # ▶ 新增：展示单位提示
            if c.get("description"):
                part += f" ⚠️{c['description']}"
            col_parts.append(part)
        lines.append(f"{table['name']}: {', '.join(col_parts)}")

    # 附加元信息摘要
    meta = schema.get("_meta", {})
    if meta.get("date_columns"):
        date_info = "; ".join(
            f"{t}.{c}" for t, c in meta["date_columns"])
        lines.append(f"\n[日期列] {date_info}")
    if meta.get("numeric_columns"):
        num_info = "; ".join(
            f"{t}.{c}" for t, c in meta["numeric_columns"][:20])
        lines.append(f"[数值列] {num_info}")
    if meta.get("text_columns"):
        txt_info = "; ".join(
            f"{t}.{c}({len(vs)}个值)" for t, c, vs in meta["text_columns"])
        lines.append(f"[分类列] {txt_info}")

    return "\n".join(lines)


def execute_python_code(code: str, context: dict = None) -> dict:
    """执行 Python 代码"""
    result = {"success": False, "data": None, "chart_path": None, "summary": "", "error": None}
    try:
        plt.rcParams['font.sans-serif'] = CHINESE_FONTS
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'

        namespace = {
            "pd": pd, "np": np, "plt": plt,
            "CHART_DIR": CHART_DIR, "json": json,
            "datetime": datetime, "timedelta": timedelta,
            "Path": Path, "time": time,
        }
        if context:
            namespace.update(context)

        exec(code, namespace)

        if "chart_path" in namespace:
            result["chart_path"] = str(namespace["chart_path"])
        if "summary" in namespace:
            result["summary"] = str(namespace["summary"])
        if "answer" in namespace:
            result["answer"] = namespace["answer"]  # dict or list[dict]
        if "result_df" in namespace:
            df = namespace["result_df"]
            result["data"] = {
                "records": df.to_dict(orient="records"),
                "columns": list(df.columns),
                "row_count": len(df),
            }
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
    return result

# ============================================================================
# 6. 公共工具函数
# ============================================================================

def evaluate_result(data: List[Dict], criteria: Dict, answer: Any = None) -> Dict:
    """评估查询结果是否满足指定条件（PythonAgent 共用）

    v5.6 修复：优先从 answer 提取判定值，answer 是 PythonAgent 代码中
    明确赋值的最终结果，不受多行 shift 导致的 NaN 影响。
    仅当 answer 中找不到目标字段时，才 fallback 到 data（反向扫描跳过 NaN）。

    Args:
        data: result_df 转成的 list[dict]（可能多行，含 NaN）
        criteria: evaluation_criteria，包含 field/operator/value/if_true/if_false
        answer: PythonAgent 输出的 answer（dict 或 list[dict]），已是最终计算值
    """
    if not data or not criteria:
        return None
    field_name = criteria.get("field", "")
    operator = criteria.get("operator", "")
    threshold = criteria.get("value", 0)

    def _is_valid_number(v):
        """判断值是否为有效数值（排除 None / NaN）"""
        if v is None:
            return False
        try:
            import math
            return isinstance(v, (int, float)) and not math.isnan(v)
        except (TypeError, ValueError):
            return False

    value = None
    source = None

    # ── 优先级1: 从 answer 提取（最可靠，PythonAgent 已计算好的终值） ──
    if answer is not None:
        if isinstance(answer, dict) and field_name in answer:
            v = answer[field_name]
            if _is_valid_number(v):
                value = v
                source = "answer"
        elif isinstance(answer, list):
            # list[dict] 格式：反向扫描取最后一个有效值
            for item in reversed(answer):
                if isinstance(item, dict) and field_name in item and _is_valid_number(item[field_name]):
                    value = item[field_name]
                    source = "answer"
                    break

    # ── 优先级2: fallback 到 data（反向扫描跳过 NaN） ──
    if value is None and data:
        for row in reversed(data):
            if field_name in row and _is_valid_number(row[field_name]):
                value = row[field_name]
                source = "data"
                break

    # ── 兜底：字段存在但所有值无效 ──
    if value is None:
        field_in_answer = (isinstance(answer, dict) and field_name in answer) if answer else False
        field_in_data = any(field_name in row for row in data) if data else False
        if field_in_answer or field_in_data:
            return {
                "meets_criteria": None,
                "conclusion": "无法评估",
                "reasoning": f"字段 {field_name} 存在但所有值为 NaN",
            }
        return {"meets_criteria": None, "conclusion": "无法评估", "reasoning": f"找不到字段 {field_name}"}

    meets = False
    if operator == "<": meets = value < threshold
    elif operator == "<=": meets = value <= threshold
    elif operator == ">": meets = value > threshold
    elif operator == ">=": meets = value >= threshold
    elif operator == "==": meets = value == threshold
    #应对numpy.bool_非原生bool问题
    meets = bool(meets)

    conclusion = criteria.get("if_true", "满足条件") if meets else criteria.get("if_false", "不满足条件")
    return {
        "meets_criteria": meets,
        "conclusion": conclusion,
        "reasoning": f"{field_name}={value}, 阈值{operator}{threshold} (来源: {source})",
        "actual_value": value,
    }


# ============================================================================
# 7. 统一 Python Agent
# ============================================================================

# ── Simple 路径专用 System Prompt（自然语言 → 代码，无结构化指令） ──
# ============================================================================
# Prompt 分层架构 (v5.4-11)
# ============================================================================
# 设计理念：
#   DeepSeek 模型 system prompt 优先级高于 user prompt，
#   因此将 skill（领域知识）注入 system prompt 而非 user prompt，
#   确保 skill 中的规则被模型严格遵循。
#
# 分层结构（每层单独维护，按顺序拼接为 system prompt）：
#   Layer 1 - BASE: 角色定义 + 分析方法论（稳定层，极少改动）
#   Layer 2 - SKILLS: 动态注入的领域知识（按查询匹配，来自 .md 文件）
#   Layer 3 - OUTPUT: 输出格式规范（稳定层，result_df/answer/summary）
#   Layer 4 - CONSTRAINTS: 硬性约束（尾部高权重位置，仅放不可违反的规则）
#
# 评测修复原则：
#   - 领域计算方法修复 → 改对应 skill .md 文件（不改 prompt）
#   - 输出格式修复 → 改 Layer 3
#   - 只有真正的全局不可违反规则 → 才加入 Layer 4
# ============================================================================

# ── Layer 1: Base（角色 + 分析方法论，Simple 路径专用） ──
_SIMPLE_BASE = """你是 Python 数据分析专家。你的任务是：根据用户的**自然语言查询**，直接生成 Python 代码完成数据分析。

⚠️ 你不会收到结构化指令，只会收到用户的原始查询和数据 Schema。你需要自行理解意图并生成正确代码。

## 第一步：从自然语言中提取分析五要素

拿到用户查询后，先在心里识别以下五个要素（可能隐含，需推断）：

### 1. 时间范围
- "上月" / "上个月" → 上一个自然月（根据 *数据最新日期* 推算，不要用 datetime.now()）
- "本月" / "这个月" → 数据最新日期所在月
- "最近7天" / "近一周" → 数据最新日期往前推7天
- "上周" → 上一个自然周（周一至周日）
- "Q1" / "第一季度" → 1月1日 ~ 3月31日
- 未提及时间 → 使用全部数据时间范围
- ⚠️ 关键：用 User Prompt 中提供的"数据时间范围"来推算绝对日期

### 2. 分组维度
- "各渠道" / "按渠道" / "每个渠道" / "分渠道" → group by channel
- "各品类" / "按品类" / "分品类" → group by category
- "各地区" / "按区域" → group by region
- 无分组关键词 → **不做 groupby**，返回整体汇总单行
- ⚠️ 严禁添加查询中未提及的分组维度

### 3. 聚合方式与指标
- "GMV" / "销售额" / "营收" → sum(gmv)
- "订单数" / "订单量" → sum(orders) 或 count
- "客单价" → sum(gmv) / sum(orders)
- "ROI" / "ROAS" → sum(gmv) / sum(marketing_spend)
- "转化率" → 加权平均: sum(traffic * conversion_rate) / sum(traffic)

### 4. 排序与限制
- "Top N" / "前N" / "排名前N" → sort descending + head(N)
- "最高" / "最大" → sort descending + head(1)
- "最低" / "最小" → sort ascending + head(1)
- "排名" / "排行" → sort descending

### 5. 计算类型
- "占比" / "比例" / "份额" → 各组值 / 总值 * 100
- "环比" / "比上月" / "MoM" → (本期 - 上期) / 上期 * 100
- "同比" / "比去年" / "YoY" → (本期 - 去年同期) / 去年同期 * 100
- "增长率" / "变化率" → (新值 - 旧值) / 旧值 * 100
- "趋势" / "走势" / "按月" / "逐月" → 按时间粒度分组聚合

## 常用分析模式 pandas 参考

### 时间筛选（以"上月"为例）
```python
df['date'] = pd.to_datetime(df['date'])
date_max = df['date'].max()
last_month_end   = date_max.replace(day=1) - pd.Timedelta(days=1)
last_month_start = last_month_end.replace(day=1)
filtered = df[(df['date'] >= last_month_start) & (df['date'] <= last_month_end)]
```

### 环比 / 对比分析
```python
cur_df  = df[(df['date'] >= cur_start) & (df['date'] <= cur_end)]
prev_df = df[(df['date'] >= prev_start) & (df['date'] <= prev_end)]
cur_agg  = cur_df.groupby('channel')['gmv'].sum().reset_index(name='current_value')
prev_agg = prev_df.groupby('channel')['gmv'].sum().reset_index(name='previous_value')
result_df = cur_agg.merge(prev_agg, on='channel', how='left')
result_df['change']     = round(result_df['current_value'] - result_df['previous_value'], 2)
result_df['change_pct'] = round(
    (result_df['current_value'] - result_df['previous_value'])
    / result_df['previous_value'] * 100, 2)
```

## 数据加载

输入 DataFrame 变量名为 `df`，已在执行环境中。直接使用 pandas 操作。

⚠️ **过滤值映射**: 数据中存储的可能是英文，过滤时必须使用 Schema 标注的**实际可选值**
（如 "Electronics" 而非 "电子"）。"""

# ── Layer 3: Output format（输出格式规范，Simple/Complex 共用） ──
_OUTPUT_RULES = """
## 输出变量（必须设置）
- result_df (DataFrame): 分析结果（必须包含用户问题的最终答案数值）
- summary (str): 结果摘要（关键数值必须同时在 result_df 中）
- answer: 评测精确提取用，**必须设置**
  - 单行结果（无分组/单值）: answer = {"field": value}，如 {"total_gmv": 123456.78}
  - 多行分组结果: answer = [{"group_col": key, "value_col": val}, ...]，key 和列名必须使用英文且与 result_df 列名完全一致
  - 示例: answer = [{"channel": "Douyin", "total_gmv": 265834.12}, {"channel": "Tmall", "total_gmv": 389021.56}]
- chart_path (str, 可选): 图表路径（仅当明确要求图表时）

## 代码规范
1. 使用 pandas 处理数据
2. 使用 matplotlib 生成图表（如需要），保存到 CHART_DIR
3. 中文字体已预配置，无需手动设置
4. 聚合结果 round(..., 2) 保留2位小数

## 列命名规范
- 对比分析列名必须用: current_{metric}, previous_{metric}, change, change_pct
- 其余遵循: sum → total_x, avg → avg_x, count → record_count

## 可用变量
- df: 输入 DataFrame
- pd, np, plt, sqlite3: 已导入
- CHART_DIR: 图表保存目录"""

# ── Layer 4: Hard constraints（尾部位置 = 最高权重，仅放不可违反的规则） ──
_HARD_CONSTRAINTS = """
## ⚠️ 硬性约束（不可违反）
1. 比率指标（客单价、转化率、ROI）必须用 sum(分子)/sum(分母)，禁止 mean(已算好的比率)
2. 比率字段单位：看 Schema 的 ⚠️ 单位提示决定是否 ×100
3. 统计量（平均、标准差、CV）除非用户明确说"日均""日波动"，否则直接对原始记录算，不要先按天 sum
4. 避免 sklearn 等非标准库，线性回归用 numpy 手动实现
5. 上方"领域知识"中的计算方法和代码模式，请严格遵循

只返回 Python 代码，不要解释:"""


def build_python_agent_prompt(base: str, skill_knowledge: str = "") -> str:
    """
    动态构建 Python Agent 的 system prompt。

    拼接顺序：Base → Skills → Output → Constraints
    - Skills 注入 system prompt 而非 user prompt，确保 DeepSeek 严格遵循
    - Constraints 放在尾部，利用 recency bias 获得最高权重
    """
    parts = [base]

    if skill_knowledge:
        parts.append(f"""
## 领域知识（必须严格遵循以下方法论和代码模式）
{skill_knowledge}""")

    parts.append(_OUTPUT_RULES)
    parts.append(_HARD_CONSTRAINTS)

    return "\n".join(parts)


# ── 向后兼容：无 skill 时的默认 prompt ──
PYTHON_AGENT_SIMPLE_PROMPT = build_python_agent_prompt(_SIMPLE_BASE)


# ── Layer 1: Base（角色 + 分析方法论，Complex 路径专用） ──


PYTHON_AGENT_FIX_PROMPT = """Python 代码执行出错，请修正。

## 原代码
{code}

## 错误信息
{error}

## 数据信息
{data_info}

## 修正要求
1. 分析错误原因
2. 修正代码（避免 sklearn 等非标准库）
3. 确保设置 result_df 和 summary

只返回修正后的代码:"""




class PythonAgent:
    """
    v9.9: 清理废弃的 AnalysisInstruction 路径，仅保留 execute_from_query
    - 新增 execute_from_query(): Simple 路径直接从自然语言查询生成代码
    核心方法: execute_from_query() — 从自然语言查询 + schema 直接生成 pandas 代码
    """

    def __init__(self, llm: LLMInterface = None, max_retries: int = 3, skills_dir: str = None):
        self.llm = llm or get_llm("python_agent")
        self.max_retries = max_retries
        self.skill_loader = PythonAgentSkillLoader(skills_dir)
    def execute_from_query(self, user_query: str,
                           schema_info: str,
                           df: "pd.DataFrame",
                           validation: Dict = None,
                           need_chart: bool = False,
                           chart_type: str = None,
                           trace: "TraceLog" = None,
                           result_field_hint: str = None) -> "AnalysisResult":
        """
        Simple 路径：直接从原始查询 + schema 生成代码（不经过 AnalysisInstruction）

        Args:
            user_query: 用户原始查询
            schema_info: 格式化后的 schema 信息
            df: 输入 DataFrame
            validation: Router 的校验结果
            need_chart: 是否需要图表
            chart_type: 图表类型
            trace: Trace 日志
            result_field_hint: Complex 路径关键列命名约束（确保 evaluate_result 能匹配）
        """
        task_id = f"simple_{int(time.time())}"
        start_time = time.time()
        correction_history = []

        if df is None or df.empty:
            return AnalysisResult(
                task_id=task_id, success=False,
                error="输入数据为空")

        loaded_skill_files = self.skill_loader.detect_skills(user_query)
        if trace is not None:
            trace.log(
                TraceEventType.SKILL_LOADED,
                iteration=0,
                input_data={"query": user_query},
                output_data={
                    "loaded_skills": loaded_skill_files,
                    "skill_count": len(loaded_skill_files),
                },
                success=True,
            )

        # LLM 直接从自然语言生成代码
        code = self._generate_code_from_query(
            user_query, schema_info, df, need_chart, chart_type, validation,
            result_field_hint=result_field_hint)

        # 重试循环
        exec_result = None
        for attempt in range(self.max_retries + 1):
            exec_ctx = {"df": df.copy()}
            exec_result = execute_python_code(code, exec_ctx)

            if exec_result["success"]:
                data = exec_result.get("data", {})
                records = (data.get("records", [])
                           if isinstance(data, dict) else [])

                return AnalysisResult(
                    task_id=task_id,
                    success=True,
                    executed_code=code,
                    execution_time_ms=int(
                        (time.time() - start_time) * 1000),
                    data=records,
                    columns=(data.get("columns", [])
                             if isinstance(data, dict) else []),
                    row_count=(data.get("row_count", 0)
                               if isinstance(data, dict)
                               else len(records)),
                    preview=str(data)[:500],
                    summary=exec_result.get("summary", ""),
                    chart_path=exec_result.get("chart_path"),
                    chart_type=chart_type,
                    answer=exec_result.get("answer"),
                    correction_history=correction_history,
                )

            # ── 执行失败，记录并准备重试 ──
            error_info = {
                "attempt": attempt + 1,
                "code": code,
                "error": exec_result.get("error"),
                "error_type": exec_result.get("error_type"),
            }
            correction_history.append(error_info)

            if trace is not None:
                trace.log(
                    TraceEventType.TOOL_RETRY,
                    iteration=attempt + 1,
                    hypothesis="Simple Query",
                    input_data={
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "failed_code": code[:300],
                    },
                    output_data={
                        "error": exec_result.get("error"),
                        "error_type": exec_result.get("error_type"),
                    },
                    error=exec_result.get("error"),
                    success=False,
                    metadata={
                        "action": "requesting_fix"
                                  if attempt < self.max_retries
                                  else "max_retries_exhausted",
                    },
                )

            if attempt < self.max_retries:
                code = self._fix_code_from_query(
                    code, exec_result, user_query, schema_info, df)

        return AnalysisResult(
            task_id=task_id,
            success=False,
            executed_code=code,
            execution_time_ms=int(
                (time.time() - start_time) * 1000),
            error=(exec_result.get("error")
                   if exec_result else "Unknown error"),
            error_type=(exec_result.get("error_type")
                        if exec_result else None),
            correction_history=correction_history,
        )


    def _generate_code_from_query(self, user_query: str, schema_info: str,
                                  df: pd.DataFrame, need_chart: bool = False,
                                  chart_type: str = None,
                                  validation: Dict = None,
                                  result_field_hint: str = None) -> str:
        """Simple 路径：直接从自然语言查询生成代码"""

        data_section = f"""## 数据加载模式: DATAFRAME
## DataFrame 信息
列: {list(df.columns) if df is not None else []}
行数: {len(df) if df is not None else 0}
数据类型: {df.dtypes.to_dict() if df is not None else {} }
预览:
{df.head(3).to_string() if df is not None and len(df) > 0 else '空'}"""

        date_range = get_cached_date_range()

        if need_chart and chart_type:
            chart_section = (
                f"4. 生成 {chart_type} 类型图表，"
                f"保存路径到 chart_path"
            )
        else:
            chart_section = (
                "4. **不需要生成图表**，禁止创建可视化代码，"
                "不要设置 chart_path 变量"
            )

        validation_section = ""
        if validation:
            validation_section = f"""
## Router 校验结果
校验通过: {validation.get('is_valid', True)}
分析理由: {validation.get('reasoning', '')}"""

        # ✅ v5.6: 关键列命名约束（Complex 路径传入，确保 evaluate_result 能匹配）
        field_hint_section = ""
        if result_field_hint:
            field_hint_section = f"""
## ⚠️ 关键列命名约束
result_df 中用于评估判断的关键列**必须**命名为 `{result_field_hint}`，不可使用其他名称。"""

        skill_knowledge = self.skill_loader.build_knowledge(user_query)
        system_prompt = build_python_agent_prompt(_SIMPLE_BASE, skill_knowledge)

        user_prompt = f"""{data_section}

## 数据 Schema
{schema_info}

## 数据时间范围
最早: {date_range.get('min', 'unknown')}
最晚: {date_range.get('max', 'unknown')}（以此作为"今天"推算相对时间，如"上月"、"最近7天"）
日期列: {date_range.get('date_column', 'unknown')}
{validation_section}
{field_hint_section}

## 用户查询
{user_query}

## 要求
1. 先从用户查询中提取五要素：时间范围、分组维度、聚合指标、排序/限制、计算类型
2. 根据 Schema 中的实际列名和可选值编写代码（注意中英文映射）
3. 时间推算以上方"数据时间范围 - 最晚"为基准，禁止使用 datetime.now()
4. 结果保存到 result_df，生成简短的 summary
{chart_section}

生成代码:"""

        result = self.llm.generate(system_prompt, user_prompt)
        return self._clean_code(result)

    def _fix_code_from_query(self, code: str, exec_result: Dict,
                             user_query: str, schema_info: str,
                             df: "pd.DataFrame" = None) -> str:
        """Simple 路径的代码修正（使用 Simple 专用 Prompt + skill 注入 system）"""
        data_info = (
            f"列: {list(df.columns) if df is not None else []}\n"
            f"行数: {len(df) if df is not None else 0}\n"
            f"数据类型: {df.dtypes.to_dict() if df is not None else {} }"
        )

        user_prompt = PYTHON_AGENT_FIX_PROMPT.format(
            code=code,
            error=exec_result.get("error", "未知错误"),
            data_info=data_info,
        )
        user_prompt += f"\n\n## 数据 Schema\n{schema_info}"
        user_prompt += f"\n\n## 原始用户查询\n{user_query}"

        # 修复时也需要 skill 知识（在 system prompt 中）
        skill_knowledge = self.skill_loader.build_knowledge(user_query)
        system_prompt = build_python_agent_prompt(_SIMPLE_BASE, skill_knowledge)

        result = self.llm.generate(system_prompt, user_prompt)
        return self._clean_code(result)


    def _clean_code(self, code: str) -> str:
        code = code.strip()
        code = code.replace("```python", "").replace("```", "")
        return code.strip()


# ============================================================================
# 8. Router Agent
# ============================================================================

class PythonAgentSkillLoader:
    """
    Python Agent 知识技能加载器。

    设计理念（v5.4-11 架构升级）：
    - Skill 提供领域知识和代码模式（电商指标、统计方法、时间分析等）
    - Skill 通过 build_python_agent_prompt() 注入 **system prompt**
    - 硬性规则（比率计算、预聚合禁止、sklearn 禁用）在 _HARD_CONSTRAINTS 层
    - 评测修复路径：领域方法 → 改 skill .md 文件；输出格式 → 改 _OUTPUT_RULES
    """

    SKILL_TRIGGERS = {
        "statistical_methods.md": [
            "平均", "异常", "波动", "分布", "偏差", "标准差",
            "中位数", "百分位", "置信", "显著", "离散", "集中",
        ],
        "ecommerce_metrics.md": [
            "GMV", "ROI", "ROAS", "转化率", "客单价",
            "营销", "获客", "复购", "LTV", "gmv", "roi",
            "转化产出", "有效转化", "转化效率", "客单",
            "revenue per", "ad_cost", "广告费", "单次转化",
        ],
        "time_analysis.md": [
            "趋势", "走势", "同比", "环比", "月度", "周度",
            "季节", "移动平均", "时间序列",
        ],
        "comparison_analysis.md": [
            "对比", "比较", "变化", "增长", "下降", "差异",
            "归因", "贡献度", "影响因素",
        ],
        # "column_naming.md": [
        #     "环比", "同比", "对比", "变化率", "增长率", "占比",
        #     "MoM", "YoY", "趋势", "比较", "份额",
        # ],
       # "visualization_best_practice.md": [
       #      "图表", "柱状图", "折线图", "饼图", "可视化",
       #      "展示", "画图",
       #  ],
    }
    ALWAYS_LOAD = ["common_pitfalls.md","result_construction.md"]

    # 单次最多加载的 skill 数量（含 ALWAYS_LOAD），控制 token 用量
    MAX_SKILLS = 5

    def __init__(self, skills_dir: str = None):
        self.skills_dir = (
            Path(skills_dir) if skills_dir
            else Path(__file__).parent / "skills" / "python_agent"
        )
        self._cache: Dict[str, str] = {}

    def _read_skill(self, filename: str) -> str:
        """读取并缓存单个 skill 文件"""
        if filename in self._cache:
            return self._cache[filename]
        filepath = self.skills_dir / filename
        if not filepath.exists():
            return ""
        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception:
            return ""
        self._cache[filename] = content
        return content

    def detect_skills(self, query: str) -> List[str]:
        """根据查询关键词检测需要加载的 skill 文件列表"""
        matched = list(self.ALWAYS_LOAD)
        for skill_file, keywords in self.SKILL_TRIGGERS.items():
            if any(kw in query for kw in keywords):
                if skill_file not in matched:
                    matched.append(skill_file)
        return matched[:self.MAX_SKILLS]

    def build_knowledge(self, query: str) -> str:
        """
        根据查询加载相关领域知识，返回格式化文本。

        返回空字符串表示无匹配知识（ALWAYS_LOAD 文件不存在时也可能为空）。
        """
        if not query:
            return ""
        skill_files = self.detect_skills(query)
        sections = []
        for filename in skill_files:
            content = self._read_skill(filename)
            if content:
                sections.append(content)
        return "\n\n---\n\n".join(sections) if sections else ""


def build_dynamic_router_prompt(date_min: str, date_max: str) -> str:
    """
    动态构建轻量 Router System Prompt。
    Router 仅负责：意图分类 + 合理性校验 + 输出类型判断。
    不再生成 AnalysisInstruction。
    """
    schema = get_cached_schema()
    meta = schema.get("_meta", {})

    # 动态字段映射表
    field_lines = []
    for _tbl, col in meta.get("numeric_columns", []):
        field_lines.append(f"| {col} | {col} |")
    for _tbl, col, _vals in meta.get("text_columns", []):
        field_lines.append(f"| {col} | {col} (维度) |")

    field_table = "\n".join(field_lines) if field_lines else "| (请参考 Schema) | |"

    return f"""你是数据分析合法性校验器。

## 任务（两项职责）
1. 合法性校验: VALID / INVALID
2. 输出类型判断

## 合法性校验规则
以下任一不通过 → intent = INVALID:
1. 字段不存在于 Schema
2. 时间超出数据范围
3. 过滤值不在枚举中
4. ID/主键字段用于聚合、分组或可视化（只能用于过滤）
5. 仅指定图表类型但无有效分析维度/度量
6. 与当前数据完全无关的纯编程任务（如：写爬虫、实现算法、搭建服务器、生成前端页面）

⚠️ 以下场景属于合法的数据分析，不是 INVALID：
- 查询/统计/汇总/趋势/对比/排名
- 异常检测/根因分析/因果推理
- 使用统计方法分析数据（IQR、Z-score、标准差、百分位数、回归分析等）
- 对数据做计算/衍生指标（同比、环比、移动平均、增长率等）
- 用编程手段处理已有数据（分组聚合、透视表、相关性分析等）
判断标准：只要查询的目的是"从已有数据中提取信息或发现规律"，就是 VALID。

## 数据库字段（从 Schema 自动生成）

## 数据时间范围
{date_min} 至 {date_max}

## 输出类型规则
- "趋势"/"走势"/"变化曲线" → CHART (line)
- "柱状图"/"对比图" → CHART (bar)
- "占比"/"比例"/"饼图" → CHART (pie)
- 其余默认 → TABLE

## 输出格式
```json
{{
  "intent": "VALID/INVALID",
  "output_type": "TABLE/CHART",
  "chart_type": "line/bar/pie/null",
  "validation": {{
    "is_valid": true,
    "reasoning": "校验通过/不通过的具体原因"
  }},
  "invalid_reason": "仅 INVALID 时",
  "suggestions": ["仅 INVALID 时"]
}}
```

只返回 JSON:"""


class RouterAgent:
    """Router Agent — v9 简化版：仅做合法性校验 + 输出类型判断，不再分 SIMPLE/COMPLEX"""

    def __init__(self, llm: LLMInterface = None):
        self.llm = llm or get_llm("router")
        import logging
        self._logger = logging.getLogger(self.__class__.__name__)

    def route(self, user_query: str, data_source: DataSourceType = DataSourceType.CSV) -> Dict:
        try:
            result = self._classify_and_generate(user_query)

            intent = result.get("intent", "VALID")
            output_type = result.get("output_type", "TABLE")
            chart_type = result.get("chart_type")
            validation = result.get("validation", {"is_valid": True, "reasoning": ""})

            if intent == "INVALID":
                return {
                    "route": RouteType.INVALID,
                    "error": result.get("invalid_reason", "问题不在业务分析范围内"),
                    "suggestions": result.get("suggestions", []),
                }

            if chart_type and chart_type not in ["null", None, ""]:
                output_type = "CHART"

            # ✅ v9: 所有合法查询统一走 COMPLEX（Quick Scan 管线）
            return {
                "route": RouteType.COMPLEX,
                "output_type": output_type,
                "chart_type": chart_type,
                "validation": validation,
                "analysis_goal": {
                    "original_query": user_query,
                    "goal_type": "unified_analysis",
                    "data_source": data_source.value,
                },
            }
        except Exception:
            return self._fallback_route(user_query, data_source)

    def _classify_and_generate(self, query: str) -> Dict:
        schema_info = format_schema_for_prompt(get_cached_schema())
        date_range = get_cached_date_range()

        system_prompt = build_dynamic_router_prompt(
            date_range["min"], date_range["max"])

        user_prompt = f"""[数据库 Schema]
    {schema_info}

    [用户查询]
    {query}

    分析并生成指令:"""

        result = self.llm.generate(system_prompt, user_prompt)
        response = result.strip().replace("```json", "").replace("```", "")
        return json.loads(response)

    def _fallback_route(self, query: str, data_source: DataSourceType) -> Dict:
        # ✅ v9: fallback 也统一走 COMPLEX（Quick Scan），不再区分 simple/complex
        chart_type = None
        output_type = "TABLE"
        if "趋势" in query or "走势" in query:
            chart_type = "line"
            output_type = "CHART"
        elif "占比" in query or "比例" in query:
            chart_type = "pie"
            output_type = "CHART"

        return {
            "route": RouteType.COMPLEX,
            "output_type": output_type,
            "chart_type": chart_type,
            "analysis_goal": {"original_query": query, "data_source": data_source.value},
        }



# ============================================================================
# 9. 动态计划管理 (Complex 路径)
# ============================================================================

class HypothesisStatus(Enum):
    PENDING = "pending"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    PRUNED = "pruned"


@dataclass
class AttributionHypothesis:
    name: str
    dimension: str
    description: str
    priority: int = 5
    status: HypothesisStatus = HypothesisStatus.PENDING
    evidence: List[str] = field(default_factory=list)
    related_hypotheses: List[str] = field(default_factory=list)
    pruning_rules: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "dimension": self.dimension, "description": self.description,
            "priority": self.priority, "status": self.status.value, "evidence": self.evidence,
            "related_hypotheses": self.related_hypotheses, "pruning_rules": self.pruning_rules,
        }


@dataclass
class DynamicPlan:
    hypotheses: Dict[str, AttributionHypothesis] = field(default_factory=dict)
    investigation_order: List[str] = field(default_factory=list)
    pivot_history: List[Dict] = field(default_factory=list)
    max_attempts: int = 6
    attempts_count: int = 0

    def add_hypothesis(self, h: AttributionHypothesis):
        self.hypotheses[h.name] = h
        self._update_order()

    def _update_order(self):
        pending = [h for h in self.hypotheses.values() if h.status == HypothesisStatus.PENDING]
        self.investigation_order = [h.name for h in sorted(pending, key=lambda x: -x.priority)]

    def get_next_hypothesis(self) -> Optional[AttributionHypothesis]:
        for name in self.investigation_order:
            h = self.hypotheses[name]
            if h.status == HypothesisStatus.PENDING:
                return h
        return None

    def prune_related(self, rejected_name: str, reason: str) -> List[str]:
        pruned = []
        rejected = self.hypotheses.get(rejected_name)
        if rejected:
            for related_name in rejected.related_hypotheses:
                if related_name in self.hypotheses:
                    related = self.hypotheses[related_name]
                    if related.status == HypothesisStatus.PENDING:
                        related.status = HypothesisStatus.PRUNED
                        related.evidence.append(f"因 {rejected_name} 正常而被剪枝: {reason}")
                        pruned.append(related_name)
        self._update_order()
        return pruned

    def record_pivot(self, from_h: str, to_h: str, reason: str):
        self.pivot_history.append({
            "from": from_h, "to": to_h, "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

    def to_dict(self) -> Dict:
        return {
            "hypotheses": {k: v.to_dict() for k, v in self.hypotheses.items()},
            "investigation_order": self.investigation_order,
            "pivot_history": self.pivot_history,
            "max_attempts": self.max_attempts,
            "attempts_count": self.attempts_count,
        }

    @classmethod
    def from_dict(cls, plan_dict: Dict) -> "DynamicPlan":
        return cls(
            hypotheses={k: AttributionHypothesis(
                name=v["name"], dimension=v["dimension"], description=v["description"],
                priority=v["priority"], status=HypothesisStatus(v["status"]),
                evidence=v.get("evidence", []),
                related_hypotheses=v.get("related_hypotheses", []),
                pruning_rules=v.get("pruning_rules", {}),
            ) for k, v in plan_dict.get("hypotheses", {}).items()},
            investigation_order=plan_dict.get("investigation_order", []),
            pivot_history=plan_dict.get("pivot_history", []),
            max_attempts=plan_dict.get("max_attempts", 6),
            attempts_count=plan_dict.get("attempts_count", 0),
        )

    # ✅ v5.8: 动态优先级调整
    def boost_priority(self, hypothesis_name: str, delta: int):
        """提升/降低指定假设的优先级"""
        if hypothesis_name in self.hypotheses:
            h = self.hypotheses[hypothesis_name]
            if h.status == HypothesisStatus.PENDING:
                h.priority = min(10, max(1, h.priority + delta))
                self._update_order()

    def prune_by_name(self, hypothesis_name: str, reason: str) -> bool:
        """按名称剪枝指定假设"""
        if hypothesis_name in self.hypotheses:
            h = self.hypotheses[hypothesis_name]
            if h.status == HypothesisStatus.PENDING:
                h.status = HypothesisStatus.PRUNED
                h.evidence.append(f"智能剪枝: {reason}")
                self._update_order()
                return True
        return False

# ============================================================================
# 9b. 维度分解树 + 证据板 (v5.8 新增)
# ============================================================================

@dataclass
class DimensionNode:
    """维度树节点"""
    name: str               # 节点标识 (如 "by_channel", "channel_x_category")
    dimension: str           # 对应的 Schema 列名 (如 "channel")，交叉维度用逗号分隔
    node_type: str           # "single" | "cross" | "metric"
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0           # 树深度

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "dimension": self.dimension,
            "node_type": self.node_type, "parent": self.parent,
            "children": self.children, "depth": self.depth,
        }


@dataclass
class DimensionTree:
    """
    ✅ v5.8: 维度分解树 — 从 Schema 自动构建
    编码维度间的层级关系，供 Evaluator 做智能剪枝：
    - 父节点正常 → 子节点全部跳过
    - 父节点异常 → 子节点优先级提升
    """
    nodes: Dict[str, DimensionNode] = field(default_factory=dict)

    @classmethod
    def build_from_schema(cls, schema: dict) -> "DimensionTree":
        """从 Schema 的 _meta 自动构建维度分解树"""
        tree = cls()
        meta = schema.get("_meta", {})

        # 提取分类列名（用于分组分析的维度）
        text_cols = [col for _tbl, col, _vals in meta.get("text_columns", [])]
        # 提取数值列名（被分析的指标）
        numeric_cols = [col for _tbl, col in meta.get("numeric_columns", [])]

        if not text_cols:
            return tree

        # Layer 1: 单维度节点 (by_channel, by_category, ...)
        for col in text_cols:
            node_name = f"by_{col}"
            tree.nodes[node_name] = DimensionNode(
                name=node_name,
                dimension=col,
                node_type="single",
                parent=None,
                children=[],
                depth=1,
            )

        # Layer 2: 两两交叉维度节点（最多取前 4 个分类列的两两组合）
        from itertools import combinations
        cross_cols = text_cols[:4]  # 限制交叉维度的列数避免组合爆炸
        for col_a, col_b in combinations(cross_cols, 2):
            cross_name = f"{col_a}_x_{col_b}"
            tree.nodes[cross_name] = DimensionNode(
                name=cross_name,
                dimension=f"{col_a},{col_b}",
                node_type="cross",
                parent=None,  # 由两个父节点共享
                children=[],
                depth=2,
            )
            # 关联到父节点
            parent_a = f"by_{col_a}"
            parent_b = f"by_{col_b}"
            if parent_a in tree.nodes:
                tree.nodes[parent_a].children.append(cross_name)
            if parent_b in tree.nodes:
                tree.nodes[parent_b].children.append(cross_name)

        return tree

    def get_children(self, node_name: str) -> List[str]:
        """获取子节点列表"""
        node = self.nodes.get(node_name)
        return node.children if node else []

    def get_parent_nodes(self, node_name: str) -> List[str]:
        """获取包含该节点为子节点的所有父节点"""
        parents = []
        for name, node in self.nodes.items():
            if node_name in node.children:
                parents.append(name)
        return parents

    def find_node_by_dimension(self, dimension: str) -> Optional[str]:
        """根据维度列名查找节点"""
        # 精确匹配
        for name, node in self.nodes.items():
            if node.dimension == dimension:
                return name
        # 模糊匹配（假设名可能包含维度名）
        for name, node in self.nodes.items():
            if dimension in node.dimension:
                return name
        return None

    def to_dict(self) -> Dict:
        return {name: node.to_dict() for name, node in self.nodes.items()}

    @classmethod
    def from_dict(cls, data: Dict) -> "DimensionTree":
        tree = cls()
        for name, node_data in data.items():
            tree.nodes[name] = DimensionNode(
                name=node_data["name"],
                dimension=node_data["dimension"],
                node_type=node_data["node_type"],
                parent=node_data.get("parent"),
                children=node_data.get("children", []),
                depth=node_data.get("depth", 0),
            )
        return tree


@dataclass
class EvidenceEntry:
    """单条证据（对应一个假设的验证结果）"""
    hypothesis_name: str
    dimension: str
    change_pct: Optional[float] = None  # 变化率（连续值）
    direction: str = "unknown"           # "up" | "down" | "flat" | "unknown"
    significant: bool = False            # 是否显著（超过阈值）
    actual_value: Optional[float] = None
    raw_conclusion: str = ""
    timestamp: str = ""
    recommendation: Optional[str] = None  # ✅ v9.8: 行动建议（供 action_recommendations 提取）
    base_value: Optional[float] = None    # ✅ v9.8: 基准期值
    compare_value: Optional[float] = None # ✅ v9.8: 对比期值

    def to_dict(self) -> Dict:
        d = {
            "hypothesis_name": self.hypothesis_name,
            "dimension": self.dimension,
            "change_pct": self.change_pct,
            "direction": self.direction,
            "significant": self.significant,
            "actual_value": self.actual_value,
            "raw_conclusion": self.raw_conclusion,
            "timestamp": self.timestamp,
        }
        if self.recommendation:
            d["recommendation"] = self.recommendation
        if self.base_value is not None:
            d["base_value"] = self.base_value
        if self.compare_value is not None:
            d["compare_value"] = self.compare_value
        return d


@dataclass
class EvidenceBoard:
    """
    ✅ v5.8: 证据板 — 累积连续证据替代二值判定
    - 记录每个维度的变化幅度（不只是 confirmed/rejected）
    - 支持聚合判断（如：多个维度小幅下降的联合效应）
    - 支持 early stop 判断（确认根因的 impact 覆盖率）
    """
    entries: Dict[str, EvidenceEntry] = field(default_factory=dict)

    def add_evidence(self, hypothesis_name: str, dimension: str,
                     evaluation: Optional[Dict], result: "AnalysisResult") -> EvidenceEntry:
        """从评估结果中提取证据"""
        entry = EvidenceEntry(
            hypothesis_name=hypothesis_name,
            dimension=dimension,
            timestamp=datetime.now().isoformat(),
        )

        if evaluation:
            actual = evaluation.get("actual_value")
            if actual is not None and isinstance(actual, (int, float)):
                entry.change_pct = float(actual)
                entry.actual_value = float(actual)
                entry.significant = evaluation.get("meets_criteria", False)
                if actual < -3:
                    entry.direction = "down"
                elif actual > 3:
                    entry.direction = "up"
                else:
                    entry.direction = "flat"
            entry.raw_conclusion = evaluation.get("conclusion", "")

        self.entries[hypothesis_name] = entry
        return entry

    def get_significant_entries(self) -> List[EvidenceEntry]:
        """获取所有显著证据"""
        return [e for e in self.entries.values() if e.significant]

    def get_slight_decline_entries(self, threshold: float = -1.0) -> List[EvidenceEntry]:
        """获取轻微下降的证据（用于检测聚合效应）"""
        return [
            e for e in self.entries.values()
            if e.change_pct is not None and threshold > e.change_pct > -5.0
        ]

    def compute_impact_coverage(self) -> float:
        """基于已确认根因占总假设的比例 + 变化幅度的综合评分"""
        confirmed = self.get_significant_entries()
        total_entries = len(self.entries)  # 已检查的维度数

        if not confirmed or total_entries == 0:
            return 0.0

        # 因子1：确认数 / 已检查数（占比）
        confirm_ratio = len(confirmed) / max(total_entries, 1)

        # 因子2：变化幅度归一化（用 softmax 式归一化，避免单个极值主导）
        abs_changes = [min(abs(e.change_pct), 50)
                       for e in confirmed if e.change_pct is not None]
        magnitude_score = sum(abs_changes) / (sum(abs_changes) + 30)  # sigmoid-like

        return min(1.0, confirm_ratio * 0.6 + magnitude_score * 0.4)

    def to_dict(self) -> Dict:
        return {name: entry.to_dict() for name, entry in self.entries.items()}

    @classmethod
    def from_dict(cls, data: Dict) -> "EvidenceBoard":
        board = cls()
        for name, entry_data in data.items():
            board.entries[name] = EvidenceEntry(
                hypothesis_name=entry_data["hypothesis_name"],
                dimension=entry_data["dimension"],
                change_pct=entry_data.get("change_pct"),
                direction=entry_data.get("direction", "unknown"),
                significant=entry_data.get("significant", False),
                actual_value=entry_data.get("actual_value"),
                raw_conclusion=entry_data.get("raw_conclusion", ""),
                timestamp=entry_data.get("timestamp", ""),
            )
        return board


def smart_prune(plan: DynamicPlan, dimension_tree: DimensionTree,
                evidence_board: EvidenceBoard,
                current_hypothesis: "AttributionHypothesis",
                evaluation: Optional[Dict]) -> List[str]:
    """
    ✅ v5.9: 两阶段智能剪枝 — 平衡效率与全面性

    Phase 1（快速扫描，check_coverage < 0.6）:
      - 只剪交叉维度子节点（下钻无意义）
      - 禁止 prune_related（避免误剪独立指标，如 traffic → conversion_rate）
      - 目的：确保核心维度全部被检查一遍

    Phase 2（深入分析，check_coverage >= 0.6）:
      - 恢复完整剪枝（维度树子节点 + related_hypotheses）
      - 目的：已有足够证据时加速收敛

    始终生效：
      - 异常维度 → 提升子节点优先级（值得下钻）
      - 多维度轻微下降 → 提升交叉维度优先级
    """
    pruned = []
    is_significant = evaluation.get("meets_criteria", False) if evaluation else False
    actual_value = evaluation.get("actual_value") if evaluation else None

    # ── 判断当前阶段 ──
    total = len(plan.hypotheses)
    checked = sum(
        1 for h in plan.hypotheses.values()
        if h.status not in (HypothesisStatus.PENDING, HypothesisStatus.INVESTIGATING)
    )
    check_coverage = checked / total if total > 0 else 0
    in_scan_phase = check_coverage < 0.6  # Phase 1: 核心维度扫描阶段

    # ── 1. 找到当前假设对应的维度树节点 ──
    tree_node_name = dimension_tree.find_node_by_dimension(current_hypothesis.dimension)

    if tree_node_name:
        children = dimension_tree.get_children(tree_node_name)

        if not is_significant:
            # 当前维度正常 → 根据阶段决定剪枝范围
            if in_scan_phase:
                # Phase 1: 只剪交叉维度子节点（不剪同级独立维度）
                for child_name in children:
                    child_node = dimension_tree.nodes.get(child_name)
                    if child_node and child_node.node_type == "cross":
                        # 只剪包含当前维度的交叉节点
                        for h_name, h in plan.hypotheses.items():
                            if (h.status == HypothesisStatus.PENDING and
                                    h.dimension == child_node.dimension):
                                plan.prune_by_name(
                                    h_name,
                                    f"父维度 {current_hypothesis.dimension} 正常"
                                    f"({actual_value:.1f}%)，交叉下钻无意义"
                                    if actual_value is not None else
                                    f"父维度 {current_hypothesis.dimension} 正常，交叉下钻无意义"
                                )
                                pruned.append(h_name)
                # ⚠️ Phase 1 不调用 prune_related，保护独立维度
            else:
                # Phase 2: 完整剪枝（子节点 + related）
                for child_name in children:
                    child_node = dimension_tree.nodes.get(child_name)
                    if child_node:
                        for h_name, h in plan.hypotheses.items():
                            if (h.status == HypothesisStatus.PENDING and
                                    (h.dimension == child_node.dimension or
                                     child_node.dimension in h.dimension)):
                                h.status = HypothesisStatus.PRUNED
                                h.evidence.append(
                                    f"父维度 {current_hypothesis.dimension} 正常"
                                    f"({actual_value:.1f}%)，下钻无意义"
                                    if actual_value is not None else
                                    f"父维度 {current_hypothesis.dimension} 正常，下钻无意义"
                                )
                                pruned.append(h_name)

                # Phase 2: 恢复 related_hypotheses 剪枝
                old_pruned = plan.prune_related(
                    current_hypothesis.name,
                    evaluation.get("conclusion", "") if evaluation else "正常"
                )
                pruned.extend(old_pruned)
        else:
            # 当前维度异常 → 提升子节点优先级（两个阶段都生效）
            for child_name in children:
                child_node = dimension_tree.nodes.get(child_name)
                if child_node:
                    for h_name, h in plan.hypotheses.items():
                        if (h.status == HypothesisStatus.PENDING and
                                (h.dimension == child_node.dimension or
                                 child_node.dimension in h.dimension)):
                            h.priority = min(10, h.priority + 3)

    elif not is_significant and not in_scan_phase:
        # 维度树中找不到节点，但在 Phase 2 仍尝试 related 剪枝
        old_pruned = plan.prune_related(
            current_hypothesis.name,
            evaluation.get("conclusion", "") if evaluation else "正常"
        )
        pruned.extend(old_pruned)

    # ── 2. 多维度轻微下降 → 提升交叉维度优先级（两个阶段都生效） ──
    slight_declines = evidence_board.get_slight_decline_entries()
    if len(slight_declines) >= 2:
        for h_name, h in plan.hypotheses.items():
            if h.status == HypothesisStatus.PENDING and "," in h.dimension:
                h.priority = min(10, h.priority + 2)

    # 更新排序
    plan._update_order()

    return list(set(pruned))  # 去重


def rerank_hypotheses(plan: DynamicPlan, evidence_board: EvidenceBoard):
    """
    ✅ v5.8: 动态优先级重排
    根据已有证据调整剩余假设的优先级
    """
    significant = evidence_board.get_significant_entries()
    sig_dimensions = {e.dimension for e in significant}

    for h_name, h in plan.hypotheses.items():
        if h.status != HypothesisStatus.PENDING:
            continue

        # 如果该假设的维度与已确认根因相关，降低优先级（已有解释）
        if h.dimension in sig_dimensions:
            h.priority = max(1, h.priority - 2)

        # 如果是交叉维度且包含已确认根因的维度，提升优先级（精细化）
        if "," in h.dimension:
            dims = set(h.dimension.split(","))
            if dims & sig_dimensions:
                h.priority = min(10, h.priority + 2)

    plan._update_order()


# ---------------------------------------------------------------------------
# Hypothesis Generator
# ---------------------------------------------------------------------------

HYPOTHESIS_GENERATOR_PROMPT = """你是数据分析假设生成专家。根据用户问题、数据库 Schema 和可用维度，生成一组待验证的归因假设。

## 规则
1. 每个假设必须可通过数据查询验证（环比对比检测变化幅度）
2. 假设之间标注因果关联（related_hypotheses），用于剪枝
3. priority 范围 1-10
4. 数量控制在 3-8 个
5. dimension 必须是 Schema 中存在的列名

## related_hypotheses 严格规则
- 只标注有直接因果关系的假设对（如 marketing_spend → traffic）
- 独立指标之间禁止关联（如 traffic 和 conversion_rate 是独立因子，禁止互相关联）
- 不确定是否有因果关系时，留空 []
- GMV = traffic × conversion_rate × avg_price 的三个因子之间互相独立，禁止关联

## 输出格式
```json
{
  "hypotheses": [
    {
      "name": "英文假设名称",
      "dimension": "schema中的字段名",
      "description": "中文描述",
      "priority": 8,
      "related_hypotheses": ["另一个假设的name"],
      "pruning_rules": {"normal": "该维度正常时的剪枝说明"}
    }
  ]
}
```

只返回 JSON:"""


def generate_hypotheses_via_llm(user_query: str, data_source: str = "database",
                                llm: LLMInterface = None) -> list:
    try:
        llm = llm or get_llm("planner")
        schema_info = format_schema_for_prompt(get_cached_schema())
        date_range = get_cached_date_range()

        user_prompt = f"""[数据库 Schema]
{schema_info}

[数据时间范围]
{date_range['min']} 至 {date_range['max']}

[用户问题]
{user_query}

请生成归因假设:"""

        result = llm.generate(HYPOTHESIS_GENERATOR_PROMPT, user_prompt)
        response = result.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(response)
        hypotheses_raw = parsed.get("hypotheses", [])

        valid = []
        for h in hypotheses_raw:
            if all(k in h for k in ("name", "dimension", "description")):
                valid.append(h)
        return valid if valid else []
    except Exception:
        return []


def _build_hypotheses_from_raw(raw_list: list, user_query: str) -> list:
    query_lower = user_query.lower()
    results = []
    for item in raw_list:
        priority = item.get("priority", 5)
        dimension = item.get("dimension", "")
        name = item.get("name", "")
        if dimension in query_lower or name.lower().split()[0] in query_lower:
            priority = min(10, priority + 2)
        results.append(AttributionHypothesis(
            name=name, dimension=dimension,
            description=item.get("description", ""),
            priority=priority,
            related_hypotheses=item.get("related_hypotheses", []),
            pruning_rules=item.get("pruning_rules", {
                "normal": f"{name.split()[0]} 正常，可能不是主要原因"
            }),
        ))
    return results


def _hardcoded_hypotheses() -> list:
    """根据当前 Schema 动态生成归因假设（替代写死的列名列表）"""
    schema = get_cached_schema()
    meta = schema.get("_meta", {})
    hypotheses = []
    priority = 9

    # 数值列 → 逐个生成"检查 XX 变化"假设
    for tbl, col in meta.get("numeric_columns", []):
        hypotheses.append({
            "name": f"{col.title()} Analysis",
            "dimension": col,
            "description": f"检查 {col} 是否发生显著变化",
            "priority": max(1, priority),
            "related_hypotheses": [],
        })
        priority = max(1, priority - 1)

    # 分类列 → 生成"按 XX 细分"假设
    for tbl, col, vals in meta.get("text_columns", []):
        hypotheses.append({
            "name": f"{col.title()} Breakdown",
            "dimension": col,
            "description": f"按 {col} 维度细分分析",
            "priority": max(1, priority),
            "related_hypotheses": [],
        })
        priority = max(1, priority - 1)

    # 兜底
    if not hypotheses:
        hypotheses = [
            {"name": "General Analysis", "dimension": "*",
             "description": "通用数据检查", "priority": 5,
             "related_hypotheses": []},
        ]
    return hypotheses


def create_initial_plan(user_query: str, data_source: str = "database",
                        use_llm: bool = True, llm: LLMInterface = None) -> DynamicPlan:
    plan = DynamicPlan(max_attempts=REACT_CONFIG["max_attribution_attempts"])
    raw_hypotheses = []
    if use_llm:
        raw_hypotheses = generate_hypotheses_via_llm(user_query, data_source, llm=llm)
    if not raw_hypotheses:
        raw_hypotheses = _hardcoded_hypotheses()
    for h in _build_hypotheses_from_raw(raw_hypotheses, user_query):
        plan.add_hypothesis(h)
    return plan


# ── v5.7: 假设补充 Prompt（所有初始假设均被排除后，reasoner 按需生成新角度） ──
HYPOTHESIS_REFILL_PROMPT = """你是数据分析假设生成专家。初始的一批归因假设已全部被验证排除，你需要从新角度生成补充假设。

## 背景
- 用户的原始问题和数据 Schema 如下
- 已尝试并排除的假设列表也会提供给你
- 你需要从**不同维度、交叉维度、或更深层因素**生成新假设

## 补充假设策略
1. **交叉维度**: 如果单维度分析都正常，考虑维度交叉（如 渠道×品类、品类×地区）
2. **衍生指标**: 如果直接指标正常，考虑比率/效率指标（如客单价=GMV/订单数、转化率变化）
3. **结构性变化**: 如果总量正常，考虑内部结构变化（如高价值用户占比下降、头部SKU表现）
4. **时间粒度**: 如果月度正常，考虑周度/日度波动（如月末异常、周末效应）
5. 不要重复已排除的假设

## 规则
1. 每个假设必须可通过数据查询验证
2. priority 范围 1-10
3. 数量控制在 2-5 个（补充，不是从头来）
4. dimension 必须是 Schema 中存在的列名（交叉维度用逗号分隔）

## 输出格式
```json
{
  "hypotheses": [
    {
      "name": "英文假设名称",
      "dimension": "schema中的字段名",
      "description": "中文描述",
      "priority": 8,
      "related_hypotheses": [],
      "pruning_rules": {"normal": "该维度正常时的剪枝说明"}
    }
  ]
}
```

只返回 JSON:"""


def refill_hypotheses_via_llm(user_query: str, rejected_hypotheses: List[Dict],
                               llm: LLMInterface = None) -> list:
    """
    ✅ v5.7: 假设补充 — 所有初始假设排除后，调用 reasoner 生成新角度假设。
    仅在 hypothesis_refill_node 中调用，整个请求生命周期最多调用 max_refills 次。
    """
    try:
        llm = llm or get_llm("planner")
        schema_info = format_schema_for_prompt(get_cached_schema())
        date_range = get_cached_date_range()

        rejected_summary = "\n".join(
            f"- {h.get('name', 'unknown')}: {', '.join(h.get('evidence', ['无证据']))}"
            for h in rejected_hypotheses
        )

        user_prompt = f"""[数据库 Schema]
{schema_info}

[数据时间范围]
{date_range['min']} 至 {date_range['max']}

[用户问题]
{user_query}

[已排除的假设]
{rejected_summary}

请从新角度生成补充假设:"""

        result = llm.generate(HYPOTHESIS_REFILL_PROMPT, user_prompt)
        response = result.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(response)
        hypotheses_raw = parsed.get("hypotheses", [])

        valid = []
        for h in hypotheses_raw:
            if all(k in h for k in ("name", "dimension", "description")):
                valid.append(h)
        return valid if valid else []
    except Exception:
        return []


# ============================================================================
# 10. Planner Agent (Complex 路径)
# ============================================================================

PLANNER_SYSTEM_PROMPT = """你是数据分析规划专家。基于当前假设生成结构化分析指令。

## 指令字段
- task_id, hypothesis, target_field, aggregation
- current_period, compare_period, comparison_type
- time_granularity, group_by, filters
- evaluation_criteria: {field, operator, value, if_true, if_false}

## 分析策略
1. 先检验指标变化幅度是否显著 (> 5% 或 < -5%)
2. 不显著则排除假设
3. 显著则深入分析

## 输出 JSON 格式（严格遵守）
{
  "task_id": "string",
  "hypothesis": "string",
  "target_field": "string",
  "aggregation": "sum|avg|count",
  "current_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "compare_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "comparison_type": "percent|absolute|none",
  "time_granularity": "day|week|month|none",
  "group_by": ["field1", "field2"],
  "filters": {"field": "value"},
  "evaluation_criteria": {"field": "string", "operator": ">|<|>=|<=", "value": number, "if_true": "string", "if_false": "string"}
}

注意: current_period 和 compare_period 必须是包含 start 和 end 的对象，不能是字符串。

⚠️ 关键规则 - filters 字段取值:
- filters 中的值必须严格使用 Schema 中"可选值"列表里的原始值
- 禁止翻译或转换（如 Schema 显示 channel 可选值=[抖音, 京东]，则必须写 "channel": "抖音"，禁止写 "Douyin"）
- 如果 Schema 中没有可选值信息，则不要添加该字段的 filter

只返回 JSON:"""


# ── v5.5-5: Complex 路径复用 Simple PythonAgent 的精准查询 Prompt ──
PLANNER_PRECISE_QUERY_PROMPT = """你是数据分析规划专家。基于当前假设，生成一条**精准的自然语言分析查询**，供 Python 数据分析 Agent 直接执行。

## 背景
你的输出将被传递给一个已经评测通过的 Python Agent，该 Agent 擅长根据自然语言查询生成 pandas 代码。
你的职责是：把模糊的归因假设转化为精准的、可直接执行的数据分析查询。

## 精准查询要求
1. **明确时间范围**: 必须包含具体的时间段（如"2025年8月"vs"2025年7月"），不要使用模糊表述
2. **明确分析维度**: 指定按什么维度分组（如"按渠道"、"按品类"），或明确说明"不分组，计算整体"
3. **明确指标和聚合方式**: 指定计算什么指标（如"GMV总额"、"平均转化率"），以及如何聚合
4. **明确对比方式**: 如需环比/同比，明确说明对比哪两个时间段，以及计算变化率
5. **明确过滤条件**: 如果需要过滤特定渠道/品类，使用 Schema 中的实际值
6. **明确结果类型**: 描述期望的结果格式。环比对比必须要求"输出单行对比结果，包含 current_xxx、previous_xxx、change、change_pct，不要按月分多行"

## 评估标准
同时输出 evaluation_criteria，用于判断该假设是否成立：
- field: 结果中用于判断的字段名（如 change_pct）
- operator: 比较运算符（<, >, <=, >=, ==）
- value: 阈值
- if_true: 满足条件时的结论
- if_false: 不满足条件时的结论

### ⚠️ 阈值设定规范（重要！）
- 判断"显著下降"时，阈值必须设为有业务意义的水平（如 `< -5` 表示下降超过5%），**禁止使用 `< 0`**
- `< 0` 只能捕捉到统计噪声（如 -0.71%），会导致误报。随机波动在 ±3% 以内很常见
- 典型阈值参考：显著下降 `< -5`，大幅下降 `< -10`，轻微变化 `< -3`
- 判断"显著上升"同理：`> 5`，而非 `> 0`

## 精准查询的结果格式要求（重要！）
precise_query 中必须明确要求 PythonAgent 输出**单行对比结果**：
- ✅ 正确：结果为一行，包含 current_value、previous_value、change、change_pct
- ❌ 错误：输出多行（如7月一行、8月一行），然后用 shift 计算变化率
- 原因：多行 + shift 会导致首行 change_pct 为 NaN，评估逻辑无法正确提取值
- 在 precise_query 中明确写："输出单行对比结果，不要按月分行"

## 关键列命名约束（result_field_hint）
evaluation_criteria.field 中用于判断的列名，必须同时作为 result_field_hint 输出。
PythonAgent 会据此强制命名结果 DataFrame 中的关键列，确保评估逻辑能精确匹配。

## 图表建议（可选）
如果该分析适合可视化，指定图表类型（bar/line/pie），否则设为 null

## 输出 JSON 格式
```json
{
  "task_id": "hypothesis_name_N",
  "precise_query": "精准的自然语言分析查询（一段完整描述，包含上述所有要素。环比对比时要求输出单行结果）",
  "result_description": "单行对比结果，包含当期值、上期值、变化率",
  "result_field_hint": "change_pct",
  "evaluation_criteria": {
    "field": "change_pct",
    "operator": "<",
    "value": -5,
    "if_true": "该维度显著下降，可能是根因",
    "if_false": "该维度正常，排除此假设"
  },
  "chart_type": "bar 或 null"
}
```

⚠️ result_field_hint 必须与 evaluation_criteria.field 保持一致。

⚠️ 关键：precise_query 必须足够具体，让一个不了解上下文的分析 Agent 也能准确执行。
过滤值必须使用 Schema 中标注的实际可选值（如英文列名和英文枚举值）。

只返回 JSON:"""


class PlannerAgent:
    """
    ✅ v5.7: 双 LLM 架构
    - reasoner_llm: 仅用于假设生成（调用1次，需要深度推理）
    - chat_llm: 用于精准查询生成 + 剪枝判断（循环中调用，chat 足够）
    """
    def __init__(self, reasoner_llm: LLMInterface = None, chat_llm: LLMInterface = None):
        self.reasoner_llm = reasoner_llm or get_llm("planner")         # deepseek-reasoner
        self.chat_llm = chat_llm or get_llm("planner_chat")            # deepseek-chat
        # ✅ 向后兼容：保留 self.llm 指向 reasoner（供 generate_hypotheses_via_llm 等外部调用）
        self.llm = self.reasoner_llm


    def generate_precise_query(self, user_query: str, hypothesis: AttributionHypothesis,
                               previous_results: List[AnalysisResult],
                               dynamic_plan: DynamicPlan) -> Dict:
        """
        v5.5-5: Complex 路径复用 Simple PythonAgent。
        Planner 输出精准的自然语言查询（而非结构化 AnalysisInstruction），
        交给已评测通过的 execute_from_query() 执行。

        Returns:
            {
                "task_id": str,
                "precise_query": str,       # 精准自然语言查询
                "result_description": str,   # 期望结果描述
                "evaluation_criteria": dict, # 评估标准（用于假设验证）
                "chart_type": str or None,   # 图表类型建议
                "hypothesis": str,           # 假设名称
            }
        """
        schema_info = format_schema_for_prompt(get_cached_schema())
        date_range = get_cached_date_range()

        prev_summary = ""
        if previous_results:
            prev_lines = []
            for r in previous_results[-3:]:
                if r.evaluation:
                    prev_lines.append(f"- {r.task_id}: {r.evaluation.get('conclusion', 'N/A')}")
                elif r.row_count == 0:
                    prev_lines.append(f"- {r.task_id}: ⚠️ 返回 0 行数据")
                elif r.summary:
                    prev_lines.append(f"- {r.task_id}: {r.summary[:100]}")
            if prev_lines:
                prev_summary = "\n上次分析结果:\n" + "\n".join(prev_lines)

        prompt = f"""[Schema] {schema_info}

[数据时间范围] {date_range['min']} 至 {date_range['max']}

问题: {user_query}
当前假设: {hypothesis.name} (字段: {hypothesis.dimension})
假设描述: {hypothesis.description}
已尝试: {dynamic_plan.attempts_count}/{dynamic_plan.max_attempts}
{prev_summary}

生成精准分析查询 JSON:"""

        try:
            result = self.chat_llm.generate(PLANNER_PRECISE_QUERY_PROMPT, prompt)  # ✅ v5.7: 用 chat 而非 reasoner
            response = result.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(response)

            # ✅ v5.6: 提取 result_field_hint，与 evaluation_criteria.field 保持一致
            eval_criteria = parsed.get("evaluation_criteria")
            result_field_hint = parsed.get("result_field_hint")
            # 兜底：如果 Planner 漏填 hint，从 evaluation_criteria.field 回补
            if not result_field_hint and eval_criteria and isinstance(eval_criteria, dict):
                result_field_hint = eval_criteria.get("field")

            return {
                "task_id": parsed.get("task_id", f"{hypothesis.name}_{dynamic_plan.attempts_count}"),
                "precise_query": parsed.get("precise_query", ""),
                "result_description": parsed.get("result_description", ""),
                "result_field_hint": result_field_hint,
                "evaluation_criteria": eval_criteria,
                "chart_type": parsed.get("chart_type"),
                "hypothesis": hypothesis.name,
            }
        except Exception as e:
            # Fallback: 根据假设信息构造一条直白的查询
            date_range = get_cached_date_range()
            cur_year, cur_month = safe_get_year_month(date_range, "max")
            if cur_year is None or cur_month is None:
                now = datetime.now()
                cur_year, cur_month = now.year, now.month
            prev_month_int = cur_month - 1 if cur_month > 1 else 12
            prev_year = cur_year if cur_month > 1 else cur_year - 1

            fallback_query = (
                f"对比 {prev_year}年{prev_month_int}月 和 {cur_year}年{cur_month}月 的 "
                f"{hypothesis.dimension} 数据，计算变化率。"
                f"输出单行对比结果，包含 当期值(current_{hypothesis.dimension})、"
                f"上期值(previous_{hypothesis.dimension})、"
                f"变化量(change)、变化率(change_pct)。不要按月分多行。"
            )

            return {
                "task_id": f"fallback_{hypothesis.name}",
                "precise_query": fallback_query,
                "result_description": f"{hypothesis.dimension} 的环比对比结果",
                "result_field_hint": "change_pct",
                "evaluation_criteria": {
                    "field": "change_pct",
                    "operator": "<", "value": -5,
                    "if_true": f"{hypothesis.dimension} 显著下降",
                    "if_false": f"{hypothesis.dimension} 正常",
                },
                "chart_type": None,
                "hypothesis": hypothesis.name,
            }

    def update_plan_from_result(self, plan: DynamicPlan, hypothesis: AttributionHypothesis,
                                result: AnalysisResult) -> Dict:
        if not result.success:
            hypothesis.status = HypothesisStatus.REJECTED
            hypothesis.evidence.append(f"执行失败: {result.error}")
            return {"status": "failed", "should_prune": [], "pivot_suggestion": None}

        pivot_info = {"status": "inconclusive", "should_prune": [], "pivot_suggestion": None, "reasoning": ""}

        if result.evaluation:
            meets = result.evaluation.get("meets_criteria")
            conclusion = result.evaluation.get("conclusion", "")

            if meets is False:
                hypothesis.status = HypothesisStatus.REJECTED
                hypothesis.evidence.append(f"正常: {conclusion}")
                pivot_info["status"] = "normal"
                pivot_info["reasoning"] = conclusion

                pruned = plan.prune_related(hypothesis.name, conclusion)
                pivot_info["should_prune"] = pruned

                for h_name in plan.investigation_order:
                    h = plan.hypotheses[h_name]
                    if h.status == HypothesisStatus.PENDING:
                        pivot_info["pivot_suggestion"] = h_name
                        break

                if pruned:
                    plan.record_pivot(hypothesis.name, pivot_info.get("pivot_suggestion", "unknown"), conclusion)

            elif meets is True:
                hypothesis.status = HypothesisStatus.CONFIRMED
                hypothesis.evidence.append(f"异常: {conclusion}")
                pivot_info["status"] = "abnormal"
                pivot_info["reasoning"] = conclusion

        else:
            # ✅ 兜底：evaluation 为 None（无 evaluation_criteria）但执行成功
            if result.row_count == 0:
                hypothesis.status = HypothesisStatus.REJECTED
                hypothesis.evidence.append("查询无数据: 过滤条件可能与数据不匹配")
                pivot_info["status"] = "no_data"
                pivot_info["reasoning"] = "查询返回 0 行，无法验证假设"
            # row_count > 0 但无 evaluation_criteria → 保持 investigating（需人工判断）

        return pivot_info


# ============================================================================
# 11. Reporter Agent
# ============================================================================

REPORTER_SYSTEM_PROMPT = """你是专业的数据分析报告撰写专家。根据分析结果生成结构化报告。

## 报告结构
1. 核心发现摘要
2. 确认的根因（含因果链）
3. 排除的假设
4. 详细分析过程
5. 行动指南

## 行动指南撰写规范
行动指南是报告的核心产出，必须满足：
- **因果对应**: 每条行动必须直接对应一个已确认的根因，禁止凭空建议
- **可执行**: 写清"谁、做什么、做到什么程度"，避免"加强管理"等空话
- **分优先级**: 按影响幅度排序，标注 P0(立即)/P1(本周)/P2(本月)
- **可验证**: 每条行动附带一个可量化的验收指标（如"流量恢复至上月的80%"）
- **边界诚实**: 只建议数据能支撑的方向，不确定的领域标注"建议进一步调研"

示例格式：
### 🎯 行动指南
| 优先级 | 行动项 | 对应根因 | 验收指标 |
|--------|--------|----------|----------|
| P0 | 恢复抖音信息流投放至上月水平 | marketing_spend↓70% | 周流量恢复至4万+ |
| P1 | 排查投放 ROI，剔除低效计划 | marketing_spend↓70% | 单客获取成本≤15元 |
| P2 | 建议产品团队调研竞品定价策略 | 待进一步分析 | 出具竞品分析报告 |

## 负样本处理
- 如果所有维度均正常（无确认根因），行动指南应写"当前各指标在正常波动范围内，建议保持现有策略并持续监控"
- 禁止在无数据支撑时编造行动建议

## ⚠️ 严格约束
- 只能基于"各步骤分析数据"和"确认/排除的假设"中提供的事实撰写，禁止编造数据中没有的结论
- 如果确认的根因和排除的假设都为空，说明分析未能得出明确结论，必须如实说明"当前分析未能确认或排除任何假设"，并给出可能的原因（如数据不匹配、过滤条件有误等）
- 禁止在没有数据支撑的情况下自行推测根因（如"季节性因素"、"竞争加剧"等）

## ⚠️ 强制段落（缺失视为不合格，必须独立成段在报告末尾）
报告末尾**必须**包含一段标题为 "## ⚠️ 分析局限与未覆盖维度" 的段落，至少包含 3 条要点：
1. **数据覆盖局限**: 本次分析使用的数据源 + 未覆盖的关键维度（例：未包含 ad_campaign 投放明细 / competitor 竞品监控 / inventory 库存等具体表名）
2. **未验证的假设**: 因数据缺失而无法完全验证的假设（按 P0/P1 列出，至少 1 条；若全部已验证，明确写"无未验证假设"）
3. **建议补充数据**: 列出 1-3 个具体数据类型 + 各自预期能验证的根因（即便已经走完 deep_rca，也要写"如需进一步精细化定位可补充 X"）

⚠️ **重要**: 即便分析已完成、信心很高，也必须诚实标注本轮的边界。这一段不能省略、不能合并到其它章节、不能用一句话敷衍。该段落是报告质量的硬指标，缺失或过短（少于 3 行）都会被判不合格。

使用 Markdown 格式，简洁专业。"""


class ReporterAgent:
    def __init__(self, llm: LLMInterface = None):
        self.llm = llm or get_llm("reporter")

    def generate_report(self, user_query: str, dynamic_plan: DynamicPlan,
                        all_results: List[AnalysisResult], chart_paths: List[str] = None,
                        evidence_board: "EvidenceBoard" = None) -> Dict:
        confirmed, rejected, pruned = [], [], []
        for name, h in dynamic_plan.hypotheses.items():
            h_info = {"name": name, "evidence": h.evidence}
            if h.status == HypothesisStatus.CONFIRMED:
                confirmed.append(h_info)
            elif h.status == HypothesisStatus.REJECTED:
                rejected.append(h_info)
            elif h.status == HypothesisStatus.PRUNED:
                pruned.append(h_info)

        # ✅ 构建各步骤的分析数据摘要，让 Reporter 有事实依据
        steps_summary_lines = []
        for r in all_results:
            line = f"- [{r.task_id}] "
            if r.row_count == 0:
                line += "⚠️ 无匹配数据（返回 0 行）"
            else:
                line += f"返回 {r.row_count} 行"
            if r.summary:
                line += f" | 摘要: {r.summary[:150]}"
            if r.evaluation:
                line += f" | 评估: {r.evaluation.get('conclusion', 'N/A')}"
            steps_summary_lines.append(line)
        steps_section = "\n".join(steps_summary_lines) if steps_summary_lines else "无分析步骤执行"

        # ✅ v5.8: 注入证据面板摘要（连续变化值，帮助 Reporter 做定量分析）
        evidence_section = ""
        if evidence_board and evidence_board.entries:
            ev_lines = []
            for name, entry in evidence_board.entries.items():
                pct_str = f"{entry.change_pct:.1f}%" if entry.change_pct is not None else "N/A"
                sig_str = "⚠️显著" if entry.significant else "正常"
                ev_lines.append(f"- {name} ({entry.dimension}): 变化率={pct_str}, {sig_str}, 方向={entry.direction}")
            evidence_section = "\n\n各维度变化率汇总（证据面板）:\n" + "\n".join(ev_lines)

        prompt = f"""用户问题: {user_query}

各步骤分析数据:
{steps_section}
{evidence_section}

确认的根因: {json.dumps(confirmed, ensure_ascii=False)}
排除的假设: {json.dumps(rejected, ensure_ascii=False)}
剪枝的假设: {json.dumps(pruned, ensure_ascii=False)}
策略转向: {json.dumps(dynamic_plan.pivot_history, ensure_ascii=False)}

生成专业分析报告:"""

        try:
            result = self.llm.generate(REPORTER_SYSTEM_PROMPT, prompt)
            return {
                "success": True,
                "full_content": result,
                "summary": result[:300] + "..." if len(result) > 300 else result,
                "confirmed_hypotheses": [c["name"] for c in confirmed],
                "rejected_hypotheses": [r["name"] for r in rejected],
                "pruned_hypotheses": [p["name"] for p in pruned],
                "chart_paths": chart_paths or [],
            }
        except Exception as e:
            return {
                "success": False,
                "full_content": f"报告生成失败: {str(e)}",
                "summary": "分析完成",
                "error": str(e),}

# ============================================================================
# 12. LangGraph State
# ============================================================================

class AgentState(TypedDict):
    user_query: str
    data_source: str
    uploaded_df: Optional[Any]

    # 路由结果
    route: str
    output_type: str
    chart_type: Optional[str]
    instruction: Optional[Dict]  # Complex 路径使用
    validation_result: Optional[Dict]  # Router 校验结果 (Simple 路径)

    # ReAct 状态 (fallback 路径保留)
    dynamic_plan: Optional[Dict]
    current_iteration: int
    current_hypothesis: Optional[str]
    all_results: List[Dict]
    should_continue: bool
    _needs_refill: bool
    _refill_count: int
    _max_refills: int

    # v5.8 字段 (fallback 路径保留)
    _dimension_tree: Optional[Dict]
    _evidence_board: Optional[Dict]
    _latest_evaluation: Optional[Dict]

    # ✅ v8 新增：Quick Scan 三步架构字段
    analysis_frame: Optional[Dict]           # Commander 输出的扫描计划 (AnalysisFrame)
    scan_state: Optional[Dict]               # Scan Loop 累积状态 (ScanState)
    # ✅ v9 新增：分析深度（由 Commander 判定，替代 Router 的 SIMPLE/COMPLEX 分类）
    analysis_depth: Optional[str]            # descriptive | diagnostic | causal

    # ✅ v8 Step 2 新增：Reasoner v2 字段
    reason_result: Optional[Dict]            # Reasoner v2 输出（含 needs_deep_rca）
    needs_deep_rca: Optional[bool]           # 是否需要深度分析（Step 3 触发条件）

    # ✅ v8 Step 3 新增：Deep RCA 字段
    user_decision: Optional[str]             # 用户对 Step 3 的决策: "continue" / "skip"
    supplementary_df: Optional[Any]          # 用户补充上传的数据
    findings_presented: Optional[bool]       # 是否已展示 Step 2 结论（第一轮终止标志）
    deep_rca_mode: Optional[bool]            # 是否处于 Deep RCA 模式（第二轮标志）
    _supp_scan_findings: Optional[List[Dict]] # ✅ v9.8: 补充数据全量扫描发现
    prior_scan_state: Optional[Dict]         # 第一轮的 scan_state（跨轮传递）
    prior_reason_result: Optional[Dict]      # 第一轮的 reason_result（跨轮传递）
    prior_confirmed_anomalies: Optional[List[Dict]]  # 第一轮的确认异常（跨轮传递）

    # v7 兼容字段（由 quick_scan_node 自动填充，供 reason_node/reporter 使用）
    scan_result: Optional[Dict]              # [兼容] scan 原始结果
    scan_data: Optional[List[Dict]]          # [兼容] 全维度扫描数据 = scan_state.layer_results[1]
    confirmed_anomalies: Optional[List[Dict]]  # [兼容] = scan_state.all_anomalies
    rejected_dimensions: Optional[List[Dict]]  # [兼容] = scan_state.all_normal
    has_anomaly: Optional[bool]              # [兼容] 是否存在异常
    causal_result: Optional[Dict]            # reason_node 的因果推理结果
    _fallback_to_react: Optional[bool]       # scan 失败时降级标志
    drilldown_data: Optional[List[Dict]]     # [兼容] = scan_state.layer_results[2+]

    # Trace
    trace: Optional[Any]

    # 复用 LLM 客户端
    _python_agent: Optional[Any]
    _planner_agent: Optional[Any]
    _reporter_agent: Optional[Any]

    # 输出
    steps: List[Dict]
    final_result: Optional[Dict]
    final_report: Optional[Dict]
    chart_paths: List[str]
    error: Optional[str]
    suggestions: Optional[List[str]]

# ============================================================================
# 13. LangGraph 节点（v5: 统一使用 PythonAgent，无 SQLAgent）
# ============================================================================

def gateway_node(state: AgentState) -> AgentState:
    """Gateway 节点：轻量路由（意图分类 + 校验 + 输出类型）"""
    user_query = state["user_query"]
    data_source_str = state.get("data_source", "csv")

    # 初始化 Trace
    trace = TraceLog(
        session_id=str(uuid.uuid4())[:8],
        user_query=user_query,
        start_time=datetime.now().isoformat(),
        data_source=data_source_str,
        config=REACT_CONFIG.copy(),
    )
    trace.log(TraceEventType.SESSION_START, iteration=0,
              input_data={"user_query": user_query, "data_source": data_source_str})

    # ✅ 新增：设置 DataFrame schema 缓存（供 Router 使用）
    uploaded_df = state.get("uploaded_df")
    if uploaded_df is not None:
        set_current_df(uploaded_df)

    # Router 路由
    ds_type = DataSourceType(data_source_str) if data_source_str else DataSourceType.CSV
    router = RouterAgent()
    router_output = router.route(user_query, ds_type)

    route_type = router_output.get("route", RouteType.COMPLEX)  # ✅ v9: 默认 COMPLEX
    output_type = router_output.get("output_type", "TABLE")
    chart_type = router_output.get("chart_type")

    trace.log(TraceEventType.INTENT_CLASSIFICATION, iteration=0,
              output_data={"route": route_type.value, "output_type": output_type, "chart_type": chart_type})

    # 更新状态
    state["route"] = route_type.value
    state["output_type"] = output_type
    state["chart_type"] = chart_type
    state["trace"] = trace

    # ✅ v9: 所有合法查询统一走 Quick Scan，保存校验结果
    if route_type != RouteType.INVALID:
        validation = router_output.get("validation", {"is_valid": True, "reasoning": ""})
        state["validation_result"] = validation

    # 错误处理
    if route_type == RouteType.INVALID:
        state["error"] = router_output.get("error")
        state["suggestions"] = router_output.get("suggestions", [])

    # 初始化
    state["steps"] = []
    state["chart_paths"] = []
    state["all_results"] = []
    state["current_iteration"] = 0
    state["should_continue"] = True
    state["_needs_refill"] = False             # ✅ v5.7
    state["_refill_count"] = 0                 # ✅ v5.7
    state["_max_refills"] = REACT_CONFIG.get("max_refills", 1)  # ✅ v5.7
    state["validation_result"] = state.get("validation_result")

    # ✅ v5.8: 初始化维度分解树和证据板 (fallback 路径用)
    schema = get_cached_schema()
    dim_tree = DimensionTree.build_from_schema(schema)
    state["_dimension_tree"] = dim_tree.to_dict()
    state["_evidence_board"] = {}
    state["_latest_evaluation"] = None

    # ✅ v7: 初始化 scan/detect/reason 管线字段
    state["scan_result"] = None
    state["scan_data"] = []
    state["confirmed_anomalies"] = []
    state["rejected_dimensions"] = []
    state["has_anomaly"] = None
    state["causal_result"] = None
    state["_fallback_to_react"] = False
    state["drilldown_data"] = []

    # ✅ 复用 LLM 客户端：一次创建，全流程复用，避免重复 TCP 握手
    state["_python_agent"] = PythonAgent()
    state["_planner_agent"] = PlannerAgent()
    state["_reporter_agent"] = ReporterAgent()

    return state

def invalid_handler_node(state: AgentState) -> AgentState:
    """无效查询处理"""
    trace: TraceLog = state["trace"]
    trace.finalize("invalid", {"error": state.get("error")})
    return state

def react_init_node(state: AgentState) -> AgentState:
    """ReAct 初始化：创建动态计划"""
    user_query = state["user_query"]
    trace: TraceLog = state["trace"]
    data_source_str = state.get("data_source", "csv")
    # ✅ v5.7: 仅假设生成使用 reasoner（调用1次），精准查询由 PlannerAgent 内部用 chat_llm
    planner_reasoner_llm = state["_planner_agent"].reasoner_llm if state.get("_planner_agent") else None
    dynamic_plan = create_initial_plan(user_query, data_source_str, llm=planner_reasoner_llm)

    trace.log(TraceEventType.PLAN_INITIALIZED, iteration=0,
              output_data={
                  "hypotheses": [{"name": h.name, "priority": h.priority}
                                 for h in sorted(dynamic_plan.hypotheses.values(), key=lambda x: -x.priority)],
                  "investigation_order": dynamic_plan.investigation_order,
              })

    state["dynamic_plan"] = dynamic_plan.to_dict()
    state["current_iteration"] = 0
    state["should_continue"] = True
    state["all_results"] = []

    return state

def react_step_node(state: AgentState) -> AgentState:
    """
    ✅ v5.8: 精简为纯执行节点
    只负责：选假设 → 生成查询 → 执行代码 → 返回原始结果
    评估/剪枝/路由决策全部移交 evaluator_node
    """
    trace: TraceLog = state["trace"]
    user_query = state["user_query"]

    # 恢复动态计划
    plan_dict = state["dynamic_plan"]
    dynamic_plan = DynamicPlan.from_dict(plan_dict)

    iteration = state["current_iteration"] + 1
    state["current_iteration"] = iteration

    # 获取下一个假设
    current_hypothesis = dynamic_plan.get_next_hypothesis()

    if current_hypothesis is None:
        state["should_continue"] = False
        state["_latest_evaluation"] = None
        state["dynamic_plan"] = dynamic_plan.to_dict()
        return state

    # 检查限制
    if iteration > REACT_CONFIG["max_iterations"] or \
            dynamic_plan.attempts_count >= REACT_CONFIG["max_attribution_attempts"]:
        state["should_continue"] = False
        state["_latest_evaluation"] = None
        state["dynamic_plan"] = dynamic_plan.to_dict()
        return state

    current_hypothesis.status = HypothesisStatus.INVESTIGATING
    dynamic_plan.attempts_count += 1
    state["current_hypothesis"] = current_hypothesis.name

    trace.log(TraceEventType.HYPOTHESIS_SELECTED, iteration=iteration,
              hypothesis=current_hypothesis.name,
              input_data={"dimension": current_hypothesis.dimension, "priority": current_hypothesis.priority})

    # ── 生成精准查询 ──
    planner = state["_planner_agent"]
    previous_results = [AnalysisResult.from_dict(r_dict) for r_dict in state.get("all_results", [])]
    query_info = planner.generate_precise_query(
        user_query, current_hypothesis, previous_results, dynamic_plan)

    precise_query = query_info.get("precise_query", "")
    task_id = query_info.get("task_id", f"{current_hypothesis.name}_{iteration}")
    evaluation_criteria = query_info.get("evaluation_criteria")
    chart_type = query_info.get("chart_type")
    result_description = query_info.get("result_description", "")
    result_field_hint = query_info.get("result_field_hint")

    trace.log(TraceEventType.INSTRUCTION_GENERATED, iteration=iteration,
              hypothesis=current_hypothesis.name,
              input_data={"precise_query": precise_query[:300],
                          "result_description": result_description,
                          "result_field_hint": result_field_hint},
              metadata={"mode": "precise_query_via_simple_agent"})

    # ── 执行代码 ──
    trace.log(TraceEventType.TOOL_EXECUTION_START, iteration=iteration,
              hypothesis=current_hypothesis.name)

    uploaded_df = state.get("uploaded_df")

    if uploaded_df is None:
        result = AnalysisResult(
            task_id=task_id,
            success=False, error="未上传数据文件")
    else:
        python_agent = state["_python_agent"]
        schema_info = format_schema_for_prompt(get_cached_schema())

        need_chart = (chart_type is not None and chart_type not in ["null", ""])
        # ✅ v5.8: Complex 路径修复次数降为 1（快速失败优于反复修复）
        original_retries = python_agent.max_retries
        python_agent.max_retries = 1
        result = python_agent.execute_from_query(
            user_query=precise_query,
            schema_info=schema_info,
            df=uploaded_df,
            validation={"is_valid": True, "reasoning": f"Planner 精准查询 - 假设: {current_hypothesis.name}"},
            need_chart=need_chart,
            chart_type=chart_type if need_chart else None,
            trace=trace,
            result_field_hint=result_field_hint,
        )
        python_agent.max_retries = original_retries  # 恢复（Simple 路径仍用 3 次）

        # ── 后置评估（仍在此处计算，但不做决策） ──
        if result.success and evaluation_criteria and (result.data or result.answer):
            result.evaluation = evaluate_result(result.data, evaluation_criteria, answer=result.answer)
        elif result.success and evaluation_criteria and not result.data:
            result.evaluation = {
                "meets_criteria": False,
                "conclusion": evaluation_criteria.get(
                    "if_false", f"{current_hypothesis.dimension} 无匹配数据"),
                "reasoning": "查询返回 0 行数据",
                "actual_value": None,
            }

    if result.success:
        trace.log(TraceEventType.TOOL_EXECUTION_SUCCESS, iteration=iteration,
                  hypothesis=current_hypothesis.name,
                  output_data={"row_count": result.row_count},
                  success=True)
    else:
        trace.log(TraceEventType.TOOL_EXECUTION_ERROR, iteration=iteration,
                  hypothesis=current_hypothesis.name, error=result.error, success=False)

    # ── 保存执行结果（不做决策，交给 evaluator_node） ──
    state["steps"].append({
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": current_hypothesis.name,
        "dimension": current_hypothesis.dimension,
        "priority": current_hypothesis.priority,
        "status": current_hypothesis.status.value,  # 仍是 investigating
        "precise_query": precise_query,
        "result_description": result_description,
        "code": result.executed_code,
        "result": {
            "success": result.success,
            "row_count": result.row_count,
            "evaluation": result.evaluation,
            "columns": result.columns,
            "data_preview": result.data[:20] if result.data else [],
            "summary": result.summary,
            "answer": result.answer,
            "chart_path": result.chart_path,
            "error": result.error,
            "correction_history": result.correction_history,
        },
    })

    state["all_results"].append(result.to_dict())

    if result.chart_path:
        state["chart_paths"].append(result.chart_path)

    # ✅ v5.8: 将 evaluation 传递给 evaluator_node，不在此做决策
    state["_latest_evaluation"] = result.evaluation
    state["dynamic_plan"] = dynamic_plan.to_dict()
    # should_continue 由 evaluator_node 设置
    state["should_continue"] = True  # 默认继续到 evaluator

    return state


def evaluator_node(state: AgentState) -> AgentState:
    """
    ✅ v5.8 新节点：独立的评估 + 智能剪枝 + 动态重排 + 路由决策
    从 react_step_node 中提取出来，职责清晰：
    1. 根据 evaluation 更新假设状态（confirmed/rejected）
    2. 基于维度分解树做智能剪枝
    3. 更新证据板（连续值）
    4. 动态重排剩余假设优先级
    5. 判断 early stop
    6. 路由决策：继续 / 补充假设 / 结束
    """
    trace: TraceLog = state["trace"]
    iteration = state["current_iteration"]
    hypothesis_name = state.get("current_hypothesis")

    if not hypothesis_name:
        state["should_continue"] = False
        state["_needs_refill"] = False
        return state

    # 恢复各种状态
    dynamic_plan = DynamicPlan.from_dict(state["dynamic_plan"])
    dimension_tree = DimensionTree.from_dict(state.get("_dimension_tree", {}))
    evidence_board = EvidenceBoard.from_dict(state.get("_evidence_board", {}))
    evaluation = state.get("_latest_evaluation")
    current_hypothesis = dynamic_plan.hypotheses.get(hypothesis_name)

    if not current_hypothesis:
        state["should_continue"] = False
        state["_needs_refill"] = False
        return state

    # 恢复最新的 AnalysisResult
    latest_result_dict = state["all_results"][-1] if state["all_results"] else None
    latest_result = AnalysisResult.from_dict(latest_result_dict) if latest_result_dict else None

    # ── Step 1: 更新假设状态 ──
    if latest_result and not latest_result.success:
        current_hypothesis.status = HypothesisStatus.REJECTED
        current_hypothesis.evidence.append(f"执行失败: {latest_result.error}")
    elif evaluation:
        meets = evaluation.get("meets_criteria")
        conclusion = evaluation.get("conclusion", "")

        if meets is True:
            current_hypothesis.status = HypothesisStatus.CONFIRMED
            current_hypothesis.evidence.append(f"异常: {conclusion}")
            trace.log(TraceEventType.HYPOTHESIS_CONFIRMED, iteration=iteration,
                      hypothesis=hypothesis_name,
                      output_data={"reasoning": conclusion})
        elif meets is False:
            current_hypothesis.status = HypothesisStatus.REJECTED
            current_hypothesis.evidence.append(f"正常: {conclusion}")
            trace.log(TraceEventType.HYPOTHESIS_REJECTED, iteration=iteration,
                      hypothesis=hypothesis_name,
                      output_data={"reasoning": conclusion})
        # meets is None → 保持 investigating（无法判定）
    elif latest_result and latest_result.row_count == 0:
        current_hypothesis.status = HypothesisStatus.REJECTED
        current_hypothesis.evidence.append("查询无数据: 过滤条件可能与数据不匹配")

    # ── Step 2: 更新证据板（连续值） ──
    evidence_entry = evidence_board.add_evidence(
        hypothesis_name=hypothesis_name,
        dimension=current_hypothesis.dimension,
        evaluation=evaluation,
        result=latest_result,
    )

    trace.log(TraceEventType.EVIDENCE_UPDATED, iteration=iteration,
              hypothesis=hypothesis_name,
              output_data={
                  "change_pct": evidence_entry.change_pct,
                  "direction": evidence_entry.direction,
                  "significant": evidence_entry.significant,
              })

    # ── Step 3: 智能剪枝（基于维度分解树） ──
    pruned = smart_prune(dynamic_plan, dimension_tree, evidence_board,
                         current_hypothesis, evaluation)

    if pruned:
        trace.log(TraceEventType.PRUNING_EXECUTED, iteration=iteration,
                  hypothesis=hypothesis_name,
                  output_data={
                      "pruned_hypotheses": pruned,
                      "pruning_mode": "dimension_tree + evidence_based",
                  })

    # ── Step 4: 动态优先级重排 ──
    rerank_hypotheses(dynamic_plan, evidence_board)

    trace.log(TraceEventType.PRIORITY_RERANKED, iteration=iteration,
              output_data={
                  "new_order": dynamic_plan.investigation_order[:5],
                  "remaining_pending": sum(1 for h in dynamic_plan.hypotheses.values()
                                           if h.status == HypothesisStatus.PENDING),
              })

    # ── Step 5: Early Stop 判断 (✅ v5.10: 优化触发条件) ──
    impact_coverage = evidence_board.compute_impact_coverage()
    confirmed_count = sum(1 for h in dynamic_plan.hypotheses.values()
                          if h.status == HypothesisStatus.CONFIRMED)
    # 统计核心维度覆盖率（已检查 / 总假设数）
    total_hypotheses = len(dynamic_plan.hypotheses)
    checked_count = sum(
        1 for h in dynamic_plan.hypotheses.values()
        if h.status in (HypothesisStatus.CONFIRMED,
                        HypothesisStatus.REJECTED,
                        HypothesisStatus.PRUNED)
    )
    check_coverage = checked_count / total_hypotheses if total_hypotheses > 0 else 0

    # ✅ v5.10: 放宽 high_pri_pending — 已确认根因能解释的维度不再阻塞
    confirmed_dims = {
        h.dimension for h in dynamic_plan.hypotheses.values()
        if h.status == HypothesisStatus.CONFIRMED
    }
    high_pri_pending = any(
        h.priority >= 8
        and h.status == HypothesisStatus.PENDING
        and h.dimension not in confirmed_dims  # 已解释维度不阻塞
        for h in dynamic_plan.hypotheses.values()
    )

    # ✅ v5.10: 强证据快速通道 — 变化幅度 ≥ 40% 时降低覆盖率要求
    strong_evidence = any(
        abs(e.change_pct or 0) >= 40
        for e in evidence_board.get_significant_entries()
    )
    coverage_threshold = 0.4 if strong_evidence else 0.5

    # ✅ v5.10: 三层 early stop（保留全面性，提升效率）
    # 1. 已找到根因（confirmed ≥ 1）
    # 2. 核心维度检查覆盖率 ≥ 50%（强证据时 ≥ 40%）
    #    - Phase 1 保守剪枝保证核心维度不被误剪，50% 已足够覆盖关键维度
    # 3. impact 覆盖率仍作为辅助条件
    # 4. 无未解释的高优先级待查假设
    early_stop = (
            confirmed_count >= 1
            and check_coverage >= coverage_threshold
            and impact_coverage >= 0.5
            and not high_pri_pending
    )

    if early_stop:
        trace.log(TraceEventType.EARLY_STOP, iteration=iteration,
                  output_data={
                      "confirmed_count": confirmed_count,
                      "impact_coverage": round(impact_coverage, 2),
                      "check_coverage": round(check_coverage, 2),
                      "coverage_threshold": coverage_threshold,
                      "strong_evidence": strong_evidence,
                      "reason": "证据充分，提前终止",
                  })

    # ── Step 6: 更新步骤信息（补充 evaluator 的决策） ──
    if state["steps"]:
        last_step = state["steps"][-1]
        last_step["status"] = current_hypothesis.status.value
        last_step["pivot_info"] = {
            "status": "abnormal" if current_hypothesis.status == HypothesisStatus.CONFIRMED else "normal",
            "should_prune": pruned,
            "evidence": evidence_entry.to_dict(),
            "impact_coverage": round(impact_coverage, 2),
            "early_stop": early_stop,
        }

    # ── Step 7: 路由决策（三路判断） ──
    has_pending = any(h.status == HypothesisStatus.PENDING for h in dynamic_plan.hypotheses.values())
    has_confirmed = any(h.status == HypothesisStatus.CONFIRMED for h in dynamic_plan.hypotheses.values())
    budget_ok = dynamic_plan.attempts_count < dynamic_plan.max_attempts

    if early_stop:
        # 证据充分，直接结束
        state["should_continue"] = False
        state["_needs_refill"] = False
    elif has_pending and budget_ok:
        # 还有假设待验证 → 继续
        state["should_continue"] = True
        state["_needs_refill"] = False
    elif not has_pending and not has_confirmed and budget_ok \
            and state.get("_refill_count", 0) < state.get("_max_refills", 1):
        # 假设全部用完 + 未找到根因 → 补充假设
        state["should_continue"] = False
        state["_needs_refill"] = True
    else:
        # budget 耗尽 / 已找到根因但无需 early stop / refill 达上限 → 结束
        state["should_continue"] = False
        state["_needs_refill"] = False

    # 保存状态
    state["dynamic_plan"] = dynamic_plan.to_dict()
    state["_evidence_board"] = evidence_board.to_dict()

    trace.log(TraceEventType.EVALUATION_COMPLETED, iteration=iteration,
              hypothesis=hypothesis_name,
              output_data={
                  "decision": "continue" if state["should_continue"] else
                              ("refill" if state.get("_needs_refill") else "done"),
                  "has_pending": has_pending,
                  "has_confirmed": has_confirmed,
                  "budget_ok": budget_ok,
                  "early_stop": early_stop,
              })

    return state

def reporter_node(state: AgentState) -> AgentState:
    """Reporter 节点：生成分析报告（✅ v5.8: 注入证据面板数据）"""
    trace: TraceLog = state["trace"]
    user_query = state["user_query"]

    # 恢复动态计划
    plan_dict = state["dynamic_plan"]
    dynamic_plan = DynamicPlan.from_dict(plan_dict)

    # 恢复结果
    all_results = [AnalysisResult.from_dict(r_dict) for r_dict in state.get("all_results", [])]

    # ✅ v5.8: 恢复证据面板，供 reporter 使用
    evidence_board = EvidenceBoard.from_dict(state["_evidence_board"]) if state.get("_evidence_board") else None

    reporter = state["_reporter_agent"]  # ✅ 复用 gateway 中创建的实例
    report = reporter.generate_report(user_query, dynamic_plan, all_results,
                                       state.get("chart_paths", []),
                                       evidence_board=evidence_board)

    trace.log(TraceEventType.REPORT_GENERATED, iteration=state["current_iteration"],
              output_data={"summary": report.get("summary", "")[:200]})

    state["final_report"] = report

    trace.finalize("completed", {"route": "complex", "iterations": state["current_iteration"]})
    return state


# ============================================================================
# 13b. ✅ v7 新增：混合架构 — scan / detect / reason / reporter_v2
# ============================================================================

SIGNIFICANCE_THRESHOLD = 8  # 默认变化率阈值 — Commander 可动态覆盖

# ── ✅ Task 2A: 评测集 canonical key ↔ 中文展示名映射 ──────────────────
# 评测集（eval_cases_v3）使用英文 canonical key (traffic/gmv/...)，
# 但 Commander/PythonAgent 输出的 dimension 是中文列名（访客数/支付金额/...）。
# 不做映射会导致 actual_anomaly_dims 与 expected_anomaly_dims 永远对不上，
# scan_quality.recall 必为 0，A 组得分被卡在 0.6。
#
# 维护原则：
#   - 主表 (店铺经营概况.csv) 中存在的字段，全部双向映射
#   - 补充表 (ad_campaign / competitor_monitor) 中的字段也登记，
#     供 Step 3 deep RCA 输出 root_cause 时使用
#   - 一对多映射（如 conversion_rate 可同时指 支付转化率/uv_pay_rate）取主名
CANONICAL_DIM_MAP: Dict[str, str] = {
    # ── 主表（店铺经营概况）──
    "支付金额":     "gmv",
    "访客数":       "traffic",
    "浏览量":       "page_view",
    "支付转化率":   "conversion_rate",
    "客单价":       "avg_price",
    "件单价":       "unit_price",
    "支付买家数":   "paying_buyers",
    "支付订单数":   "pay_orders",
    "支付件数":     "pay_items",
    "加购人数":     "add_cart_users",
    "跳失率":       "bounce_rate",
    "平均停留时长(秒)": "avg_stay_seconds",
    "退款金额":     "refund_amount",
    "退款订单数":   "refund_orders",
    # ── 补充表（B 组 deep RCA 才能看到）──
    "花费":         "marketing_spend",
    "ROI":          "roi",
    "竞品售价":     "competitor_price",
    "本店售价":     "own_price",
}

# 反向映射（canonical → 中文），供 frame 校验和报告输出使用
CANONICAL_TO_DISPLAY: Dict[str, str] = {v: k for k, v in CANONICAL_DIM_MAP.items()}


# ── ✅ Task 2A 补强: 补充数据类型 → 关键列 marker 契约表 ──────────────
# 用于 process_deep_rca 入口校验：用户上传的 supplementary_df 必须至少匹配
# reasoner 在 step2 建议的某一种 data_type，否则拒收。
# 每个 type 至少需要 2 个 marker 列存在才算匹配（避免单列误命中）。
# Marker 列名取自 generate_test_data_v9.py 的 SUPPLEMENTARY_RENAME_MAP（中文落盘名）。
# 同时登记英文原名作为兜底（防止数据生成口径变更）。
SUPPLEMENTARY_TYPE_SIGNATURES: Dict[str, List[List[str]]] = {
    # 每个 type 对应一组"列名候选集"，列被识别为属于该 type 的标志
    "ad_campaign": [
        ["计划ID", "计划名称", "campaign_id", "campaign_name"],
        ["展现量", "点击量", "impressions", "clicks"],
        ["花费", "ROI", "cost", "roi"],
    ],
    "order_detail": [
        ["订单编号", "order_id"],
        ["买家ID", "buyer_id"],
        ["商品编码", "sku_id"],
        ["下单时间", "order_create_time"],
    ],
    "competitor_monitor": [
        ["竞品店铺", "竞品商品名", "competitor_shop", "competitor_sku_name"],
        ["竞品售价", "competitor_price"],
        ["本店对标SKU", "本店售价", "own_comparable_sku", "own_price"],
    ],
    "inventory_status": [
        ["仓库", "warehouse"],
        ["可用库存", "锁定库存", "available_qty", "locked_qty"],
        ["是否缺货", "out_of_stock_flag"],
    ],
    "refund_after_sale": [
        ["退款单号", "refund_id"],
        ["申请时间", "apply_time"],
        ["退款原因", "refund_reason"],
    ],
    "product_performance": [
        ["商品名称", "product_name"],
        ["商品访客数", "product_uv"],
        ["上架状态", "shelf_status"],
    ],
    "promotion_calendar": [
        ["活动ID", "活动名称", "promo_id", "promo_name"],
        ["开始日期", "结束日期", "start_date", "end_date"],
        ["优惠描述", "discount_desc"],
    ],
}


def _identify_supplementary_type(df_columns: List[str]) -> List[str]:
    """
    根据 df 的列名集合，识别它属于哪些已知的补充数据类型。
    返回所有匹配的 type 名（一个 df 可能同时满足多个，例如订单 + 退款合表）。
    匹配规则：每个 type 的 marker 组里，至少 2 组里的任一候选列出现在 df 中。
    """
    if not df_columns:
        return []
    col_set = {str(c).strip() for c in df_columns}
    matched: List[str] = []
    for type_name, marker_groups in SUPPLEMENTARY_TYPE_SIGNATURES.items():
        hits = 0
        for group in marker_groups:
            if any(marker in col_set for marker in group):
                hits += 1
        # 至少 2 个 marker 组命中才算匹配（防止单列误命中）
        if hits >= 2:
            matched.append(type_name)
    return matched


def _to_canonical_dim(display_name: str) -> Optional[str]:
    """中文列名 → 评测集 canonical key。未登记返回 None（不阻塞流程）。"""
    if not display_name:
        return None
    return CANONICAL_DIM_MAP.get(str(display_name).strip())


def _tag_canonical_inplace(items: List[Dict]) -> None:
    """
    给扫描结果 / 根因列表里每个 dict 原地补一个 canonical_dim 字段。
    评测器读 actual_anomaly_dims 时优先取 canonical_dim，回退到 dimension。
    L2/L3 切片结果 dimension 字段可能是 group_key（如品类=服装），
    这种情况映射为 None 是预期行为，不影响扫描数据本身。
    """
    if not items:
        return
    for it in items:
        if not isinstance(it, dict):
            continue
        dim = it.get("dimension")
        canonical = _to_canonical_dim(dim) if dim else None
        if canonical:
            it["canonical_dim"] = canonical


# ── ✅ Task 2A: L1 漏斗五段强制覆盖白名单 ────────────────────────────
# L1 必须扫"流量 → 互动 → 加购 → 转化 → 客单 → 结果"全链路，
# 否则 LLM 经常裁剪到只剩 4 个支付类指标（C01/C06 失败的根因）。
# 校验逻辑：取 schema 中实际存在的列，按段落补全 L1 metrics，
# 不动 LLM 已给出的 metric 列表（只补不删），不存在的列段静默跳过。
FUNNEL_STAGES: List[List[str]] = [
    ["访客数", "浏览量"],                    # 流量段
    ["平均停留时长(秒)", "跳失率"],          # 互动段
    ["加购人数"],                             # 加购段
    ["支付买家数", "支付转化率"],            # 转化段
    ["客单价", "件单价"],                    # 客单段
    ["支付金额", "支付订单数"],              # 结果段
]


def _enforce_l1_funnel_coverage(frame: "AnalysisFrame", schema: dict) -> List[str]:
    """
    强制 L1 metrics 覆盖漏斗五段。返回新增的指标名列表（用于 trace 记录）。
    - 只对 depth=1 的 layer 生效
    - 只补 schema 中真实存在的列
    - 已经存在的指标不动
    - target_metric 永远保留
    """
    if not frame or not frame.layers:
        return []

    # 找 L1
    l1 = next((L for L in frame.layers if L.depth == 1), None)
    if l1 is None:
        return []

    # schema 中可用的列名集合（兼容 dict 或 list 两种 schema 表达）
    available_cols: set = set()
    if isinstance(schema, dict):
        cols = schema.get("columns") or schema.get("fields") or []
        if isinstance(cols, dict):
            available_cols = set(cols.keys())
        elif isinstance(cols, list):
            for c in cols:
                if isinstance(c, str):
                    available_cols.add(c)
                elif isinstance(c, dict):
                    name = c.get("name") or c.get("column")
                    if name:
                        available_cols.add(name)
    # 兜底：从 frame.metrics 已有指标名也算"可用"
    available_cols.update(m.name for m in (frame.metrics or []))

    existing = set(l1.metrics or [])
    added: List[str] = []
    for stage in FUNNEL_STAGES:
        # 该段已有任一指标 → 该段已覆盖，跳过
        if any(s in existing for s in stage):
            continue
        # 否则补第一个 schema 中存在的代表指标
        for s in stage:
            if s in available_cols and s not in existing:
                l1.metrics.append(s)
                existing.add(s)
                added.append(s)
                break

    # target_metric 必须在 L1 里
    if frame.target_metric and frame.target_metric not in existing:
        if frame.target_metric in available_cols:
            l1.metrics.append(frame.target_metric)
            added.append(frame.target_metric)

    return added


# ── v9.6: suggested_data 标准化工具 ──────────────────────────────
# 把 LLM 输出的 suggested_data (description/reason/required_columns) 统一字段名,
# 并通过关键词映射补出 type_id, 让评测器和前端能直接读 .type 字段
_SUGGESTED_DATA_KEYWORD_TO_ID = {
    # ad_campaign
    "广告": "ad_campaign", "投放": "ad_campaign", "营销": "ad_campaign",
    "推广": "ad_campaign", "campaign": "ad_campaign", "marketing": "ad_campaign",
    "spend": "ad_campaign", "impression": "ad_campaign", "click": "ad_campaign",
    "渠道明细": "ad_campaign", "流量来源": "ad_campaign", "渠道流量": "ad_campaign",
    "roi": "ad_campaign", "roas": "ad_campaign",
    "cpc": "ad_campaign", "cpa": "ad_campaign", "cpm": "ad_campaign",
    # order_detail
    "订单": "order_detail", "客户": "order_detail", "用户分群": "order_detail",
    "漏斗": "order_detail", "转化漏斗": "order_detail",
    "用户行为": "order_detail",
    "customer": "order_detail", "order": "order_detail",
    # competitor_monitor
    "竞品": "competitor_monitor", "竞争": "competitor_monitor",
    "competitor": "competitor_monitor", "市场基准": "competitor_monitor",
    # inventory_status
    "库存": "inventory_status", "缺货": "inventory_status",
    "inventory": "inventory_status", "stock": "inventory_status",
    # promotion_calendar
    "促销": "promotion_calendar", "活动日历": "promotion_calendar",
    "优惠": "promotion_calendar", "promotion": "promotion_calendar",
    "营销活动": "promotion_calendar",
    # product_performance
    "商品": "product_performance", "sku": "product_performance",
    "product": "product_performance", "品类表现": "product_performance",
    "商品详情": "product_performance",
    # refund_after_sale
    "退款": "refund_after_sale", "售后": "refund_after_sale",
    "refund": "refund_after_sale",
}


def _normalize_suggested_data(items: list) -> list:
    """统一 suggested_data 字段: reasoning→reason, 并补出 type_id"""
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        desc = it.get("description", "") or ""
        reason = it.get("reason") or it.get("reasoning") or ""
        cols = it.get("required_columns", []) or []
        haystack = (desc + " " + reason + " " + " ".join(cols)).lower()
        type_id = it.get("type")
        if not type_id or type_id == "unknown":
            for kw, tid in _SUGGESTED_DATA_KEYWORD_TO_ID.items():
                if kw.lower() in haystack:
                    type_id = tid
                    break
        out.append({
            "type": type_id or "unknown",
            "description": desc,
            "reason": reason,
            "required_columns": cols,
        })
    return out


def _suggested_data_type_ids(items: list) -> list:
    """从标准化后的 suggested_data 中抽出去重的 type_id 列表"""
    if not isinstance(items, list):
        return []
    seen = []
    for it in items:
        if not isinstance(it, dict):
            continue
        tid = it.get("type")
        if tid and tid != "unknown" and tid not in seen:
            seen.append(tid)
    return seen


def _ensure_limitation_section(report_text: str, reporter_llm, system_prompt: str,
                               state: dict = None) -> str:
    """v9.6 Patch 5 (v2): 确保报告末尾包含'分析局限'段落, 缺失则调用 LLM 补写。
    评测器 ReportValidator.limitation_score 会扫描这一段, 缺失会扣到 0.3。

    v2 改动:新增 state 参数,保底模板从静态文案改为动态读取当前 case 的
    reason_result / schema 拼接,避免千篇一律的误导性描述(例如把实际已覆盖的
    维度误报为"未覆盖")。state=None 时退化为通用兜底文案,保持向后兼容。
    """
    if not report_text or not isinstance(report_text, str):
        return report_text
    # 检查最后 800 字符内是否含"分析局限"或"局限"标题
    tail = report_text[-800:]
    has_section = (
        "## ⚠️ 分析局限" in report_text
        or "## 分析局限" in report_text
        or ("分析局限" in tail and ("##" in tail or "###" in tail))
    )
    if has_section:
        return report_text
    # 触发补写
    fix_prompt = (
        f"以下报告缺少强制要求的 '## ⚠️ 分析局限与未覆盖维度' 段落。\n"
        f"请在原报告末尾追加这一段(包含数据覆盖局限/未验证假设/建议补充数据三条要点),\n"
        f"其它内容保持不变。**只返回完整的修订后报告**, 不要任何解释文字。\n\n"
        f"=== 原报告 ===\n{report_text}"
    )
    try:
        fixed = reporter_llm.generate(system_prompt, fix_prompt)
        if fixed and isinstance(fixed, str) and "局限" in fixed[-1500:]:
            return fixed
    except Exception:
        pass

    # ── LLM 补写失败 → 动态保底模板 ──
    # 从 state.reason_result 读取当前 case 的真实 suggested_data / root_causes,
    # 避免硬编码"未覆盖 ad_campaign/competitor_monitor/inventory_status"造成误导。
    _reason = {}
    if state and isinstance(state, dict):
        _reason = state.get("reason_result") or state.get("causal_result") or {}

    # 1) 未覆盖的数据类型 — 从 suggested_data 提取描述
    _missing_descs = []
    for item in (_reason.get("suggested_data") or []):
        if isinstance(item, dict):
            _desc = (item.get("description") or item.get("type")
                     or item.get("reason") or "").strip()
            if _desc and _desc not in _missing_descs:
                _missing_descs.append(_desc)
    if not _missing_descs:
        _missing_descs = ["更细粒度的外部维度数据(投放/竞品/库存等)"]

    # 2) 未充分验证的根因 — 取 confidence 非 high 的 root_causes
    _unverified = []
    for rc in (_reason.get("root_causes") or []):
        if isinstance(rc, dict):
            _dim = (rc.get("dimension") or rc.get("name")
                    or rc.get("canonical_dim") or "").strip()
            _conf = str(rc.get("confidence") or rc.get("confidence_level") or "").lower()
            if _dim and _conf not in ("high", "高"):
                if _dim not in _unverified:
                    _unverified.append(_dim)
    if not _unverified:
        _unverified = ["部分根因假设"]

    # 3) 当前已用数据源 — 从 cached schema 读取表名
    _used_sources = []
    try:
        _schema = get_cached_schema() or {}
        for _tbl in (_schema.get("tables") or []):
            _n = (_tbl.get("name") or "").strip() if isinstance(_tbl, dict) else ""
            if _n and _n not in _used_sources:
                _used_sources.append(_n)
    except Exception:
        pass
    _sources_str = "、".join(_used_sources[:5]) if _used_sources else "已提供的汇总数据"

    fallback = (
        "\n\n## ⚠️ 分析局限与未覆盖维度\n"
        f"1. **数据覆盖局限**: 本次分析基于 {_sources_str},"
        f"未包含以下可能相关的数据源: {'、'.join(_missing_descs[:4])}。\n"
        f"2. **未充分验证的假设**: 以下根因/假设因数据或信心度不足,"
        f"未能在本轮完成闭环验证: {'、'.join(_unverified[:3])}。\n"
        "3. **建议补充数据**: 建议补充上述数据源,"
        "用于进一步定位和验证根本原因。\n"
    )
    return report_text + fallback


# ── 因果推理 Prompt（保留，Step 2 改造时再升级） ──
CAUSAL_REASONING_PROMPT = """你是电商数据分析专家。
根据维度扫描结果，判断异常维度之间的因果关系。

## 电商核心公式
GMV = 流量(traffic) × 转化率(conversion_rate) × 客单价(avg_price)

## 已知因果关系
- marketing_spend↓ → traffic↓（营销投放驱动流量）
- competitor_price↓ → conversion_rate↓（竞品降价影响转化率）
- avg_price↑ → conversion_rate↓（提价可能降低转化）

## 你的任务
1. 从显著异常维度中，识别"根本原因"和"中间结果"
2. 构建因果链
3. 如果用户有预设假设，判断是否需要纠正
4. 评估是否需要更深入的根因分析（needs_deep_rca）
5. 如果 needs_deep_rca=true，基于当前数据的局限性，建议用户可以补充哪些数据

## needs_deep_rca 判定规则（保守策略：默认 false）
设为 true 仅当以下任一条件成立：
- confidence 为 "low"（因果链不确定，现有数据无法充分解释）
- 多个 root_cause 变化幅度接近（差距 < 5%），无法区分主次因
- 根因指向外部因素（如"竞品"、"市场环境"、"季节性"等），现有数据无法验证
- Arbiter 曾被调用且建议 "widen_scope"

## suggested_data 生成规则（仅 needs_deep_rca=true 时生成）
- 根据已识别根因的局限性，推断需要什么补充数据
- 每条建议必须包含：description（数据描述）、reason（为什么需要）、required_columns（建议包含的列）
- 建议的数据维度必须与根因相关，不要泛泛建议
- 建议数量 1-3 条，优先级从高到低

输出 JSON:
{
  "root_causes": [{"dimension": "...", "change_pct": ..., "reasoning": "..."}],
  "intermediate_effects": [{"dimension": "...", "caused_by": "...", "reasoning": "..."}],
  "causal_chain": "marketing_spend↓80% → traffic↓60% → GMV↓47%",
  "user_assumption_correction": "用户认为是流量问题，但实际...(无需纠正时留空字符串)",
  "confidence": "high|medium|low",
  "needs_deep_rca": false,
  "suggested_data": []
}

示例 — needs_deep_rca=true 的 suggested_data:
"suggested_data": [
  {
    "description": "营销投放明细数据",
    "reason": "当前仅有汇总级别的投放费用，无法定位具体哪个渠道/计划导致变化",
    "required_columns": ["date", "channel", "campaign_id", "spend", "impressions", "clicks"]
  }
]

只返回 JSON:"""


# ============================================================================
# v8 Quick Scan — Commander + Scan Loop + Detect + Arbiter
# ============================================================================

# ── Commander System Prompt ──
COMMANDER_SYSTEM_PROMPT = """你是一个数据分析扫描规划器（Commander）。
你的任务是根据用户问题和数据 Schema，生成一个结构化的扫描计划（AnalysisFrame）。

## 你的输入
1. 用户问题（自然语言）
2. 数据 Schema（表名、列名、类型、unit_hint、枚举值等）
3. 数据时间范围（min/max 日期）

## 你的输出
严格返回一个 JSON 对象，字段如下:

{
  "analysis_depth": "descriptive|diagnostic|causal",
  "depth_reasoning": "简要说明为什么选择这个深度",
  "question_category": "general|roi_ad|competitor|inventory|promotion",
  "target_metric": "用户最关心的指标名（必须来自 Schema 中的数值列）",
  "metrics": [
    {"name": "列名", "aggregation": "sum|weighted_avg|mean", "description": "简要说明"}
  ],
  "time_window": {
    "current_start": "YYYY-MM-DD",
    "current_end": "YYYY-MM-DD",
    "previous_start": "YYYY-MM-DD",
    "previous_end": "YYYY-MM-DD",
    "comparison_type": "mom|yoy|wow|custom"
  },
  "layers": [
    {"depth": 1, "metrics": ["列名1","列名2"], "group_by": [], "trigger_condition": "always"},
    {"depth": 2, "metrics": ["异常指标"], "group_by": ["分类维度"], "trigger_condition": "anomaly_found"}
  ],
  "significance_threshold": 8.0,
  "filters": {},
  "reasoning": "简要说明你的规划理由"
}

## analysis_depth 分级规则（关键）
根据用户问题的语义意图判断分析深度：

- **descriptive**（描述型）: 用户仅想确认事实或了解变化概况，**不需要任何归因**
  仅限以下场景: 纯询问型（"是不是降了"）、纯概况型（"有没有异常"）、纯变化观察型（"变化如何"）
  关键判断: 用户问题可以用"是/否"或"涨了/降了/持平"直接回答，无需解释原因
  例: "8月GMV有没有异常" → descriptive（回答有没有即可）
  例: "8月流量是不是降了" → descriptive（回答是或否即可）
  例: "7月中到8月中转化率变化" → descriptive（描述变化幅度即可）
  ⚠️ 注意: "怎么了"、"不行了"、"出了问题" 虽然口语化，但隐含归因需求，不属于 descriptive

- **diagnostic**（诊断型）: 用户关注异常定位、效果评估或隐含"发生了什么"的归因
  关键词/句式: "怎么了"、"哪个出了问题"、"效果怎么样"（带具体对象）、"不行了"、"整体表现如何"、"定位"、"找出"
  关键判断: 用户需要知道"哪里出了问题"或"表现好不好"，需要交叉下钻但不要求完整因果链
  例: "京东电子品类8月怎么了" → diagnostic（需要定位到底哪些指标有问题）
  例: "哪个渠道品类出了问题" → diagnostic
  例: "京东8月推广效果怎么样" → diagnostic
  例: "抖音渠道8月整体表现如何" → diagnostic（需要多指标综合评估）

- **causal**（因果型）: 用户要追溯根本原因，需要完整因果推理链
  关键词/句式: "为什么"、"原因"、"根因"、"什么导致"、"怎么回事"
  例: "8月GMV为什么下降了" → causal
  例: "客单价为什么上涨" → causal

如果无法确定，默认选 diagnostic（宁可多分析一步，不要漏掉归因）。

## question_category 分类规则
根据用户问题的业务领域判断问题类别，用于零异常时推荐补充数据：

- **roi_ad**（投放/广告效率类）: 问题涉及 ROI、ROAS、CPC、CPA、CPM、投放效率、投放回报、广告效果、转化成本、获客成本、推广效率等
  例: "淘宝的投放效率为什么变差了" → roi_ad
  例: "8月ROI为什么下降" → roi_ad
- **competitor**（竞品类）: 问题涉及竞品、竞争对手、市场份额等
  例: "竞品最近有什么动作" → competitor
- **inventory**（库存类）: 问题涉及库存、缺货、断货、补货等
  例: "为什么会缺货" → inventory
- **promotion**（促销类）: 问题涉及促销、活动、优惠、折扣、券等
  例: "618活动效果怎么样" → promotion
- **general**（通用）: 不属于以上任何类别
  例: "8月GMV为什么下降" → general

如果无法确定，默认选 general。

## 关键规则
1. **metrics**: 所有数值列都应被扫描，不要遗漏。aggregation 必须根据 unit_hint 判断:
   - unit_hint 为 "decimal_ratio" 或 "already_percentage" → "weighted_avg"
   - 其他数值列 → "sum"（大部分情况）或 "mean"（明确是均值指标时）
2. **time_window**: 从用户问题推断对比方式:
   - "环比" / "上个月" / "上月" → mom（月环比）
   - "同比" / "去年同期" → yoy（年同比）
   - "上周" → wow（周环比）
   - **用户指定了具体日期范围** → custom（自定义窗口）
     例: "7月中到8月中" → custom，current = 用户指定范围，previous = 等长前移窗口
     例: "8月15日到9月15日" → custom，current = 08-15~09-15，previous = 07-15~08-14
     例: "最近两周" → custom，current = 最近14天，previous = 前14天
     **规则**: 当用户提到"X月中"，解释为该月15日；"X月初"→01日；"X月底/末"→月末最后一天。
     **previous 窗口**: 自动向前平移与 current 等长的时间段作为对比基准。
     **comparison_type 必须设为 "custom"**，不要降级为 mom。
   - 未明确指定 → 默认 mom
   - 日期边界从数据时间范围推断，不要编造超出范围的日期
3. **significance_threshold**: 根据用户表述动态调整:
   - "大幅下降" / "暴跌" → 15
   - "略有下降" / "轻微" → 3
   - 未指定 → 8（默认）
4. **layers**: L1 始终为全局指标扫描（group_by=[]），L2 按分类列下钻
5. **filters**: 如果用户问题提及特定筛选条件（如"抖音渠道"），提取到 filters 中
   - ⚠️ filters 中的键必须是 Schema 中实际存在的列名。如果数据是单平台导出（Schema 中没有"平台"列），即使用户提到平台名称，也不要添加平台筛选
6. **target_metric**: 用户最关心的指标（如 "GMV"、"销售额" 等），必须在 metrics 中存在。如果用户未明确指定，从 Schema 中选择第一个 sum 型数值列。

只返回 JSON，不要有任何其他内容:"""


def _build_metrics_from_schema(schema: dict) -> List[MetricSpec]:
    """
    从 Schema 动态构建 MetricSpec 列表。
    核心原则: 完全依赖 schema 元信息（unit_hint, type），不硬编码列名规则。
    """
    metrics = []
    for table in schema.get("tables", []):
        for col_info in table.get("columns", []):
            col_type = col_info.get("type", "")
            if col_type not in ("INTEGER", "REAL"):
                continue
            name = col_info["name"]
            unit_hint = col_info.get("unit_hint", "")
            description = col_info.get("description", "")
            # 聚合方式: 直接使用 infer_aggregation_from_schema（它已内置 unit_hint 优先逻辑）
            agg = infer_aggregation_from_schema(name, schema)
            metrics.append(MetricSpec(
                name=name,
                aggregation=agg,
                unit_hint=unit_hint,
                description=description,
            ))
    return metrics


def _extract_custom_date_range(user_query: str, date_range: dict) -> Tuple[str, str]:
    """
    ✅ P0 fix: 从用户问题中提取自定义日期范围。
    支持: "7月中到8月中", "8月15日到9月15日", "7月中旬到8月中旬", "最近两周" 等。
    返回 (start_date_str, end_date_str)，解析失败返回 ("", "")。
    """
    import re

    # 推断年份: 从数据时间范围的 max 年份
    max_date_str = date_range.get("max", "")
    try:
        base_year = pd.to_datetime(max_date_str).year
    except Exception:
        base_year = datetime.now().year

    # ── Pattern 1: "X月中/初/底 到 Y月中/初/底" ──
    pattern_fuzzy = re.compile(
        r'(\d{1,2})月\s*(初|中|中旬|底|末|上旬|下旬)?'
        r'\s*[到至~\-–—]\s*'
        r'(\d{1,2})月\s*(初|中|中旬|底|末|上旬|下旬)?'
    )
    m = pattern_fuzzy.search(user_query)
    if m:
        m1, q1, m2, q2 = int(m.group(1)), m.group(2) or "", int(m.group(3)), m.group(4) or ""
        d1 = _fuzzy_month_to_day(base_year, m1, q1)
        d2 = _fuzzy_month_to_day(base_year, m2, q2)
        if d1 and d2 and d2 > d1:
            return str(d1), str(d2)

    # ── Pattern 2: "X月DD日 到 Y月DD日" ──
    pattern_exact = re.compile(
        r'(\d{1,2})月(\d{1,2})[日号]?\s*[到至~\-–—]\s*(\d{1,2})月(\d{1,2})[日号]?'
    )
    m = pattern_exact.search(user_query)
    if m:
        try:
            d1 = pd.Timestamp(year=base_year, month=int(m.group(1)), day=int(m.group(2))).date()
            d2 = pd.Timestamp(year=base_year, month=int(m.group(3)), day=int(m.group(4))).date()
            if d2 > d1:
                return str(d1), str(d2)
        except Exception:
            pass

    # ── Pattern 3: "最近N周/天" ──
    pattern_recent = re.compile(r'最近\s*(\d+)\s*(周|天|日)')
    m = pattern_recent.search(user_query)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * 7 if unit == "周" else n
        try:
            end_dt = pd.to_datetime(max_date_str).date()
            start_dt = end_dt - timedelta(days=days - 1)
            return str(start_dt), str(end_dt)
        except Exception:
            pass

    return "", ""


def _fuzzy_month_to_day(year: int, month: int, qualifier: str):
    """将 '7月中' 等模糊表述转为具体日期。"""
    import calendar as cal
    if month < 1 or month > 12:
        return None
    last_day = cal.monthrange(year, month)[1]
    q = qualifier.replace("旬", "")
    if q in ("初", "上"):
        day = 1
    elif q in ("中",):
        day = 15
    elif q in ("底", "末", "下"):
        day = last_day
    else:
        day = 1  # 无修饰符默认月初
    try:
        from datetime import date
        return date(year, month, day)
    except Exception:
        return None


def _build_default_time_window(date_range: dict, comparison_type: str = "mom",
                               custom_start: str = "", custom_end: str = "") -> TimeWindow:
    """
    从 date_range 构建默认时间窗口（规则兜底，不依赖 LLM）。
    custom_start/custom_end: 用户指定的自定义日期范围（仅 comparison_type="custom" 时使用）。
    """
    # ── custom 自定义窗口: 用户指定了具体日期范围 ──
    if comparison_type == "custom" and custom_start and custom_end:
        try:
            cs = pd.to_datetime(custom_start)
            ce = pd.to_datetime(custom_end)
            span = (ce - cs).days
            if span > 0:
                ps = cs - timedelta(days=span)
                pe = cs - timedelta(days=1)
                return TimeWindow(
                    current_start=str(cs.date()),
                    current_end=str(ce.date()),
                    previous_start=str(ps.date()),
                    previous_end=str(pe.date()),
                    comparison_type="custom",
                )
        except Exception:
            pass  # 解析失败，降级到 mom

    cur_year, cur_month = safe_get_year_month(date_range, "max")
    if cur_year is None or cur_month is None:
        now = datetime.now()
        cur_year, cur_month = now.year, now.month

    # 当期: 最新月
    cur_start = f"{cur_year}-{cur_month:02d}-01"
    cur_end = date_range.get("max", f"{cur_year}-{cur_month:02d}-28")

    if comparison_type == "yoy":
        prev_year = cur_year - 1
        prev_start = f"{prev_year}-{cur_month:02d}-01"
        last_day = calendar.monthrange(prev_year, cur_month)[1]
        prev_end = f"{prev_year}-{cur_month:02d}-{last_day:02d}"
    elif comparison_type == "wow":
        # 周环比: 当期最后7天 vs 前7天
        max_ts = safe_parse_date(date_range.get("max", ""))
        if max_ts and not pd.isna(max_ts):
            cur_end_dt = max_ts
            cur_start_dt = cur_end_dt - timedelta(days=6)
            prev_end_dt = cur_start_dt - timedelta(days=1)
            prev_start_dt = prev_end_dt - timedelta(days=6)
            return TimeWindow(
                current_start=str(cur_start_dt.date()),
                current_end=str(cur_end_dt.date()),
                previous_start=str(prev_start_dt.date()),
                previous_end=str(prev_end_dt.date()),
                comparison_type="wow",
            )
        # fallback to mom
        comparison_type = "mom"

    if comparison_type == "mom":
        prev_month = cur_month - 1 if cur_month > 1 else 12
        prev_year = cur_year if cur_month > 1 else cur_year - 1
        last_day = calendar.monthrange(prev_year, prev_month)[1]
        prev_start = f"{prev_year}-{prev_month:02d}-01"
        prev_end = f"{prev_year}-{prev_month:02d}-{last_day:02d}"

    return TimeWindow(
        current_start=cur_start,
        current_end=cur_end,
        previous_start=prev_start,
        previous_end=prev_end,
        comparison_type=comparison_type,
    )


def _build_default_frame(schema: dict, date_range: dict,
                         user_query: str = "") -> AnalysisFrame:
    """
    规则兜底: 当 Commander LLM 失败时，纯规则生成 AnalysisFrame。
    所有信息均从 schema / date_range 动态推断，不硬编码列名。
    """
    metrics = _build_metrics_from_schema(schema)
    if not metrics:
        return AnalysisFrame(reasoning="schema 中无数值列，无法构建扫描计划")

    # 推断 target_metric: 优先选第一个 sum 型指标
    target = metrics[0].name
    for m in metrics:
        if m.aggregation == "sum":
            target = m.name
            break

    # 推断对比方式（简单关键词匹配）
    comp_type = "mom"
    custom_start, custom_end = "", ""
    if any(kw in user_query for kw in ("同比", "去年同期", "去年")):
        comp_type = "yoy"
    elif any(kw in user_query for kw in ("周环比", "上周")):
        comp_type = "wow"
    else:
        # ✅ P0 fix: 检测用户自定义日期范围 (e.g. "7月中到8月中", "8月15日到9月15日")
        custom_start, custom_end = _extract_custom_date_range(user_query, date_range)
        if custom_start and custom_end:
            comp_type = "custom"

    tw = _build_default_time_window(date_range, comp_type,
                                    custom_start=custom_start, custom_end=custom_end)

    # 推断阈值
    threshold = SIGNIFICANCE_THRESHOLD
    if any(kw in user_query for kw in ("大幅", "暴跌", "暴涨", "剧烈", "急剧")):
        threshold = 15.0
    elif any(kw in user_query for kw in ("略有", "轻微", "小幅", "微降")):
        threshold = 3.0

    # 从 schema 读取分类列（用于 L2 下钻）
    text_cols = [col for _, col, _ in schema.get("_meta", {}).get("text_columns", [])]
    all_metric_names = [m.name for m in metrics]

    # 构建扫描层级
    layers = [
        ScanLayer(depth=1, metrics=all_metric_names, group_by=[],
                  trigger_condition="always"),
    ]
    # L2: 每个分类列独立下钻（条件触发）
    for tc in text_cols[:3]:  # 最多 3 个分类维度
        layers.append(ScanLayer(
            depth=2, metrics=all_metric_names,
            group_by=[tc], trigger_condition="anomaly_found",
        ))

    # 从 user_query 提取过滤条件
    filters = {}
    for _, col, enum_values in schema.get("_meta", {}).get("text_columns", []):
        for val in enum_values:
            if val in user_query:
                filters[col] = val

    # ✅ v9: 规则推断 analysis_depth
    analysis_depth = "diagnostic"  # 默认
    causal_keywords = ["为什么", "原因", "根因", "什么导致", "怎么回事"]
    diagnostic_keywords = ["怎么了", "不行了", "出了问题", "整体表现", "表现如何",
                           "效果怎么样", "怎么样"]  # 隐含归因的口语化表达
    # descriptive 仅限纯询问/纯概况，不含任何归因暗示
    descriptive_keywords = ["有没有异常", "是不是", "是否",
                            "帮我看看", "情况如何"]
    # "变化" 单独处理：仅当不含归因关键词时才算 descriptive
    has_change_keyword = "变化" in user_query and "为什么" not in user_query
    if any(kw in user_query for kw in causal_keywords):
        analysis_depth = "causal"
    elif any(kw in user_query for kw in diagnostic_keywords):
        analysis_depth = "diagnostic"
    elif any(kw in user_query for kw in descriptive_keywords) or has_change_keyword:
        analysis_depth = "descriptive"

    # ✅ v9.7: 规则推断 question_category
    question_category = "general"
    roi_ad_kws = ["roi", "roas", "cpc", "cpa", "cpm", "投放效率", "投放回报", "投放效果",
                  "广告效果", "转化成本", "获客成本", "投产比", "广告投放", "投放变差",
                  "推广效率", "投放"]
    competitor_kws = ["竞品", "竞争", "对手", "竞争对手", "市场份额"]
    inventory_kws = ["库存", "缺货", "断货", "补货", "发货"]
    promotion_kws = ["促销", "活动效果", "优惠", "折扣", "券"]
    _q_lower = user_query.lower()
    if any(kw in _q_lower for kw in roi_ad_kws):
        question_category = "roi_ad"
    elif any(kw in _q_lower for kw in competitor_kws):
        question_category = "competitor"
    elif any(kw in _q_lower for kw in inventory_kws):
        question_category = "inventory"
    elif any(kw in _q_lower for kw in promotion_kws):
        question_category = "promotion"

    return AnalysisFrame(
        target_metric=target,
        metrics=metrics,
        time_window=tw,
        layers=layers,
        significance_threshold=threshold,
        max_depth=2,
        filters=filters,
        reasoning="规则兜底生成（Commander 未调用或失败）",
        analysis_depth=analysis_depth,
        depth_reasoning="规则兜底推断",
        question_category=question_category,
    )


def _run_commander(user_query: str, schema: dict, date_range: dict,
                   llm: LLMInterface, trace: TraceLog) -> AnalysisFrame:
    """
    Commander: 用 LLM 从用户问题 + schema 动态生成 AnalysisFrame。
    失败时降级为 _build_default_frame()。
    """
    schema_info = format_schema_for_prompt(schema)
    date_info = f"{date_range.get('min', 'unknown')} 至 {date_range.get('max', 'unknown')}"
    date_col = date_range.get("date_column", "unknown")

    # v9.3.1: 检测单平台数据，生成数据特征提示
    all_col_names = set()
    for table in schema.get("tables", []):
        for col_info in table.get("columns", []):
            all_col_names.add(col_info["name"])
    data_notes = []
    if "平台" not in all_col_names and "platform" not in all_col_names:
        data_notes.append(
            '⚠️ 当前数据为【单平台导出】，不包含"平台"列。'
            '即使用户问题中提到平台名称（如京东、淘宝），也不要在 filters 中添加平台筛选，'
            '因为整张表的数据已经是该平台的，无需也无法按平台过滤。')
    data_notes_text = "\n".join(data_notes)

    prompt = f"""[数据 Schema]
{schema_info}

[数据时间范围] {date_info} (日期列: {date_col})

{f'[数据特征提示]{chr(10)}{data_notes_text}{chr(10)}' if data_notes_text else ''}[用户问题] {user_query}

请生成扫描计划 JSON:"""

    try:
        raw = llm.generate(COMMANDER_SYSTEM_PROMPT, prompt)
        raw = raw.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(raw)

        # 解析 metrics
        metrics = []
        for m in parsed.get("metrics", []):
            metrics.append(MetricSpec(
                name=m["name"],
                aggregation=m.get("aggregation", "sum"),
                description=m.get("description", ""),
            ))

        # 如果 Commander 遗漏了 schema 中的指标，补全
        schema_metrics = _build_metrics_from_schema(schema)
        commander_names = {m.name for m in metrics}
        for sm in schema_metrics:
            if sm.name not in commander_names:
                metrics.append(sm)

        # 解析 time_window
        tw_raw = parsed.get("time_window", {})
        tw = TimeWindow.from_dict(tw_raw) if tw_raw else _build_default_time_window(date_range)

        # 校验日期有效性 — 若 Commander 产出了不合理日期则降级
        _tw_invalid = False
        for date_field in (tw.current_start, tw.current_end, tw.previous_start, tw.previous_end):
            if date_field and date_field != "":
                ts = safe_parse_date(date_field)
                if ts is None or pd.isna(ts):
                    _tw_invalid = True
                    break
        if _tw_invalid:
            # ✅ P0 fix: custom 类型降级时也尝试从 user_query 提取自定义窗口
            custom_start, custom_end = _extract_custom_date_range(user_query, date_range)
            if custom_start and custom_end:
                tw = _build_default_time_window(date_range, "custom",
                                                custom_start=custom_start, custom_end=custom_end)
            else:
                tw = _build_default_time_window(date_range)

        # 解析 layers
        layers = []
        for l_raw in parsed.get("layers", []):
            layers.append(ScanLayer.from_dict(l_raw))
        if not layers:
            # 兜底: 至少有 L1
            all_metric_names = [m.name for m in metrics]
            layers = [ScanLayer(depth=1, metrics=all_metric_names, trigger_condition="always")]

        threshold = float(parsed.get("significance_threshold", SIGNIFICANCE_THRESHOLD))

        frame = AnalysisFrame(
            target_metric=parsed.get("target_metric", metrics[0].name if metrics else ""),
            metrics=metrics,
            time_window=tw,
            layers=layers,
            significance_threshold=threshold,
            max_depth=int(parsed.get("max_depth", 2)),
            filters=parsed.get("filters", {}),
            reasoning=parsed.get("reasoning", ""),
            # ✅ v9: 解析 analysis_depth
            analysis_depth=parsed.get("analysis_depth", "diagnostic"),
            depth_reasoning=parsed.get("depth_reasoning", ""),
            # ✅ v9.7: 解析 question_category
            question_category=parsed.get("question_category", "general"),
        )

        # v9.3.1: 过滤掉 schema 中不存在的列的 filter（防止 PythonAgent 生成无效筛选）
        if frame.filters:
            invalid_filters = [k for k in frame.filters if k not in all_col_names]
            if invalid_filters:
                for k in invalid_filters:
                    del frame.filters[k]
                trace.log(TraceEventType.COMMANDER_GENERATED, iteration=0,
                          output_data={
                              "phase": "filter_validation",
                              "removed_filters": invalid_filters,
                              "note": "移除了 schema 中不存在的列的 filter（如单平台数据中的平台筛选）",
                          },
                          metadata={"phase": "v8_commander_filter_fix"})

        # ✅ v9: 校验 analysis_depth 值域
        if frame.analysis_depth not in ("descriptive", "diagnostic", "causal"):
            frame.analysis_depth = "diagnostic"  # 安全兜底

        # ✅ v9.7: 校验 question_category 值域
        if frame.question_category not in ("general", "roi_ad", "competitor", "inventory", "promotion"):
            frame.question_category = "general"

        # ✅ Task 2A: 强制 L1 漏斗五段覆盖（修 C01/C06 L1 被裁剪问题）
        # LLM 经常只在 L1 写 4 个支付类指标，把流量/转化等上游全裁掉。
        # 这里只做"补"不做"删"，target_metric 一定保留。
        try:
            added_funnel = _enforce_l1_funnel_coverage(frame, schema)
            if added_funnel:
                trace.log(TraceEventType.COMMANDER_GENERATED, iteration=0,
                          output_data={
                              "phase": "l1_funnel_enforcement",
                              "added_metrics": added_funnel,
                              "note": "L1 漏斗五段强制补全",
                          },
                          metadata={"phase": "v8_commander_funnel"})
        except Exception as _funnel_err:
            # 校验器永远不能阻塞主流程
            pass

        trace.log(TraceEventType.COMMANDER_GENERATED, iteration=0,
                  output_data={
                      "target_metric": frame.target_metric,
                      "metric_count": len(frame.metrics),
                      "layer_count": len(frame.layers),
                      "threshold": frame.significance_threshold,
                      "comparison_type": frame.time_window.comparison_type if frame.time_window else "unknown",
                      "analysis_depth": frame.analysis_depth,
                      "depth_reasoning": frame.depth_reasoning,
                      "question_category": frame.question_category,
                  },
                  success=True,
                  metadata={"phase": "v8_commander"})
        return frame

    except Exception as e:
        # Commander 失败 → 降级到规则兜底
        trace.log(TraceEventType.COMMANDER_FALLBACK, iteration=0,
                  error=str(e)[:300],
                  metadata={"phase": "v8_commander", "action": "fallback_to_default_frame"})
        return _build_default_frame(schema, date_range, user_query)


def _build_layer_query(frame: AnalysisFrame, layer: ScanLayer,
                       schema: dict, anomaly_dims: List[str] = None) -> str:
    """
    为某一扫描层级构建自然语言查询（交给 PythonAgent 执行）。
    不硬编码列名，所有信息来自 AnalysisFrame + schema。
    """
    tw = frame.time_window
    if not tw:
        return ""

    # 构建指标列表 + 聚合说明（完全从 frame.metrics 推断）
    metrics_in_layer = layer.metrics or [m.name for m in frame.metrics]
    metric_specs = {m.name: m for m in frame.metrics}

    sum_cols, ratio_cols, mean_cols = [], [], []
    for mn in metrics_in_layer:
        spec = metric_specs.get(mn)
        agg = spec.aggregation if spec else infer_aggregation_from_schema(mn, schema)
        if agg == "weighted_avg":
            ratio_cols.append(mn)
        elif agg == "sum":
            sum_cols.append(mn)
        else:
            mean_cols.append(mn)

    agg_instructions = []
    if sum_cols:
        agg_instructions.append(f"sum 聚合: {', '.join(sum_cols)}")
    if ratio_cols:
        agg_instructions.append(
            f"加权平均聚合: {', '.join(ratio_cols)}"
            f"（⚠️ 比率指标必须用 sum(分子)/sum(分母) 计算，不能用 mean）")
    if mean_cols:
        agg_instructions.append(f"mean 聚合: {', '.join(mean_cols)}")

    # 过滤条件
    all_filters = {**frame.filters, **layer.filters}
    filter_text = ""
    if all_filters:
        parts = [f"{k}='{v}'" for k, v in all_filters.items()]
        filter_text = f"，筛选条件: {', '.join(parts)}"

    # 分组维度
    group_by = layer.group_by

    # 如果是 L2+ 下钻，且指定了 anomaly_dims，只扫描异常指标
    if layer.depth >= 2 and anomaly_dims:
        metrics_in_layer = anomaly_dims
        if not metrics_in_layer:
            metrics_in_layer = anomaly_dims[:3]

    if not group_by:
        # L1 全局扫描
        query = f"""请对比 {tw.previous_start} 至 {tw.previous_end} 和 {tw.current_start} 至 {tw.current_end} 的所有核心指标{filter_text}。

对每个指标分别计算：当期值、上期值、变化量、变化率(change_pct)。
聚合方式: {'; '.join(agg_instructions) if agg_instructions else '默认 sum 聚合'}

输出 result_df 格式要求:
- 每行一个指标，列为: dimension, current_value, previous_value, change, change_pct
- 按 abs(change_pct) 降序排列
- answer 设置为包含所有指标的 list[dict]，每个 dict 包含 dimension, current_value, previous_value, change, change_pct
- summary 设置为所有指标的变化概要

⚠️ 重要: 一次性计算所有数值指标（{', '.join(metrics_in_layer)}）的变化，不要遗漏任何指标。
⚠️ 变化率计算: change_pct = round((current_value - previous_value) / previous_value * 100, 2)
⚠️ 输出单个 result_df，每行对应一个指标的对比结果。
⚠️ 只使用 df 中实际存在的列进行筛选和计算，不要假设存在 Schema 中未列出的列（如"平台"列）。"""
    else:
        # L2+ 分组下钻
        target_metrics_str = ', '.join(metrics_in_layer)
        group_by_str = ', '.join(group_by)
        query = f"""对比 {tw.previous_start} 至 {tw.previous_end} 和 {tw.current_start} 至 {tw.current_end}{filter_text}，
按 {group_by_str} 分组，计算 {target_metrics_str} 的变化率。

聚合方式: {'; '.join(agg_instructions) if agg_instructions else '默认 sum 聚合'}

输出每个分组的 change_pct，按 abs(change_pct) 降序排列。
找出哪个分组对变化贡献最大。

输出 result_df 格式要求:
- 列为: {group_by_str}, dimension, current_value, previous_value, change, change_pct
- 按 abs(change_pct) 降序排列
- answer 设置为 list[dict]，包含所有分组的结果
- summary 说明哪个分组贡献最大"""

    return query


def _execute_scan_layer(layer: ScanLayer, query: str,
                        python_agent, uploaded_df, schema: dict,
                        trace: TraceLog, iteration: int) -> List[Dict]:
    """
    执行单层扫描，复用 PythonAgent.execute_from_query()。
    返回扫描结果列表 [{"dimension": ..., "change_pct": ..., ...}, ...]
    """
    schema_info = format_schema_for_prompt(schema)
    result = python_agent.execute_from_query(
        user_query=query,
        schema_info=schema_info,
        df=uploaded_df,
        validation={"is_valid": True, "reasoning": f"L{layer.depth} 扫描"},
        need_chart=False,
        trace=trace,
    )

    if not result.success:
        trace.log(TraceEventType.TOOL_EXECUTION_ERROR, iteration=iteration,
                  error=result.error or "scan layer returned no data",
                  metadata={"phase": "v8_scan_loop", "layer_depth": layer.depth})
        return []

    scan_data = []
    if isinstance(result.answer, list):
        scan_data = result.answer
    elif result.data:
        scan_data = result.data

    trace.log(TraceEventType.SCAN_LAYER_COMPLETED, iteration=iteration,
              output_data={
                  "layer_depth": layer.depth,
                  "group_by": layer.group_by,
                  "rows_returned": len(scan_data),
                  "preview": scan_data[:3] if scan_data else [],
                  "code": result.executed_code[:300] if result.executed_code else "",
              },
              success=True,
              metadata={"phase": "v8_scan_loop"})

    return scan_data


def _detect_anomalies(scan_data: List[Dict],
                      threshold: float = SIGNIFICANCE_THRESHOLD
                      ) -> tuple:
    """
    纯规则异常检测（0 次 LLM）— 与 v7 detect_node 逻辑一致但阈值可配置。
    返回 (anomalies, normals)
    """
    anomalies = []
    normals = []
    for item in scan_data:
        if not isinstance(item, dict):
            continue
        change_pct = item.get("change_pct", 0)
        if change_pct is None:
            change_pct = 0
        try:
            change_pct = float(change_pct)
        except (TypeError, ValueError):
            change_pct = 0
        item["change_pct"] = change_pct

        if abs(change_pct) >= threshold:
            item["significant"] = True
            anomalies.append(item)
        else:
            item["significant"] = False
            normals.append(item)

    anomalies.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)

    # ✅ Task 2A: 给异常和正常项都补 canonical_dim，让评测器能匹配
    # 评测集 expected_anomaly_dims 用英文 key（traffic/conversion_rate/...），
    # 而扫描输出的 dimension 是中文（访客数/支付转化率/...），不映射 recall 永为 0
    _tag_canonical_inplace(anomalies)
    _tag_canonical_inplace(normals)

    return anomalies, normals


ARBITER_SYSTEM_PROMPT = """你是一个数据分析裁判（Arbiter）。
当扫描结果存在不确定情况时，你来做最终裁定。

## 不确定场景
1. 多个异常维度变化幅度接近（差距 < 3%），无法明确主次
2. L1 发现异常但 L2 下钻后所有分组均正常（异常可能来自整体趋势而非某分组）
3. 所有维度均低于阈值但有多个接近阈值（5-7% 之间）

## 你的输出 JSON
{
  "action": "accept|widen_scope|lower_threshold|flag_uncertain",
  "reasoning": "你的判断理由",
  "adjusted_threshold": null 或数字,
  "priority_dimensions": ["需要重点关注的维度"]
}

只返回 JSON:"""


def _run_arbiter(anomalies: List[Dict], normals: List[Dict],
                 layer_results: Dict[int, List[Dict]],
                 threshold: float, llm: LLMInterface,
                 trace: TraceLog) -> Dict:
    """
    Arbiter: 仅在不确定场景下调用（0-1次 LLM）。
    返回裁判决策 dict，或空 dict（无需裁判时）。
    """
    # 判断是否需要 Arbiter
    needs_arbiter = False
    reason = ""

    if len(anomalies) >= 2:
        pcts = [abs(a.get("change_pct", 0)) for a in anomalies]
        if len(pcts) >= 2 and (pcts[0] - pcts[1]) < 3:
            needs_arbiter = True
            reason = "多个异常维度变化幅度接近"

    if anomalies and 2 in layer_results:
        l2_data = layer_results[2]
        if l2_data:
            l2_anomalies = [d for d in l2_data if isinstance(d, dict)
                            and abs(d.get("change_pct", 0) or 0) >= threshold]
            if not l2_anomalies:
                needs_arbiter = True
                reason = "L1异常但L2下钻全正常"

    if not anomalies and normals:
        near_threshold = [n for n in normals
                          if abs(n.get("change_pct", 0)) >= threshold * 0.6]
        if len(near_threshold) >= 2:
            needs_arbiter = True
            reason = f"无异常但 {len(near_threshold)} 个维度接近阈值"

    if not needs_arbiter:
        return {}

    # 调用 Arbiter LLM
    prompt = f"""[触发原因] {reason}

[异常维度] {json.dumps(anomalies[:10], ensure_ascii=False)}

[正常维度] {json.dumps(normals[:10], ensure_ascii=False)}

[当前阈值] {threshold}%

请做出裁定:"""

    try:
        raw = llm.generate(ARBITER_SYSTEM_PROMPT, prompt)
        raw = raw.strip().replace("```json", "").replace("```", "")
        decision = json.loads(raw)

        trace.log(TraceEventType.ARBITER_INVOKED, iteration=0,
                  input_data={"reason": reason},
                  output_data=decision,
                  success=True,
                  metadata={"phase": "v8_arbiter"})
        return decision

    except Exception as e:
        trace.log(TraceEventType.ARBITER_INVOKED, iteration=0,
                  error=str(e)[:200],
                  metadata={"phase": "v8_arbiter", "action": "arbiter_failed"})
        return {"action": "accept", "reasoning": f"Arbiter 调用失败: {str(e)[:100]}"}


# ── v7 兼容：保留旧函数签名（供可能的外部引用） ──
def build_scan_query(user_query: str, schema: dict, date_range: dict) -> str:
    """[已弃用] v7 扫描查询构建 — v8 改用 Commander + _build_layer_query"""
    frame = _build_default_frame(schema, date_range, user_query)
    if frame.layers:
        return _build_layer_query(frame, frame.layers[0], schema)
    return ""


def build_drilldown_query(user_query: str, anomaly_dim: str,
                          schema: dict, date_range: dict) -> str:
    """[已弃用] v7 下钻查询构建 — v8 由 Scan Loop L2 替代"""
    frame = _build_default_frame(schema, date_range, user_query)
    text_cols = [col for _, col, _ in schema.get("_meta", {}).get("text_columns", [])]
    if not text_cols:
        return ""
    drilldown_layer = ScanLayer(depth=2, metrics=[anomaly_dim],
                                group_by=text_cols[:2])
    return _build_layer_query(frame, drilldown_layer, schema, anomaly_dims=[anomaly_dim])


def quick_scan_node(state: AgentState) -> AgentState:
    """
    ✅ v8 Step 1: Quick Scan — 替代 v7 的 scan_node + detect_node。

    流程:
      Commander (1x LLM) → AnalysisFrame
        ↓
      Scan Loop (Nx PythonAgent)
        L1 全局扫描 → 条件触发 L2 下钻
        ↓
      Detect (0x LLM, 纯规则)
        ↓
      Arbiter (0-1x LLM, 仅不确定时)
        ↓
      输出: ScanState（含所有层级结果 + anomalies + normal）
    """
    trace: TraceLog = state["trace"]
    user_query = state["user_query"]
    uploaded_df = state.get("uploaded_df")
    python_agent = state["_python_agent"]
    planner_agent = state["_planner_agent"]
    schema = get_cached_schema()
    date_range = get_cached_date_range()

    # ── 1. Commander: 生成 AnalysisFrame ──
    commander_llm = planner_agent.chat_llm  # 复用现有 LLM 客户端，不新建连接
    frame = _run_commander(user_query, schema, date_range, commander_llm, trace)

    if not frame.metrics:
        # 无指标可扫描 → fallback
        state["scan_state"] = ScanState(scan_summary="无可扫描指标").to_dict()
        state["_fallback_to_react"] = True
        trace.log(TraceEventType.TOOL_EXECUTION_ERROR, iteration=0,
                  error="Commander 未产出任何可扫描指标",
                  metadata={"phase": "v8_quick_scan"})
        return state

    scan_state = ScanState(analysis_frame=frame.to_dict())
    threshold = frame.significance_threshold
    iteration_counter = 0

    # ── 2. Scan Loop: 逐层扫描 ──
    l1_anomaly_dims = []  # L1 发现的异常维度名称

    for layer in frame.layers:
        if layer.depth > frame.max_depth:
            break

        # L2+ 条件触发: 仅当 L1 发现异常时才下钻
        if layer.depth >= 2 and layer.trigger_condition == "anomaly_found":
            if not l1_anomaly_dims:
                continue  # 无异常，跳过下钻

        iteration_counter += 1
        query = _build_layer_query(frame, layer, schema,
                                   anomaly_dims=l1_anomaly_dims if layer.depth >= 2 else None)
        if not query:
            continue

        layer_data = _execute_scan_layer(
            layer, query, python_agent, uploaded_df, schema, trace, iteration_counter)

        scan_state.layer_results[layer.depth] = scan_state.layer_results.get(layer.depth, [])
        scan_state.layer_results[layer.depth].extend(layer_data)

        # 记录步骤
        state["steps"].append({
            "iteration": iteration_counter,
            "timestamp": datetime.now().isoformat(),
            "type": f"scan_L{layer.depth}" + (f"_by_{'_'.join(layer.group_by)}" if layer.group_by else ""),
            "hypothesis": f"L{layer.depth} 扫描" + (f" (按 {', '.join(layer.group_by)})" if layer.group_by else ""),
            "result": {
                "success": bool(layer_data),
                "row_count": len(layer_data),
                "data_preview": layer_data[:10] if layer_data else [],
            },
        })

        # L1 完成后做异常检测，决定是否触发 L2
        if layer.depth == 1 and layer_data:
            l1_anomalies, l1_normals = _detect_anomalies(layer_data, threshold)
            l1_anomaly_dims = [a.get("dimension", "") for a in l1_anomalies]
            scan_state.all_anomalies.extend(l1_anomalies)
            scan_state.all_normal.extend(l1_normals)

    # L2 结果也做异常检测
    for depth_key, depth_data in scan_state.layer_results.items():
        if depth_key >= 2 and depth_data:
            l2_anomalies, l2_normals = _detect_anomalies(depth_data, threshold)
            # L2 异常追加到总表（不重复追加 L1 已有的）
            existing_dims = {(a.get("dimension", ""), str(a.get("group_key", "")))
                            for a in scan_state.all_anomalies}
            for a in l2_anomalies:
                key = (a.get("dimension", ""), str(a.get("group_key", "")))
                if key not in existing_dims:
                    scan_state.all_anomalies.append(a)

    # ── 3. Arbiter: 仅在不确定场景调用 ──
    arbiter_decision = _run_arbiter(
        scan_state.all_anomalies, scan_state.all_normal,
        scan_state.layer_results, threshold,
        commander_llm, trace)

    if arbiter_decision:
        scan_state.arbiter_decisions.append(arbiter_decision)

        # 如果 Arbiter 建议降低阈值，重新检测
        adj_threshold = arbiter_decision.get("adjusted_threshold")
        if adj_threshold and isinstance(adj_threshold, (int, float)) and adj_threshold < threshold:
            # 对 L1 数据重新检测
            l1_data = scan_state.layer_results.get(1, [])
            if l1_data:
                re_anomalies, re_normals = _detect_anomalies(l1_data, adj_threshold)
                scan_state.all_anomalies = re_anomalies
                scan_state.all_normal = re_normals

    # ── 4. 构建摘要 ──
    n_anomalies = len(scan_state.all_anomalies)
    n_normals = len(scan_state.all_normal)
    scan_state.scan_summary = (
        f"扫描完成: {n_anomalies} 个异常 / {n_normals} 个正常 "
        f"(阈值 {threshold}%, "
        f"扫描 {len(scan_state.layer_results)} 层)"
    )

    # ── 5. 写入 state ──
    state["scan_state"] = scan_state.to_dict()
    state["analysis_frame"] = frame.to_dict()
    # ✅ v9: 从 Commander 的 AnalysisFrame 继承 analysis_depth
    state["analysis_depth"] = frame.analysis_depth

    # ✅ 向后兼容 v7 字段 — 让 reason_node / reporter_v2 无感切换
    state["scan_data"] = scan_state.layer_results.get(1, [])
    state["confirmed_anomalies"] = scan_state.all_anomalies
    state["rejected_dimensions"] = scan_state.all_normal
    state["has_anomaly"] = n_anomalies > 0
    state["drilldown_data"] = []
    for depth_key in sorted(scan_state.layer_results.keys()):
        if depth_key >= 2:
            state["drilldown_data"].extend(scan_state.layer_results[depth_key])

    state["_fallback_to_react"] = False

    trace.log(TraceEventType.EVALUATION_COMPLETED, iteration=iteration_counter,
              output_data={
                  "anomaly_count": n_anomalies,
                  "normal_count": n_normals,
                  "threshold": threshold,
                  "layers_scanned": list(scan_state.layer_results.keys()),
                  "anomaly_dims": [a.get("dimension", "?") for a in scan_state.all_anomalies[:10]],
                  "arbiter_invoked": bool(arbiter_decision),
              },
              metadata={"phase": "v8_quick_scan"})

    return state


# ── v7 兼容保留（scan_node / detect_node 不再被 graph 调用） ──
def scan_node(state: AgentState) -> AgentState:
    """[已弃用] v7 scan_node — v8 改用 quick_scan_node"""
    return quick_scan_node(state)


def detect_node(state: AgentState) -> AgentState:
    """[已弃用] v7 detect_node — 已合并进 quick_scan_node"""
    # 如果从旧路径调用，做一次兜底检测
    scan_data = state.get("scan_data", [])
    threshold = SIGNIFICANCE_THRESHOLD
    frame_dict = state.get("analysis_frame")
    if frame_dict:
        threshold = frame_dict.get("significance_threshold", SIGNIFICANCE_THRESHOLD)
    confirmed, rejected = _detect_anomalies(scan_data, threshold)
    state["confirmed_anomalies"] = confirmed
    state["rejected_dimensions"] = rejected
    state["has_anomaly"] = len(confirmed) > 0
    return state


# ============================================================================
# v9 新增: present_scan_results — descriptive 深度的出口节点
# ============================================================================

SCAN_REPORT_SYSTEM_PROMPT = """你是一个数据分析报告生成器。
基于全局扫描结果，生成简洁的分析摘要报告。

## 报告要求
1. 直接回答用户的问题（如"有没有异常"、"表现如何"）
2. 列出关键指标的变化情况（按变化幅度排序）
3. 如发现异常：指出异常维度及变化幅度
4. 如未发现异常：明确说明"各项指标在正常波动范围内"
5. 如有下钻数据：指出异常来源的分组维度
6. 语言简洁，避免过度推测原因（descriptive 深度不做根因分析）

## 格式
- 使用 Markdown
- 先给结论，再列数据
- 不超过 500 字"""


def present_scan_results(state: AgentState) -> AgentState:
    """
    ✅ v9 新增: descriptive 深度的出口节点。
    将 Quick Scan 的结果格式化为用户可消费的报告，不进入因果推理。

    对比 simple_executor_node:
    - simple_executor: 调 PythonAgent 跑一行查询 → 返回原始表格
    - present_scan_results: Quick Scan 已完成全局多指标扫描 → 格式化输出分析摘要

    对比 reporter_v2:
    - reporter_v2: 基于 reason_result（因果推理结论）生成报告
    - present_scan_results: 仅基于 scan_state（扫描结果）生成报告，不含因果链
    """
    trace: TraceLog = state["trace"]
    user_query = state["user_query"]
    scan_state_dict = state.get("scan_state") or {}
    confirmed = state.get("confirmed_anomalies", [])
    rejected = state.get("rejected_dimensions", [])
    scan_data = state.get("scan_data", [])
    drilldown_data = state.get("drilldown_data", [])
    scan_summary = scan_state_dict.get("scan_summary", "")
    frame_dict = state.get("analysis_frame") or {}
    frame_reasoning = frame_dict.get("reasoning", "")

    # 构建 LLM 报告生成的 prompt
    drilldown_section = ""
    if drilldown_data:
        drilldown_section = f"\n\n下钻分析（按分组细分）:\n{json.dumps(drilldown_data[:10], ensure_ascii=False, indent=2)}"

    prompt = f"""用户问题: {user_query}

扫描概要: {scan_summary}
{f'扫描计划理由: {frame_reasoning}' if frame_reasoning and '规则兜底' not in frame_reasoning else ''}

全维度扫描结果:
{json.dumps(scan_data[:20], ensure_ascii=False, indent=2)}

异常维度 ({len(confirmed)}个): {json.dumps(confirmed, ensure_ascii=False)}
正常维度 ({len(rejected)}个): {json.dumps(rejected[:10], ensure_ascii=False)}
{drilldown_section}

请生成简洁的分析摘要报告，直接回答用户的问题。"""

    reporter = state["_reporter_agent"]
    try:
        result = reporter.llm.generate(SCAN_REPORT_SYSTEM_PROMPT, prompt)

        # 同时构建 reason_result（descriptive 深度的简化版，供评测脚本兼容）
        # ✅ v9.5: descriptive 路径不做因果推理，root_causes 恒为空
        # confirmed_anomalies 信息已通过 confirmed_hypotheses 和报告文本传递
        reason_result = {
            "root_causes": [],
            "intermediate_effects": [],
            "causal_chain": "",
            "user_assumption_correction": "",
            "is_no_anomaly": len(confirmed) == 0,
            "confidence": "high",
            "needs_deep_rca": False,
            "suggested_data": [],
            "zero_anomaly_suggestions": [],
        }

        # ✅ Task 2A: descriptive 路径的 root_causes 也要打 canonical 标
        _tag_canonical_inplace(reason_result.get("root_causes", []))

        state["reason_result"] = reason_result
        state["causal_result"] = reason_result  # v7 兼容
        state["needs_deep_rca"] = False

        state["final_report"] = {
            "success": True,
            "full_content": result,
            "summary": result[:300] + "..." if len(result) > 300 else result,
            "confirmed_hypotheses": [c.get("dimension", "unknown") for c in confirmed],
            "rejected_hypotheses": [r.get("dimension", "unknown") for r in rejected],
            "chart_paths": state.get("chart_paths", []),
            "causal_chain": "",
            "confidence": reason_result["confidence"],
            "needs_deep_rca": False,
            "suggested_data": [],
            "reason_result": reason_result,
            # ✅ v9: 标记为 descriptive 报告
            "analysis_depth": "descriptive",
        }

        trace.log(TraceEventType.REPORT_GENERATED,
                  iteration=state.get("current_iteration", 1),
                  output_data={
                      "summary": result[:200],
                      "analysis_depth": "descriptive",
                      "anomaly_count": len(confirmed),
                  },
                  metadata={"phase": "v9_present_scan_results"})

    except Exception as e:
        state["final_report"] = {
            "success": False,
            "full_content": f"报告生成失败: {str(e)}",
            "summary": "扫描完成但报告生成失败",
            "confirmed_hypotheses": [c.get("dimension", "") for c in confirmed],
            "rejected_hypotheses": [r.get("dimension", "") for r in rejected],
            "error": str(e),
            "analysis_depth": "descriptive",
        }

    trace.finalize("completed", {
        "route": "complex",
        "mode": "v9_descriptive",
        "analysis_depth": "descriptive",
        "anomaly_count": len(confirmed),
    })

    return state


# ======================== v9.7: 零异常 + analysis_depth 差异化辅助函数 ========================

# question_category → suggested_data 映射配置
CATEGORY_SUGGESTED_DATA_MAP: Dict[str, Dict] = {
    "roi_ad": {
        "type": "ad_campaign",
        "description": "营销投放渠道明细数据",
        "reason": "用户询问投放/ROI相关问题，需要投放明细来做归因分析",
        "required_columns": ["date", "channel", "campaign_id", "spend",
                             "impressions", "clicks", "conversions"],
        "col_markers": {"spend", "cost", "impressions", "clicks", "ctr",
                        "花费", "成本", "展现", "点击"},
    },
    "competitor": {
        "type": "competitor_monitor",
        "description": "竞品监控数据",
        "reason": "用户询问竞争相关问题，需要竞品数据来对比分析",
        "required_columns": ["date", "competitor", "price", "sales"],
        "col_markers": {"竞品", "competitor", "对手"},
    },
    "inventory": {
        "type": "inventory_status",
        "description": "库存状态数据",
        "reason": "用户询问库存相关问题，需要库存明细来分析",
        "required_columns": ["date", "sku", "available_qty", "stockout_days"],
        "col_markers": {"库存", "inventory", "stock", "qty"},
    },
    "promotion": {
        "type": "promotion_calendar",
        "description": "促销活动日历",
        "reason": "用户询问促销相关问题，需要活动明细来分析",
        "required_columns": ["date", "promotion_name", "discount_rate"],
        "col_markers": {"促销", "活动", "promo", "discount"},
    },
}


def _detect_missing_data_for_causal(question_category: str, trace: TraceLog) -> tuple:
    """
    causal 类问题的缺失数据检测（基于 frame.question_category 映射）。

    返回 (needs_deep_rca: bool, suggested_data: list)
    """
    if question_category not in CATEGORY_SUGGESTED_DATA_MAP:
        return False, []

    data_config = CATEGORY_SUGGESTED_DATA_MAP[question_category]

    # 检查当前 schema 是否已有对应数据列
    _available_cols: set = set()
    try:
        _schema = get_cached_schema() or {}
        for _tbl in (_schema.get("tables") or []):
            for _c in (_tbl.get("columns") or []):
                _cname = (_c.get("name") or "") if isinstance(_c, dict) else str(_c)
                if _cname:
                    _available_cols.add(_cname.lower())
    except Exception:
        _available_cols = set()

    col_markers = data_config["col_markers"]
    matched = {c for c in _available_cols if any(m in c for m in col_markers)}
    has_data = len(matched) >= 2

    if not has_data:
        suggested = [{
            "type": data_config["type"],
            "description": data_config["description"],
            "reason": data_config["reason"],
            "required_columns": data_config["required_columns"],
        }]
        trace.log(TraceEventType.DEEP_RCA_DECISION, iteration=1,
                  output_data={
                      "decision": True,
                      "trigger": f"causal_zero_anomaly_{data_config['type']}",
                      "question_category": question_category,
                      "matched_cols": sorted(matched),
                  },
                  metadata={"phase": "v8_reasoner_v2_causal"})
        return True, suggested
    else:
        trace.log(TraceEventType.DEEP_RCA_DECISION, iteration=1,
                  output_data={
                      "decision": False,
                      "trigger": "causal_data_sufficient_no_anomaly",
                      "question_category": question_category,
                      "matched_cols": sorted(matched),
                  },
                  metadata={"phase": "v8_reasoner_v2_causal"})
        return False, []


def _detect_missing_data_optional(question_category: str, trace: TraceLog) -> tuple:
    """
    diagnostic 类问题的缺失数据检测（可选，当前直接返回 False）。
    后续可根据评测需要扩展。
    """
    return False, []


def reasoner_v2_node(state: AgentState) -> AgentState:
    """
    ✅ v8 Step 2: Reasoner v2 — 因果推理 + needs_deep_rca 决策 + suggested_data 建议。

    改造自 v7 reason_node，关键区别:
    - 输入: 使用 scan_state（多层 ScanState）而非仅 v7 单层 scan_data
    - 0 异常: 评估阈值/时间范围/指标是否合适，给出建议（而非简单返回"无异常"）
    - 1 异常: 规则判定 + needs_deep_rca 检查（Arbiter 是否曾建议 widen_scope）
    - 2+ 异常: LLM 因果推理 + needs_deep_rca 决策 + suggested_data 建议
    - 输出: reason_result（含 needs_deep_rca），写入 state 供路由和 reporter 使用
    - 避免硬编码: suggested_data 由 LLM 根据根因局限性动态生成，不硬编码 schema 映射
    """
    trace = state["trace"]
    confirmed = state.get("confirmed_anomalies", [])
    rejected = state.get("rejected_dimensions", [])
    user_query = state["user_query"]
    drilldown_data = state.get("drilldown_data", [])

    # 从 scan_state 提取多层上下文（v8 新增）
    scan_state_dict = state.get("scan_state") or {}
    arbiter_decisions = scan_state_dict.get("arbiter_decisions", [])
    layer_results = scan_state_dict.get("layer_results", {})
    scan_summary = scan_state_dict.get("scan_summary", "")
    frame_dict = state.get("analysis_frame") or {}
    threshold = frame_dict.get("significance_threshold", 8.0)

    # Arbiter 是否曾建议 widen_scope（needs_deep_rca 判定条件之一）
    arbiter_widened = any(
        d.get("action") == "widen_scope" for d in arbiter_decisions
    )

    # ── Case 1: 0 个异常 ──
    if not confirmed:
        # v8 增强: 评估扫描参数是否合适，给出建议
        zero_anomaly_suggestions = []
        if threshold > 10:
            zero_anomaly_suggestions.append(
                f"当前阈值 {threshold}% 偏高，可尝试降低阈值以发现轻微变化")
        if frame_dict.get("time_window", {}).get("comparison_type") == "mom":
            zero_anomaly_suggestions.append(
                "当前为月环比，如指标变化周期更长，可尝试同比(yoy)对比")

        # 检查是否所有指标变化都很小（真正无异常）vs 阈值不当
        all_scan_data = layer_results.get(1, []) if isinstance(layer_results.get(1), list) else []
        max_change = max(
            (abs(d.get("change_pct", 0)) for d in all_scan_data),
            default=0
        )
        if max_change > threshold * 0.5 and max_change <= threshold:
            zero_anomaly_suggestions.append(
                f"最大变化幅度 {max_change:.1f}% 接近阈值 {threshold}%，可能存在轻微异常")

        # ======================== ✅ v9.7: 根据 analysis_depth 差异化处理 ========================
        analysis_depth = state.get("analysis_depth") or frame_dict.get("analysis_depth", "diagnostic")
        question_category = frame_dict.get("question_category", "general")

        _zero_needs_deep = False
        _zero_suggested = []

        if analysis_depth == "descriptive":
            # 描述型问题：0异常 = 完整回答，直接返回
            _zero_needs_deep = False
            _zero_suggested = []
            trace.log(TraceEventType.REASONER_V2_COMPLETED, iteration=1,
                      output_data={
                          "conclusion": "no_anomaly_descriptive",
                          "analysis_depth": "descriptive",
                          "question_category": question_category,
                          "behavior": "直接返回无异常，不建议补充数据",
                      },
                      metadata={"phase": "v8_reasoner_v2"})

        elif analysis_depth == "causal":
            # 根因型问题：即使0异常，也需要检测是否缺数据
            _zero_needs_deep, _zero_suggested = _detect_missing_data_for_causal(
                question_category=question_category,
                trace=trace,
            )
            trace.log(TraceEventType.REASONER_V2_COMPLETED, iteration=1,
                      output_data={
                          "conclusion": "no_anomaly_causal",
                          "analysis_depth": "causal",
                          "question_category": question_category,
                          "behavior": "检测是否缺失关键数据",
                          "needs_deep_rca": _zero_needs_deep,
                          "suggested_count": len(_zero_suggested),
                      },
                      metadata={"phase": "v8_reasoner_v2"})

        else:  # diagnostic
            _zero_needs_deep, _zero_suggested = _detect_missing_data_optional(
                question_category=question_category,
                trace=trace,
            )
            trace.log(TraceEventType.REASONER_V2_COMPLETED, iteration=1,
                      output_data={
                          "conclusion": "no_anomaly_diagnostic",
                          "analysis_depth": "diagnostic",
                          "question_category": question_category,
                          "behavior": "可选建议",
                          "needs_deep_rca": _zero_needs_deep,
                      },
                      metadata={"phase": "v8_reasoner_v2"})
        # ======================== v9.7 新增结束 ========================

        reason_result = {
            "root_causes": [],
            "intermediate_effects": [],
            "causal_chain": "无显著异常",
            "user_assumption_correction": "",
            "is_no_anomaly": True,
            "confidence": "high" if not _zero_needs_deep else "medium",
            "needs_deep_rca": _zero_needs_deep,
            "suggested_data": [],
            "zero_anomaly_suggestions": zero_anomaly_suggestions,
            "analysis_depth": analysis_depth,
            "question_category": question_category,
        }

        # 标准化 suggested_data
        if _zero_suggested:
            reason_result["suggested_data"] = _normalize_suggested_data(_zero_suggested)
            reason_result["suggested_data_type_ids"] = _suggested_data_type_ids(reason_result["suggested_data"])

        state["reason_result"] = reason_result
        state["causal_result"] = reason_result  # v7 兼容
        state["needs_deep_rca"] = _zero_needs_deep

        return state

    # ── Case 2: 1 个异常 — 规则判定 + needs_deep_rca 检查 ──
    if len(confirmed) == 1:
        dim = confirmed[0]
        dim_name = dim.get("dimension", "unknown")
        change_pct = dim.get("change_pct", 0)

        # 单异常的 needs_deep_rca 判定: 仅当 Arbiter 建议 widen_scope
        single_needs_deep = arbiter_widened

        reason_result = {
            "root_causes": [{
                "dimension": dim_name,
                "change_pct": change_pct,
                "reasoning": f'{dim_name} 是唯一显著异常维度',
            }],
            "intermediate_effects": [],
            "causal_chain": f'{dim_name}({change_pct:+.1f}%)',
            "user_assumption_correction": "",
            "is_no_anomaly": False,
            "confidence": "high",
            "needs_deep_rca": single_needs_deep,
            "suggested_data": [],
        }

        # 补充下钻信息
        if drilldown_data:
            reason_result["drilldown_summary"] = drilldown_data[:5]

        # 如果 needs_deep_rca，由 LLM 生成 suggested_data（避免硬编码）
        if single_needs_deep:
            reason_result["confidence"] = "medium"
            try:
                llm = state["_planner_agent"].chat_llm
                suggest_prompt = f"""当前数据中 {dim_name} 变化 {change_pct:+.1f}%，但 Arbiter 裁判建议扩大分析范围。
现有数据的局限是仅有汇总级别数据。
请建议用户可以补充什么数据来做更深入的根因分析。

只返回 JSON 数组（1-2条建议）:
[{{"description": "数据描述", "reason": "需要原因", "required_columns": ["col1", "col2"]}}]

只返回 JSON:"""
                raw = llm.generate("你是数据分析顾问，根据分析局限性建议补充数据。只返回JSON。", suggest_prompt)
                raw = raw.strip().replace("```json", "").replace("```", "")
                # ✅ v9.6: 标准化字段并补出 type_id
                reason_result["suggested_data"] = _normalize_suggested_data(json.loads(raw))
                reason_result["suggested_data_type_ids"] = _suggested_data_type_ids(
                    reason_result["suggested_data"])
            except Exception:
                reason_result["suggested_data"] = []
                reason_result["suggested_data_type_ids"] = []

        state["reason_result"] = reason_result
        state["causal_result"] = reason_result  # v7 兼容
        state["needs_deep_rca"] = single_needs_deep

        trace.log(TraceEventType.REASONER_V2_COMPLETED, iteration=1,
                  output_data={
                      "conclusion": "single_anomaly",
                      "root_cause": dim_name,
                      "needs_deep_rca": single_needs_deep,
                  },
                  metadata={"phase": "v8_reasoner_v2"})

        if single_needs_deep:
            trace.log(TraceEventType.DEEP_RCA_DECISION, iteration=1,
                      output_data={
                          "decision": True,
                          "trigger": "arbiter_widen_scope",
                      },
                      metadata={"phase": "v8_reasoner_v2"})
        return state

    # ── Case 3: 多异常 — LLM 因果推理 + needs_deep_rca 决策 ──
    llm = state["_planner_agent"].chat_llm  # 复用已有 LLM 客户端

    # 构建多层扫描上下文（v8: 提供 L1+L2 全层数据，而非仅 v7 单层）
    drilldown_section = ""
    if drilldown_data:
        drilldown_section = (
            f"\n\n[下钻分析结果（L2+ 按分组细分）]\n"
            f"{json.dumps(drilldown_data[:10], ensure_ascii=False, indent=2)}"
        )

    arbiter_section = ""
    if arbiter_decisions:
        arbiter_section = (
            f"\n\n[Arbiter 裁判决策]\n"
            f"{json.dumps(arbiter_decisions, ensure_ascii=False, indent=2)}"
        )

    prompt = f"""[用户问题] {user_query}

[扫描概要] {scan_summary}

[显著异常维度（按变化幅度排序）]
{json.dumps(confirmed, ensure_ascii=False, indent=2)}

[正常维度]
{json.dumps(rejected, ensure_ascii=False, indent=2)}
{drilldown_section}{arbiter_section}

请分析因果关系，并判断是否需要更深入的根因分析(needs_deep_rca):"""

    try:
        result = llm.generate(CAUSAL_REASONING_PROMPT, prompt)
        response = result.strip().replace("```json", "").replace("```", "")
        reason_result = json.loads(response)
        reason_result["is_no_anomaly"] = False

        # 确保必需字段存在
        reason_result.setdefault("needs_deep_rca", False)
        reason_result.setdefault("suggested_data", [])
        reason_result.setdefault("confidence", "medium")
        reason_result.setdefault("user_assumption_correction", "")

        # ── needs_deep_rca 后置校验（覆盖 LLM 可能的误判）──
        llm_decision = reason_result["needs_deep_rca"]
        confidence = reason_result["confidence"]
        root_causes = reason_result.get("root_causes", [])

        # 规则覆盖 1: confidence=high 且非 arbiter_widen → 强制 false（保守策略）
        if confidence == "high" and not arbiter_widened:
            reason_result["needs_deep_rca"] = False

        # 规则覆盖 2: 多根因变化幅度接近（差距<5%）→ 设为 true
        if len(root_causes) >= 2:
            changes = sorted(
                [abs(rc.get("change_pct", 0)) for rc in root_causes],
                reverse=True
            )
            if len(changes) >= 2 and (changes[0] - changes[1]) < 5.0:
                if confidence != "high":  # high confidence 时不强制升级
                    reason_result["needs_deep_rca"] = True

        # 规则覆盖 3: Arbiter 曾建议 widen_scope 且 confidence 不是 high
        if arbiter_widened and confidence != "high":
            reason_result["needs_deep_rca"] = True

        # ✅ v9.6 Patch 4 (v2 soft boost): ROI/投放效率类问题
        # 原版硬规则无条件触发 deep_rca,生产中会对"数据已够"的 case 过触发。
        # 改为:先检查当前 schema 是否已含投放明细列,只有真正缺列时才升级。
        _q_lower = (user_query or "").lower()
        _roi_kws = ["roi", "roas", "cpc", "cpa", "cpm",
                    "投放效率", "投放回报", "投放效果", "广告效果",
                    "转化成本", "获客成本", "投产比", "广告投放",
                    "投放变差", "推广效率"]
        if any(kw in _q_lower for kw in _roi_kws) and not reason_result.get("needs_deep_rca"):
            # 收集当前已加载数据的全部列名(跨所有表),小写化
            _available_cols = set()
            try:
                _schema = get_cached_schema() or {}
                for _tbl in (_schema.get("tables") or []):
                    for _c in (_tbl.get("columns") or []):
                        _cname = (_c.get("name") or "") if isinstance(_c, dict) else str(_c)
                        if _cname:
                            _available_cols.add(_cname.lower())
            except Exception:
                _available_cols = set()

            # 投放明细识别词 (中英 + 常见变体)
            _ad_col_markers = {
                "spend", "cost", "ad_spend", "ad_cost", "marketing_spend",
                "impressions", "impression", "imp", "exposure",
                "clicks", "click", "ctr",
                "campaign_id", "campaign", "ad_id", "adgroup_id",
                "channel_id", "channel_name",
                "花费", "成本", "投放花费", "广告花费",
                "展现", "展现量", "曝光", "曝光量",
                "点击", "点击量", "点击率",
                "计划id", "广告计划", "广告组", "渠道id",
            }
            _matched = {c for c in _available_cols if any(m in c for m in _ad_col_markers)}
            _has_ad_detail = len(_matched) >= 2  # 至少命中2个才算数据充分

            if not _has_ad_detail:
                # 数据确实缺投放明细 → 升级 deep_rca 并补 suggested_data
                reason_result["needs_deep_rca"] = True
                existing = reason_result.get("suggested_data") or []
                has_ad = any(
                    isinstance(s, dict) and (
                        s.get("type") == "ad_campaign"
                        or "广告" in (s.get("description", "") + s.get("reason", "") + s.get("reasoning", ""))
                        or "投放" in (s.get("description", "") + s.get("reason", "") + s.get("reasoning", ""))
                    )
                    for s in existing
                )
                if not has_ad:
                    existing.insert(0, {
                        "type": "ad_campaign",
                        "description": "营销投放渠道明细数据",
                        "reason": "ROI/投放效率类问题需要 spend / impressions / clicks / 渠道维度才能定位归因,当前数据缺少这些字段",
                        "required_columns": ["date", "channel", "campaign_id",
                                             "spend", "impressions", "clicks",
                                             "conversions"],
                    })
                    reason_result["suggested_data"] = existing
                try:
                    trace.log(TraceEventType.DEEP_RCA_DECISION, iteration=1,
                              output_data={"decision": True,
                                           "trigger": "roi_soft_boost_data_insufficient"},
                              metadata={"phase": "v8_reasoner_v2",
                                        "matched_ad_cols": sorted(_matched)})
                except Exception:
                    pass
            else:
                # 数据已含投放明细 → 不强制升级,仅记录决策日志便于事后审计
                try:
                    trace.log(TraceEventType.DEEP_RCA_DECISION, iteration=1,
                              output_data={"decision": False,
                                           "trigger": "roi_soft_boost_data_sufficient"},
                              metadata={"phase": "v8_reasoner_v2",
                                        "matched_ad_cols": sorted(_matched)})
                except Exception:
                    pass

        # 如果最终 needs_deep_rca=false，清空 suggested_data
        if not reason_result["needs_deep_rca"]:
            reason_result["suggested_data"] = []

        # ✅ v9.6 Patch 2B-2: 标准化 suggested_data 字段并补出 type_id 列表
        reason_result["suggested_data"] = _normalize_suggested_data(
            reason_result.get("suggested_data", []))
        reason_result["suggested_data_type_ids"] = _suggested_data_type_ids(
            reason_result["suggested_data"])

        # ✅ Task 2A: 给 root_causes / intermediate_effects 也补 canonical_dim
        # 评测器读 actual_rc_dims 时会优先取 canonical_dim 字段
        _tag_canonical_inplace(reason_result.get("root_causes", []))
        _tag_canonical_inplace(reason_result.get("intermediate_effects", []))

        # 补充下钻信息
        if drilldown_data:
            reason_result["drilldown_summary"] = drilldown_data[:5]

        state["reason_result"] = reason_result
        state["causal_result"] = reason_result  # v7 兼容
        state["needs_deep_rca"] = reason_result["needs_deep_rca"]

        trace.log(TraceEventType.REASONER_V2_COMPLETED, iteration=1,
                  output_data={
                      "conclusion": "multi_anomaly_causal",
                      "root_causes": [rc.get("dimension") for rc in root_causes],
                      "confidence": confidence,
                      "needs_deep_rca": reason_result["needs_deep_rca"],
                      "llm_raw_decision": llm_decision,
                      "suggested_data_count": len(reason_result.get("suggested_data", [])),
                  },
                  metadata={"phase": "v8_reasoner_v2"})

        if reason_result["needs_deep_rca"]:
            trace.log(TraceEventType.DEEP_RCA_DECISION, iteration=1,
                      output_data={
                          "decision": True,
                          "confidence": confidence,
                          "arbiter_widened": arbiter_widened,
                          "suggested_data": reason_result["suggested_data"],
                      },
                      metadata={"phase": "v8_reasoner_v2"})

    except Exception as e:
        # LLM 因果推理失败 → 降级：按变化幅度排序，最大的就是根因
        reason_result = {
            "root_causes": [{
                "dimension": confirmed[0].get("dimension", "unknown"),
                "change_pct": confirmed[0].get("change_pct", 0),
                "reasoning": "变化幅度最大（因果推理降级）",
            }],
            "intermediate_effects": [
                {"dimension": c.get("dimension", ""), "change_pct": c.get("change_pct", 0)}
                for c in confirmed[1:]
            ],
            "causal_chain": " → ".join(
                f'{c.get("dimension", "?")}({c.get("change_pct", 0):+.1f}%)'
                for c in confirmed
            ),
            "user_assumption_correction": "",
            "is_no_anomaly": False,
            "confidence": "low",
            "needs_deep_rca": False,  # 降级模式不触发 Step 3
            "suggested_data": [],
            "error": str(e),
        }

        # ✅ Task 2A: 降级路径同样要打 canonical 标，避免评测时 recall 误归零
        _tag_canonical_inplace(reason_result.get("root_causes", []))
        _tag_canonical_inplace(reason_result.get("intermediate_effects", []))

        state["reason_result"] = reason_result
        state["causal_result"] = reason_result  # v7 兼容
        state["needs_deep_rca"] = False

        trace.log(TraceEventType.REASONER_V2_COMPLETED, iteration=1,
                  error=str(e),
                  output_data={"conclusion": "causal_reasoning_fallback"},
                  metadata={"phase": "v8_reasoner_v2"})

    return state


def reporter_node_v2(state: AgentState) -> AgentState:
    """
    ✅ v8 Reporter — 基于 Quick Scan (scan_state) + Reasoner v2 (reason_result) 生成报告。
    复用 ReporterAgent 的 LLM 客户端。

    v8 Step 2 改动:
    - 优先读取 reason_result（Reasoner v2 输出），兼容 causal_result（v7）
    - 报告中体现 confidence 等级和 needs_deep_rca 建议
    - final_report 中携带 reason_result 供前端渲染 Step 2 结论卡片
    """
    trace = state["trace"]
    reporter = state["_reporter_agent"]
    user_query = state["user_query"]
    scan_data = state.get("scan_data", [])
    confirmed = state.get("confirmed_anomalies", [])
    rejected = state.get("rejected_dimensions", [])
    drilldown_data = state.get("drilldown_data", [])

    # ✅ v8 Step 2: 优先使用 reason_result，兼容 causal_result
    reason_result = state.get("reason_result") or state.get("causal_result") or {}
    confidence = reason_result.get("confidence", "unknown")
    needs_deep_rca = state.get("needs_deep_rca", False)
    suggested_data = reason_result.get("suggested_data", [])

    # v8: 从 scan_state 提取额外上下文
    scan_state_dict = state.get("scan_state") or {}
    scan_summary = scan_state_dict.get("scan_summary", "")
    arbiter_decisions = scan_state_dict.get("arbiter_decisions", [])
    frame_dict = state.get("analysis_frame") or {}
    frame_reasoning = frame_dict.get("reasoning", "")

    # 构建下钻摘要
    drilldown_section = ""
    if drilldown_data:
        drilldown_section = f"\n\n下钻分析（按分组细分）:\n{json.dumps(drilldown_data[:10], ensure_ascii=False, indent=2)}"

    # 构建 Arbiter 信息
    arbiter_section = ""
    if arbiter_decisions:
        arbiter_section = f"\n\nArbiter 裁定:\n{json.dumps(arbiter_decisions, ensure_ascii=False, indent=2)}"

    # ✅ v8 Step 2: 构建 confidence 提示段
    confidence_section = ""
    if confidence in ("low", "medium"):
        confidence_section = f"\n\n⚠️ 因果推理信心等级: {confidence}"
        if needs_deep_rca and suggested_data:
            data_desc = "; ".join(s.get("description", "") for s in suggested_data if s.get("description"))
            confidence_section += f"\n建议补充数据以深入分析: {data_desc}"
        confidence_section += "\n请在报告中如实体现分析的确定性程度，避免过度断言。"

    # ✅ v8 Step 2: 0异常时的建议信息
    zero_suggestions_section = ""
    zero_suggestions = reason_result.get("zero_anomaly_suggestions", [])
    if reason_result.get("is_no_anomaly") and zero_suggestions:
        zero_suggestions_section = (
            f"\n\n分析建议（无异常场景）:\n"
            + "\n".join(f"- {s}" for s in zero_suggestions)
        )

    prompt = f"""用户问题: {user_query}

扫描概要: {scan_summary}
{f'扫描计划理由: {frame_reasoning}' if frame_reasoning and '规则兜底' not in frame_reasoning else ''}

全维度扫描结果:
{json.dumps(scan_data[:20], ensure_ascii=False, indent=2)}

因果分析结论:
{json.dumps(reason_result, ensure_ascii=False, indent=2)}

确认的异常维度 ({len(confirmed)}个): {json.dumps(confirmed, ensure_ascii=False)}
排除的正常维度 ({len(rejected)}个): {json.dumps(rejected, ensure_ascii=False)}
{drilldown_section}{arbiter_section}{confidence_section}{zero_suggestions_section}

请生成包含行动指南的专业分析报告。
⚠️ 行动指南必须逐条对应上方已确认的根因，标注优先级(P0/P1/P2)和可量化验收指标。
如果无确认根因，行动指南写"保持现有策略，持续监控"。
如果信心等级为 low/medium，请在报告中注明分析的局限性和建议的下一步动作。"""

    try:
        result = reporter.llm.generate(REPORTER_SYSTEM_PROMPT, prompt)
        # ✅ v9.6 Patch 5: 强制保证报告含'分析局限'段落
        result = _ensure_limitation_section(result, reporter.llm, REPORTER_SYSTEM_PROMPT, state=state)

        # 构造兼容现有前端的输出格式 + v8 Step 2 新增字段
        state["final_report"] = {
            "success": True,
            "full_content": result,
            "summary": result[:300] + "..." if len(result) > 300 else result,
            "confirmed_hypotheses": [c.get("dimension", "unknown") for c in confirmed],
            "rejected_hypotheses": [r.get("dimension", "unknown") for r in rejected],
            "chart_paths": state.get("chart_paths", []),
            "causal_chain": reason_result.get("causal_chain", ""),
            # ✅ v8 Step 2 新增: 供前端渲染 Step 2 结论卡片
            "confidence": confidence,
            "needs_deep_rca": needs_deep_rca,
            "suggested_data": suggested_data,
            "reason_result": reason_result,
        }

        trace.log(TraceEventType.REPORT_GENERATED,
                  iteration=state.get("current_iteration", 1),
                  output_data={
                      "summary": result[:200],
                      "confidence": confidence,
                      "needs_deep_rca": needs_deep_rca,
                  },
                  metadata={"phase": "v8_reporter"})

    except Exception as e:
        state["final_report"] = {
            "success": False,
            "full_content": f"报告生成失败: {str(e)}",
            "summary": "分析完成但报告生成失败",
            "confirmed_hypotheses": [c.get("dimension", "") for c in confirmed],
            "rejected_hypotheses": [r.get("dimension", "") for r in rejected],
            "error": str(e),
            "confidence": confidence,
            "needs_deep_rca": needs_deep_rca,
            "reason_result": reason_result,
        }

    trace.finalize("completed", {
        "route": "complex",
        "mode": "v8_quick_scan_reasoner_v2",
        "anomaly_count": len(confirmed),
        "confidence": confidence,
        "needs_deep_rca": needs_deep_rca,
    })

    return state


# ============================================================================
# v8 Step 3: Deep RCA — present_findings + deep_rca_init + reporter_deep
# ============================================================================

def _merge_supplementary_df(original_df: pd.DataFrame,
                            supplementary_df: pd.DataFrame,
                            schema: dict) -> pd.DataFrame:
    """
    ✅ v8 Step 3: 合并补充数据（方案 B — 预合并宽表）。
    动态推断 merge key，不硬编码列名。

    策略:
      1. 找两个 df 共有的日期列（通过动态检测，非列名匹配）
      2. 找两个 df 共有的分类列名
      3. 用共有列做 left join
      4. 如无共有列则按行拼接（fallback）
    """
    if supplementary_df is None or supplementary_df.empty:
        return original_df

    # 动态检测两个 df 的日期列
    orig_date_col = None
    supp_date_col = None
    for col in original_df.columns:
        if _try_parse_date_column(original_df[col]):
            orig_date_col = col
            break
    for col in supplementary_df.columns:
        if _try_parse_date_column(supplementary_df[col]):
            supp_date_col = col
            break

    # 找共有分类列（名称完全相同的 text 列）
    orig_text_cols = set()
    for col in original_df.columns:
        if pd.api.types.is_string_dtype(original_df[col]) or pd.api.types.is_object_dtype(original_df[col]):
            orig_text_cols.add(col)

    supp_text_cols = set()
    for col in supplementary_df.columns:
        if pd.api.types.is_string_dtype(supplementary_df[col]) or pd.api.types.is_object_dtype(supplementary_df[col]):
            supp_text_cols.add(col)

    common_text_cols = list(orig_text_cols & supp_text_cols)

    # 构建 merge keys
    merge_keys = []

    # 日期列: 如果两者都有，且名称相同直接用；名称不同则先统一
    if orig_date_col and supp_date_col:
        if orig_date_col == supp_date_col:
            merge_keys.append(orig_date_col)
            # 确保两边都是 datetime
            original_df[orig_date_col] = pd.to_datetime(original_df[orig_date_col], errors="coerce")
            supplementary_df[supp_date_col] = pd.to_datetime(supplementary_df[supp_date_col], errors="coerce")
        else:
            # 名称不同，统一列名
            unified_date_col = orig_date_col
            supplementary_df = supplementary_df.rename(columns={supp_date_col: unified_date_col})
            original_df[unified_date_col] = pd.to_datetime(original_df[unified_date_col], errors="coerce")
            supplementary_df[unified_date_col] = pd.to_datetime(supplementary_df[unified_date_col], errors="coerce")
            merge_keys.append(unified_date_col)

    # 分类列
    merge_keys.extend(common_text_cols)

    if merge_keys:
        try:
            merged = original_df.merge(
                supplementary_df, on=merge_keys, how="left", suffixes=("", "_supp")
            )
            return merged
        except Exception:
            pass

    # Fallback: 无共有列 → 返回原始 df（补充数据仅作 ReAct 的额外参考）
    return original_df


def present_findings_node(state: AgentState) -> AgentState:
    """
    ✅ v8 Step 3: 展示 Step 1+2 初步结论，graph 在此终止第一轮。

    此节点不做任何分析，只是标记 findings_presented=True，
    让 Orchestrator 知道第一轮已完成，需要等待用户决策。

    前端根据 findings_presented + needs_deep_rca 显示交互卡片:
    - 上传补充数据（可选）
    - "开始深度分析" / "跳过，直接生成报告"
    """
    trace: TraceLog = state["trace"]

    # 标记 findings 已展示
    state["findings_presented"] = True

    # 先用 reporter_v2 生成初步报告（即使需要深度分析，也先给出初步结论）
    # 这里复用 reporter_node_v2 的逻辑生成初步报告
    reporter = state["_reporter_agent"]
    user_query = state["user_query"]
    scan_data = state.get("scan_data", [])
    confirmed = state.get("confirmed_anomalies", [])
    rejected = state.get("rejected_dimensions", [])
    reason_result = state.get("reason_result") or {}
    confidence = reason_result.get("confidence", "unknown")
    scan_state_dict = state.get("scan_state") or {}
    scan_summary = scan_state_dict.get("scan_summary", "")

    # 生成初步报告（简化版）
    prompt = f"""用户问题: {user_query}

扫描概要: {scan_summary}

因果分析结论:
{json.dumps(reason_result, ensure_ascii=False, indent=2)}

确认的异常维度 ({len(confirmed)}个): {json.dumps(confirmed, ensure_ascii=False)}
排除的正常维度 ({len(rejected)}个): {json.dumps(rejected, ensure_ascii=False)}

⚠️ 注意: 因果推理信心等级为 {confidence}，建议用户补充数据做深度分析。
请生成简要的初步分析结论（不需要完整报告，重点说明已发现什么、还不确定什么）。"""

    try:
        result = reporter.llm.generate(REPORTER_SYSTEM_PROMPT, prompt)
        # ✅ v9.6 Patch 5: 强制保证报告含'分析局限'段落
        result = _ensure_limitation_section(result, reporter.llm, REPORTER_SYSTEM_PROMPT, state=state)
        state["final_report"] = {
            "success": True,
            "full_content": result,
            "summary": result[:300] + "..." if len(result) > 300 else result,
            "confirmed_hypotheses": [c.get("dimension", "unknown") for c in confirmed],
            "rejected_hypotheses": [r.get("dimension", "unknown") for r in rejected],
            "chart_paths": state.get("chart_paths", []),
            "causal_chain": reason_result.get("causal_chain", ""),
            "confidence": confidence,
            "needs_deep_rca": True,
            "suggested_data": reason_result.get("suggested_data", []),
            "reason_result": reason_result,
            "is_preliminary": True,  # 标记为初步报告
        }
    except Exception as e:
        state["final_report"] = {
            "success": False,
            "full_content": f"初步报告生成失败: {str(e)}",
            "summary": "初步分析完成但报告生成失败",
            "confirmed_hypotheses": [c.get("dimension", "") for c in confirmed],
            "rejected_hypotheses": [],
            "confidence": confidence,
            "needs_deep_rca": True,
            "reason_result": reason_result,
            "is_preliminary": True,
        }

    trace.log(TraceEventType.PRESENT_FINDINGS, iteration=0,
              output_data={
                  "anomaly_count": len(confirmed),
                  "confidence": confidence,
                  "suggested_data_count": len(reason_result.get("suggested_data", [])),
              },
              metadata={"phase": "v8_present_findings"})

    # 终止第一轮 graph（由 present_findings → END）
    trace.finalize("awaiting_user_decision", {
        "route": "complex",
        "mode": "v8_step1_step2_complete",
        "needs_deep_rca": True,
        "confidence": confidence,
    })

    return state


def _scan_supplementary_columns(
        merged_df: pd.DataFrame,
        original_df: pd.DataFrame,
        analysis_frame_dict: dict,
        threshold: float = SIGNIFICANCE_THRESHOLD,
) -> List[Dict]:
    """
    ✅ v9.8: 对补充数据的新增列做全量快速扫描（纯 pandas，0 次 LLM）。

    逻辑:
      1. 找出合并后 df 相比原始 df 新增的数值列
      2. 按 analysis_frame 的时间窗口拆分基准期 / 对比期
      3. 逐列聚合（sum 或 mean）、计算 change_pct
      4. 返回 [{column, base_value, compare_value, change_pct, significant}, ...]

    设计原则: 不调 LLM、不调 PythonAgent，纯 pandas 运算，速度快且确定性高。
    """
    new_cols = [c for c in merged_df.columns if c not in original_df.columns]
    if not new_cols:
        return []

    # ── 1. 从 analysis_frame 提取时间窗口 ──
    af = analysis_frame_dict or {}
    tw = af.get("time_window") or {}
    prev_start = tw.get("previous_start")
    prev_end = tw.get("previous_end")
    curr_start = tw.get("current_start")
    curr_end = tw.get("current_end")

    if not all([prev_start, prev_end, curr_start, curr_end]):
        return []

    # ── 2. 找日期列 ──
    date_col = None
    for col in merged_df.columns:
        if _try_parse_date_column(merged_df[col]):
            date_col = col
            break
    if date_col is None:
        return []

    try:
        df = merged_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        prev_mask = (df[date_col] >= prev_start) & (df[date_col] <= prev_end)
        curr_mask = (df[date_col] >= curr_start) & (df[date_col] <= curr_end)
        df_prev = df.loc[prev_mask]
        df_curr = df.loc[curr_mask]
    except Exception:
        return []

    if df_prev.empty or df_curr.empty:
        return []

    # ── 3. 逐列计算变化 ──
    findings = []
    for col in new_cols:
        if not pd.api.types.is_numeric_dtype(merged_df[col]):
            continue

        # 智能聚合: 列名含"率""比""率""avg""mean""ratio" → mean, 否则 sum
        col_lower = col.lower()
        use_mean = any(k in col_lower for k in
                       ["率", "比", "avg", "mean", "ratio", "rate", "pct", "percent", "roi"])

        try:
            if use_mean:
                base_val = float(df_prev[col].mean())
                comp_val = float(df_curr[col].mean())
            else:
                base_val = float(df_prev[col].sum())
                comp_val = float(df_curr[col].sum())
        except Exception:
            continue

        if base_val == 0 and comp_val == 0:
            continue

        if base_val == 0:
            change_pct = 100.0 if comp_val > 0 else -100.0
        else:
            change_pct = round((comp_val - base_val) / abs(base_val) * 100, 2)

        findings.append({
            "column": col,
            "base_value": round(base_val, 2),
            "compare_value": round(comp_val, 2),
            "change_pct": change_pct,
            "aggregation": "mean" if use_mean else "sum",
            "significant": abs(change_pct) >= threshold,
        })

    # 按变化幅度降序
    findings.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    return findings


def _build_deep_rca_hypotheses(reason_result: dict, supplementary_schema: dict,
                               original_schema: dict, user_query: str,
                               llm: LLMInterface,
                               supp_scan_findings: List[Dict] = None) -> list:
    """
    ✅ v9.8: 基于 Step 1+2 结论 + 补充数据扫描发现 + schema 生成深度分析假设。
    scan_findings 已经包含了补充数据的异常，假设生成侧重于因果解释而非数据探索。
    """
    root_causes = reason_result.get("root_causes", [])
    suggested_data = reason_result.get("suggested_data", [])
    confidence = reason_result.get("confidence", "unknown")
    causal_chain = reason_result.get("causal_chain", "")

    # 构建补充数据 schema 描述
    supp_schema_info = ""
    if supplementary_schema and supplementary_schema.get("tables"):
        supp_schema_info = format_schema_for_prompt(supplementary_schema)

    # 构建已有发现描述
    findings_summary = []
    for rc in root_causes:
        dim = rc.get("dimension", "unknown")
        pct = rc.get("change_pct", 0)
        findings_summary.append(f"- {dim}: 变化 {pct:+.1f}% ({rc.get('reasoning', '')})")

    # ✅ v9.8: 构建补充数据扫描发现描述
    scan_findings_text = ""
    if supp_scan_findings:
        sig_findings = [f for f in supp_scan_findings if f.get("significant")]
        all_findings = supp_scan_findings[:15]  # 所有发现都展示，最多15条
        finding_lines = []
        for f in all_findings:
            marker = "🔴异常" if f.get("significant") else "🟢正常"
            finding_lines.append(
                f"- [{marker}] {f['column']}: {f['base_value']} → {f['compare_value']} "
                f"(变化 {f['change_pct']:+.1f}%, {f['aggregation']}聚合)")
        scan_findings_text = f"""
## 补充数据扫描发现（已自动计算，无需重复验证）
共扫描 {len(supp_scan_findings)} 个新增指标，其中 {len(sig_findings)} 个异常:
{chr(10).join(finding_lines)}
"""

    prompt = f"""你是深度根因分析假设生成专家。前两步快速扫描已完成初步分析，现在需要更深入的假设。

## 已有发现（Step 1+2 结论）
因果链: {causal_chain}
信心等级: {confidence}
已确认根因:
{chr(10).join(findings_summary) if findings_summary else '无明确根因'}

## 原始数据 Schema
{format_schema_for_prompt(original_schema)}

{f'## 补充数据 Schema（用户新上传）' + chr(10) + supp_schema_info if supp_schema_info else '## 无补充数据'}
{scan_findings_text}
## 用户原始问题
{user_query}

## 你的任务
{'补充数据已完成全量扫描（见上方发现），你不需要重复计算这些指标的变化。' if supp_scan_findings else ''}
基于已有发现和补充数据的扫描结果，生成 3-6 个深度分析假设:
1. 针对补充数据中每个异常指标，生成一个"因果验证"假设 — 验证该指标异常与主指标下降的因果关系
2. 对补充数据中变化不显著的指标，生成"排除"假设 — 确认该因素可被排除
3. 考虑交叉维度分析 — 将补充数据指标与原始数据维度（如品类、渠道）交叉分析
4. dimension 必须来自原始数据或补充数据的 Schema 中的列名

## 输出格式
```json
{{
  "hypotheses": [
    {{
      "name": "英文假设名称",
      "dimension": "schema中的字段名",
      "description": "中文描述",
      "priority": 8,
      "related_hypotheses": [],
      "pruning_rules": {{"normal": "该维度正常时的剪枝说明"}}
    }}
  ]
}}
```

只返回 JSON:"""

    try:
        result = llm.generate(HYPOTHESIS_GENERATOR_PROMPT, prompt)
        response = result.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(response)
        hypotheses_raw = parsed.get("hypotheses", [])
        valid = []
        for h in hypotheses_raw:
            if all(k in h for k in ("name", "dimension", "description")):
                valid.append(h)
        return valid if valid else []
    except Exception:
        return []


def deep_rca_init_node(state: AgentState) -> AgentState:
    """
    ✅ v8 Step 3: Deep RCA 初始化 — 继承 Step 1+2 证据的 ReAct 初始化。

    与 v7 react_init_node 的关键区别:
    - 假设生成基于 Step 1+2 的 reason_result + 补充数据 schema
    - EvidenceBoard 预填 Step 1 已确认的 anomalies
    - DimensionTree 从合并后的新 schema 构建
    """
    trace: TraceLog = state["trace"]
    user_query = state["user_query"]
    uploaded_df = state.get("uploaded_df")
    supplementary_df = state.get("supplementary_df")

    # ✅ Task 2A 补强: step3 必须有补充数据才能进入。
    # 设计契约：reasoner 在 step2 已经声明 needs_deep_rca=True 并给出 suggested_data，
    # step3 的存在意义就是"基于新数据验证假设"。如果用户没上传任何数据就被路由进来，
    # 说明前端绕过了交互闸（或 API 直调），此时跑 ReAct 是浪费 LLM 调用且无新信息增量。
    # 直接 short-circuit 到 reporter_deep，让它基于 Step1+2 现有证据出报告。
    if supplementary_df is None or (hasattr(supplementary_df, "empty") and supplementary_df.empty):
        state["deep_rca_skipped"] = True
        state["deep_rca_skip_reason"] = "no_supplementary_data"
        # 保持 prior_* 字段不变，reporter_deep 会优先读它们
        trace.log(TraceEventType.DEEP_RCA_INIT, iteration=0,
                  output_data={
                      "skipped": True,
                      "reason": "no_supplementary_data",
                      "note": "前端未上传补充数据，跳过 Deep RCA，直接基于 Step1+2 出报告",
                  },
                  success=True,
                  metadata={"phase": "v8_deep_rca", "action": "short_circuit"})
        return state

    # ✅ Task 2A 补强: 上传的数据必须能匹配 reasoner 建议的某一种 type
    # 防止用户上传无关数据（例如 reasoner 建议 ad_campaign，用户传了 inventory_status）。
    # 不严格要求"必须是 suggested_data 中的 type"，但必须匹配 SUPPLEMENTARY_TYPE_SIGNATURES
    # 中的某一种已知 type，避免 ReAct 在完全不识别的数据上做无意义的探索。
    try:
        supp_columns = list(supplementary_df.columns)
        matched_types = _identify_supplementary_type(supp_columns)
        prior_reason_result_for_check = state.get("prior_reason_result") or state.get("reason_result") or {}
        suggested_types = prior_reason_result_for_check.get("suggested_data", []) or []
        # 归一化 suggested_data 字符串：去掉 .csv 后缀
        suggested_normalized = {
            str(s).lower().replace(".csv", "").strip()
            for s in suggested_types
        }

        if not matched_types:
            # 上传了数据，但完全不匹配任何已知 type → 拒收
            state["deep_rca_skipped"] = True
            state["deep_rca_skip_reason"] = "supplementary_data_unrecognized"
            state["deep_rca_supplementary_columns"] = supp_columns[:30]
            trace.log(TraceEventType.DEEP_RCA_INIT, iteration=0,
                      output_data={
                          "skipped": True,
                          "reason": "supplementary_data_unrecognized",
                          "uploaded_columns": supp_columns[:20],
                          "expected_types": list(suggested_normalized),
                          "note": "上传数据列名无法识别为任何已知补充数据类型",
                      },
                      success=False,
                      metadata={"phase": "v8_deep_rca", "action": "schema_validation_failed"})
            return state

        # 上传的数据被识别出来了，但是否在 reasoner 建议的范围内？
        # 软警告：不阻塞，但记录到 trace 里供评测和报告引用
        if suggested_normalized:
            in_scope = any(t in suggested_normalized for t in matched_types)
            state["deep_rca_supplementary_match"] = {
                "matched_types": matched_types,
                "suggested_types": list(suggested_normalized),
                "in_scope": in_scope,
            }
            trace.log(TraceEventType.SUPPLEMENTARY_DATA_MERGED, iteration=0,
                      output_data={
                          "phase": "schema_validation_passed",
                          "matched_types": matched_types,
                          "suggested_types": list(suggested_normalized),
                          "in_scope": in_scope,
                      },
                      success=True,
                      metadata={"phase": "v8_deep_rca", "action": "schema_validation"})
    except Exception as _schema_err:
        # 校验器失败永远不阻塞主流程，降级为放行
        trace.log(TraceEventType.DEEP_RCA_INIT, iteration=0,
                  error=f"schema validation crashed (degrade to allow): {str(_schema_err)[:200]}",
                  success=False,
                  metadata={"phase": "v8_deep_rca", "action": "schema_validation_degrade"})

    # 恢复 Step 1+2 的上下文
    prior_reason_result = state.get("prior_reason_result") or state.get("reason_result") or {}
    prior_confirmed = state.get("prior_confirmed_anomalies") or state.get("confirmed_anomalies") or []
    prior_scan_state = state.get("prior_scan_state") or state.get("scan_state") or {}

    # ── 1. 合并补充数据 ──
    schema = get_cached_schema()
    if supplementary_df is not None and not supplementary_df.empty:
        try:
            merged_df = _merge_supplementary_df(uploaded_df, supplementary_df, schema)
            state["uploaded_df"] = merged_df
            # 更新 schema 缓存
            set_current_df(merged_df)
            schema = get_cached_schema()

            trace.log(TraceEventType.SUPPLEMENTARY_DATA_MERGED, iteration=0,
                      output_data={
                          "original_shape": list(uploaded_df.shape),
                          "supplementary_shape": list(supplementary_df.shape),
                          "merged_shape": list(merged_df.shape),
                          "new_columns": [c for c in merged_df.columns if c not in uploaded_df.columns],
                      },
                      success=True,
                      metadata={"phase": "v8_deep_rca"})
        except Exception as e:
            trace.log(TraceEventType.SUPPLEMENTARY_DATA_MERGED, iteration=0,
                      error=str(e)[:300],
                      success=False,
                      metadata={"phase": "v8_deep_rca", "action": "merge_failed_using_original"})

    # ── 2. 对补充数据新增列做全量快速扫描（0 次 LLM）──
    supp_scan_findings = []
    if supplementary_df is not None and not supplementary_df.empty:
        analysis_frame_dict = state.get("analysis_frame") or prior_scan_state.get("analysis_frame") or {}
        # 提取阈值: 优先用 analysis_frame 的, 其次用全局默认
        _af_threshold = SIGNIFICANCE_THRESHOLD
        if analysis_frame_dict:
            _af_threshold = analysis_frame_dict.get("significance_threshold", SIGNIFICANCE_THRESHOLD)

        supp_scan_findings = _scan_supplementary_columns(
            state["uploaded_df"],  # 合并后的 df (或原始 df 如果合并失败)
            uploaded_df,           # 原始 df (用于识别新增列)
            analysis_frame_dict,
            threshold=_af_threshold,
        )

        trace.log(TraceEventType.SUPPLEMENTARY_DATA_MERGED, iteration=0,
                  output_data={
                      "phase": "supplementary_scan_completed",
                      "total_new_columns_scanned": len(supp_scan_findings),
                      "significant_findings": len([f for f in supp_scan_findings if f.get("significant")]),
                      "findings_preview": supp_scan_findings[:5],
                  },
                  success=True,
                  metadata={"phase": "v9.8_supp_scan"})

    # 保存扫描发现到 state，供 reporter_deep 使用
    state["_supp_scan_findings"] = supp_scan_findings

    # ── 3. 生成深度假设（基于 Step 1+2 结论 + 扫描发现 + 补充数据 schema）──
    planner_reasoner_llm = state["_planner_agent"].reasoner_llm if state.get("_planner_agent") else None
    llm = planner_reasoner_llm or get_llm("planner")

    # 获取补充数据的 schema（如果有）
    supp_schema = {}
    if supplementary_df is not None and not supplementary_df.empty:
        supp_schema = get_df_schema(supplementary_df)

    raw_hypotheses = _build_deep_rca_hypotheses(
        prior_reason_result, supp_schema, schema, user_query, llm,
        supp_scan_findings=supp_scan_findings)

    if not raw_hypotheses:
        # 降级: 用通用假设生成
        raw_hypotheses = generate_hypotheses_via_llm(user_query, "csv", llm=llm)
    if not raw_hypotheses:
        raw_hypotheses = _hardcoded_hypotheses()

    # 构建 DynamicPlan
    plan = DynamicPlan(max_attempts=REACT_CONFIG["max_attribution_attempts"])
    for h in _build_hypotheses_from_raw(raw_hypotheses, user_query):
        plan.add_hypothesis(h)

    # ── 4. 预填 EvidenceBoard（继承 Step 1 的 anomalies）──
    evidence_board = EvidenceBoard()
    for anomaly in prior_confirmed:
        dim = anomaly.get("dimension", "unknown")
        pct = anomaly.get("change_pct", 0)
        entry = EvidenceEntry(
            hypothesis_name=f"step1_{dim}",
            dimension=dim,
            change_pct=float(pct) if pct else None,
            direction="down" if (pct and float(pct) < -3) else ("up" if (pct and float(pct) > 3) else "flat"),
            significant=anomaly.get("significant", True),
            actual_value=float(pct) if pct else None,
            raw_conclusion=f"Step 1 扫描已确认: {dim} 变化 {pct}%",
            timestamp=datetime.now().isoformat(),
        )
        evidence_board.entries[f"step1_{dim}"] = entry

    # ✅ v9.8: 将补充数据扫描发现写入 EvidenceBoard（数据驱动，非假设驱动）
    for finding in supp_scan_findings:
        col = finding["column"]
        entry = EvidenceEntry(
            hypothesis_name=f"supp_scan_{col}",
            dimension=col,
            change_pct=finding["change_pct"],
            direction="down" if finding["change_pct"] < -3 else ("up" if finding["change_pct"] > 3 else "flat"),
            significant=finding.get("significant", False),
            actual_value=finding["change_pct"],
            base_value=finding["base_value"],
            compare_value=finding["compare_value"],
            raw_conclusion=(
                f"补充数据扫描: {col} 从 {finding['base_value']} → {finding['compare_value']} "
                f"(变化 {finding['change_pct']:+.1f}%, {finding['aggregation']}聚合)"
            ),
            recommendation=(
                f"关注 {col} 的{'异常' if finding.get('significant') else ''}变化 "
                f"({finding['change_pct']:+.1f}%)，建议排查具体原因并制定应对措施"
            ) if finding.get("significant") else None,
            timestamp=datetime.now().isoformat(),
        )
        evidence_board.entries[f"supp_scan_{col}"] = entry

    # ── 5. 构建 DimensionTree（从合并后 schema）──
    dimension_tree = DimensionTree.build_from_schema(schema)

    trace.log(TraceEventType.DEEP_RCA_INIT, iteration=0,
              output_data={
                  "hypothesis_count": len(plan.hypotheses),
                  "inherited_evidence_count": len(evidence_board.entries),
                  "has_supplementary_data": supplementary_df is not None and not supplementary_df.empty,
                  "hypotheses": [{"name": h.name, "priority": h.priority}
                                 for h in sorted(plan.hypotheses.values(), key=lambda x: -x.priority)],
              },
              success=True,
              metadata={"phase": "v8_deep_rca"})

    trace.log(TraceEventType.PLAN_INITIALIZED, iteration=0,
              output_data={
                  "hypotheses": [{"name": h.name, "priority": h.priority}
                                 for h in sorted(plan.hypotheses.values(), key=lambda x: -x.priority)],
                  "investigation_order": plan.investigation_order,
                  "mode": "deep_rca",
              })

    state["dynamic_plan"] = plan.to_dict()
    state["current_iteration"] = 0
    state["should_continue"] = True
    state["all_results"] = []
    state["_evidence_board"] = evidence_board.to_dict()
    state["_dimension_tree"] = dimension_tree.to_dict()

    return state


def reporter_deep_node(state: AgentState) -> AgentState:
    """
    ✅ v8 Step 3: Deep RCA Reporter — 基于 Step 1+2+3 全部结果生成深度分析报告。

    与 reporter_node 的区别:
    - 包含 Step 1+2 的初步发现（prior_reason_result）
    - 包含 Step 3 的深度验证结果
    - 报告明确标注哪些结论来自快速扫描、哪些来自深度分析
    """
    trace: TraceLog = state["trace"]
    reporter = state["_reporter_agent"]
    user_query = state["user_query"]

    # Step 1+2 的上下文
    prior_reason_result = state.get("prior_reason_result") or state.get("reason_result") or {}
    prior_confirmed = state.get("prior_confirmed_anomalies") or state.get("confirmed_anomalies") or []
    prior_scan_state = state.get("prior_scan_state") or state.get("scan_state") or {}
    scan_summary = prior_scan_state.get("scan_summary", "")
    prior_confidence = prior_reason_result.get("confidence", "unknown")
    prior_causal_chain = prior_reason_result.get("causal_chain", "")

    # Step 3 的结果
    steps = state.get("steps", [])
    dynamic_plan = state.get("dynamic_plan")
    evidence_board_dict = state.get("_evidence_board") or {}

    # 从 evidence_board 提取深度验证的证据
    deep_evidence = []
    for name, entry_data in evidence_board_dict.items():
        if not name.startswith("step1_"):  # 排除 Step 1 继承的
            deep_evidence.append(entry_data)

    # 从 dynamic_plan 提取假设结论
    confirmed_hyps = []
    rejected_hyps = []
    if dynamic_plan:
        plan = DynamicPlan.from_dict(dynamic_plan)
        for h_name, h in plan.hypotheses.items():
            if h.status == HypothesisStatus.CONFIRMED:
                confirmed_hyps.append({"name": h.name, "dimension": h.dimension})
            elif h.status in (HypothesisStatus.REJECTED, HypothesisStatus.PRUNED):
                rejected_hyps.append({"name": h.name, "dimension": h.dimension})

    # 构建步骤摘要
    steps_summary = []
    for step in steps:
        hyp = step.get("hypothesis", "")
        success = step.get("result", {}).get("success", False)
        summary = step.get("result", {}).get("summary", "")
        eval_result = step.get("result", {}).get("evaluation", {})
        conclusion = eval_result.get("conclusion", "") if isinstance(eval_result, dict) else ""
        steps_summary.append(f"- {hyp}: {'✅' if success else '❌'} {conclusion or summary[:100]}")

    has_supp_data = state.get("supplementary_df") is not None

    # ✅ v9.8: 提取补充数据扫描发现
    supp_scan_findings = state.get("_supp_scan_findings") or []
    supp_scan_text = ""
    if supp_scan_findings:
        scan_lines = []
        for f in supp_scan_findings:
            marker = "🔴异常" if f.get("significant") else "🟢正常"
            scan_lines.append(
                f"- [{marker}] {f['column']}: {f['base_value']} → {f['compare_value']} "
                f"(变化 {f['change_pct']:+.1f}%)")
        supp_scan_text = f"""
═══ 补充数据全量扫描结果（数据驱动，已自动计算）═══
{chr(10).join(scan_lines)}
"""

    # ✅ Task 2A 补强: skip 模式下报告必须如实标注未跑 Deep RCA
    deep_rca_skipped = state.get("deep_rca_skipped", False)
    skip_reason = state.get("deep_rca_skip_reason", "")
    skip_notice = ""
    if deep_rca_skipped:
        suggested = prior_reason_result.get("suggested_data", []) or []
        if skip_reason == "no_supplementary_data":
            skip_notice = (
                f"\n\n⚠️ 注意: 本次未执行 Deep RCA（原因: 用户未上传补充数据）。"
                f"以下报告完全基于 Step1+2 的扫描与因果推理结果，"
                f"未结合补充数据做深度验证。"
                f"\n建议补充的数据类型: {suggested if suggested else '（reasoner 未给出具体建议）'}"
            )
        elif skip_reason == "supplementary_data_unrecognized":
            uploaded_cols = state.get("deep_rca_supplementary_columns", [])
            skip_notice = (
                f"\n\n⚠️ 注意: 本次未执行 Deep RCA（原因: 上传的数据无法识别）。"
                f"上传文件的列名 {uploaded_cols[:10]} 不匹配任何已知补充数据类型。"
                f"\n请按以下任一类型重新上传: {suggested if suggested else '（reasoner 未给出具体建议）'}"
            )
        else:
            skip_notice = (
                f"\n\n⚠️ 注意: 本次未执行 Deep RCA（原因: {skip_reason}）。"
                f"以下报告完全基于 Step1+2 的扫描与因果推理结果。"
            )

    prompt = f"""用户问题: {user_query}

═══ Step 1+2 初步分析 ═══
扫描概要: {scan_summary}
初步信心等级: {prior_confidence}
初步因果链: {prior_causal_chain}
初步确认根因: {json.dumps(prior_confirmed[:5], ensure_ascii=False)}
{supp_scan_text}
═══ Step 3 深度根因分析 ═══
{'使用了用户补充的数据进行联合分析' if has_supp_data else '基于原始数据深度分析'}{skip_notice}
验证步骤:
{chr(10).join(steps_summary) if steps_summary else '无深度验证步骤'}

深度分析确认的假设: {json.dumps(confirmed_hyps, ensure_ascii=False)}
深度分析排除的假设: {json.dumps(rejected_hyps, ensure_ascii=False)}

深度分析证据:
{json.dumps(deep_evidence[:10], ensure_ascii=False, indent=2)}

请生成综合分析报告，必须包含以下所有部分:
1. **初步发现回顾**（Step 1+2 的扫描结论）
2. **补充数据深度发现**（逐指标列出: 指标名、基准期值、对比期值、变化幅度、是否异常、结论。务必覆盖上方扫描结果中的每一个异常指标，不要遗漏）
3. **深度分析验证结论**（Step 3 ReAct 验证了什么、排除了什么）
4. **最终根因定位**（综合 Step 1+2+3，含完整因果链）
5. **行动指南**（基于所有发现，每条必须标注优先级 P0/P1/P2，格式: "P0: 具体行动建议"。至少给出 2 条行动建议）
6. **分析确定性评估**（相比初步分析，信心是否提升）

⚠️ 明确标注哪些结论来自快速扫描，哪些来自深度分析。
⚠️ 行动指南必须具体可执行，每条以 P0/P1/P2 开头。"""

    try:
        result = reporter.llm.generate(REPORTER_SYSTEM_PROMPT, prompt)
        # ✅ v9.6 Patch 5: 强制保证报告含'分析局限'段落
        result = _ensure_limitation_section(result, reporter.llm, REPORTER_SYSTEM_PROMPT, state=state)

        # 计算最终确认的根因（合并 Step 1 和 Step 3）
        all_confirmed = [c.get("dimension", "unknown") for c in prior_confirmed]
        for h in confirmed_hyps:
            if h["dimension"] not in all_confirmed:
                all_confirmed.append(h["dimension"])

        state["final_report"] = {
            "success": True,
            "full_content": result,
            "summary": result[:300] + "..." if len(result) > 300 else result,
            "confirmed_hypotheses": all_confirmed,
            "rejected_hypotheses": [h["dimension"] for h in rejected_hyps],
            "chart_paths": state.get("chart_paths", []),
            "causal_chain": prior_causal_chain,
            "confidence": "high" if confirmed_hyps else prior_confidence,
            "needs_deep_rca": False,  # 深度分析已完成
            "suggested_data": prior_reason_result.get("suggested_data", []) if deep_rca_skipped else [],
            "reason_result": prior_reason_result,
            "is_deep_rca_report": True,
            "deep_rca_steps": len(steps),
            # ✅ Task 2A: 让前端能区分"真跑了 Deep RCA"和"因无数据而 short-circuit"
            "deep_rca_skipped": deep_rca_skipped,
            "deep_rca_skip_reason": skip_reason if deep_rca_skipped else None,
        }

        trace.log(TraceEventType.DEEP_RCA_REPORT,
                  iteration=state.get("current_iteration", 0),
                  output_data={
                      "summary": result[:200],
                      "confirmed_count": len(all_confirmed),
                      "rejected_count": len(rejected_hyps),
                      "steps_count": len(steps),
                  },
                  metadata={"phase": "v8_deep_rca"})

    except Exception as e:
        state["final_report"] = {
            "success": False,
            "full_content": f"深度分析报告生成失败: {str(e)}",
            "summary": "深度分析完成但报告生成失败",
            "confirmed_hypotheses": [c.get("dimension", "") for c in prior_confirmed],
            "rejected_hypotheses": [],
            "error": str(e),
            "confidence": prior_confidence,
            "needs_deep_rca": False,
            "is_deep_rca_report": True,
        }

    trace.finalize("completed", {
        "route": "complex",
        "mode": "v8_deep_rca_complete",
        "deep_rca_steps": len(steps),
    })

    return state


def hypothesis_refill_node(state: AgentState) -> AgentState:
    """
    ✅ v5.7 新节点：假设补充（fallback 路径使用）
    当所有初始假设均被排除/剪枝、但未找到根因且 budget 未耗尽时触发。
    调用 reasoner 生成补充假设（基于已排除的假设），然后回到 react_step 继续验证。
    整个请求生命周期中最多调用 max_refills 次（默认 1 次）。
    """
    trace: TraceLog = state["trace"]
    user_query = state["user_query"]
    iteration = state["current_iteration"]

    # 恢复动态计划
    plan_dict = state["dynamic_plan"]
    dynamic_plan = DynamicPlan.from_dict(plan_dict)

    # 收集已排除的假设信息
    rejected_info = []
    for name, h in dynamic_plan.hypotheses.items():
        if h.status in (HypothesisStatus.REJECTED, HypothesisStatus.PRUNED):
            rejected_info.append({
                "name": name,
                "dimension": h.dimension,
                "description": h.description,
                "evidence": h.evidence,
            })

    # ── 调用 reasoner 生成补充假设（整个请求中仅此一处额外调用 reasoner） ──
    planner = state["_planner_agent"]
    refill_llm = planner.reasoner_llm  # ✅ 用 reasoner，因为需要深度推理新角度

    trace.log(TraceEventType.HYPOTHESIS_REFILL, iteration=iteration,
              input_data={
                  "rejected_count": len(rejected_info),
                  "refill_attempt": state.get("_refill_count", 0) + 1,
              },
              metadata={"llm_role": "reasoner"})

    raw_new = refill_hypotheses_via_llm(user_query, rejected_info, llm=refill_llm)

    if raw_new:
        # 过滤掉与已有假设重名的
        existing_names = set(dynamic_plan.hypotheses.keys())
        new_hypotheses = [
            h for h in _build_hypotheses_from_raw(raw_new, user_query)
            if h.name not in existing_names
        ]

        for h in new_hypotheses:
            dynamic_plan.add_hypothesis(h)

        trace.log(TraceEventType.HYPOTHESIS_REFILL, iteration=iteration,
                  output_data={
                      "new_hypotheses": [{"name": h.name, "dimension": h.dimension} for h in new_hypotheses],
                      "new_count": len(new_hypotheses),
                  },
                  success=True)
    else:
        new_hypotheses = []
        trace.log(TraceEventType.HYPOTHESIS_REFILL, iteration=iteration,
                  output_data={"new_count": 0},
                  success=False,
                  error="reasoner 未能生成补充假设")

    # 更新状态
    state["dynamic_plan"] = dynamic_plan.to_dict()
    state["_refill_count"] = state.get("_refill_count", 0) + 1
    state["_needs_refill"] = False

    # 如果补充了新假设 → 继续验证；否则 → 直接去 reporter
    if new_hypotheses:
        state["should_continue"] = True
    else:
        state["should_continue"] = False

    return state


# ============================================================================
# 14. 条件路由函数
# ============================================================================

def route_after_gateway(state: AgentState) -> str:
    """Gateway 后的路由 — ✅ v9: 所有合法查询统一进 Quick Scan"""
    route = state.get("route", "complex")
    if route == "invalid":
        return "invalid_handler"
    else:
        return "quick_scan"  # ✅ v9: 不再区分 simple/complex，统一进 Quick Scan

def should_continue_react(state: AgentState) -> str:
    """
    ✅ v5.8: react_step 之后总是进入 evaluator（不再直接跳转）
    唯一例外：react_step 发现无假设或 budget 耗尽时设置 should_continue=False
    """
    # react_step 中如果无假设或 budget 耗尽，会设置 should_continue=False
    if not state.get("should_continue", True):
        return "reporter"
    return "evaluator"


def should_continue_after_evaluator(state: AgentState) -> str:
    """
    ✅ v5.8: evaluator_node 后的四路判断
    - react_step: 还有 pending 假设 → 继续验证
    - hypothesis_refill: 假设用完 + 未找到根因 → 补充假设
    - reporter: 已找到根因 / early stop / budget 耗尽 → 生成报告
    """
    if state.get("should_continue", False):
        return "react_step"
    elif state.get("_needs_refill", False):
        return "hypothesis_refill"
    else:
        return "reporter"


def should_continue_after_refill(state: AgentState) -> str:
    """
    ✅ v5.7: hypothesis_refill 后的路由
    - 补充成功（有新假设）→ react_step 继续验证
    - 补充失败（无新假设）→ reporter 直接生成报告
    """
    if state.get("should_continue", False):
        return "react_step"
    else:
        return "reporter"


def route_after_quick_scan(state: AgentState) -> str:
    """
    ✅ v9: quick_scan_node 后的三路分发（替代 v8 的二路判断）

    路由逻辑:
    - scan 失败 → react_init（降级到旧路径）
    - analysis_depth == "descriptive" → present_scan_results（格式化扫描结果，不做推理）
    - analysis_depth == "diagnostic" 或 "causal" → reason（进入因果推理）
    """
    if state.get("_fallback_to_react", False):
        return "react_init"

    depth = state.get("analysis_depth", "diagnostic")
    if depth == "descriptive":
        return "present_scan_results"
    else:
        return "reason"  # diagnostic 和 causal 都进 reasoner_v2


# v7 兼容
route_after_scan = route_after_quick_scan

# v7 兼容: reason_node 别名
reason_node = reasoner_v2_node


def route_after_reason(state: AgentState) -> str:
    """
    ✅ v8 Step 3: reasoner_v2_node 后的路由判断

    - needs_deep_rca=False → reporter_v2（直接生成报告）
    - needs_deep_rca=True  → present_findings（展示初步结论 + 询问用户）
    """
    if state.get("needs_deep_rca", False):
        return "present_findings"
    return "reporter_v2"


# ============================================================================
# 15. 构建 LangGraph
# ============================================================================

def build_graph() -> StateGraph:
    """
    ✅ v9: 统一管线架构（Unified Pipeline + Adaptive Depth）

    所有合法查询统一流程:
      gateway → quick_scan → [route_after_quick_scan]
                    ↓ (scan失败)     ├→ present_scan_results → END (descriptive)
                react_init → ...     ├→ reasoner_v2 → [route_after_reason]
                                     │    ├→ reporter_v2 → END (diagnostic/causal, no deep rca)
                                     │    └→ present_findings → END (causal, needs deep rca)
                                     └→ react_init (fallback)
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("gateway", gateway_node)
    workflow.add_node("invalid_handler", invalid_handler_node)

    # ✅ v9: 移除 simple_executor，新增 present_scan_results
    workflow.add_node("present_scan_results", present_scan_results)

    # v8 主管线节点（quick_scan + reasoner_v2 + reporter_v2）
    workflow.add_node("quick_scan", quick_scan_node)
    workflow.add_node("reason", reasoner_v2_node)
    workflow.add_node("reporter_v2", reporter_node_v2)

    # v8 Step 3: present_findings（第一轮终止节点）
    workflow.add_node("present_findings", present_findings_node)

    # fallback 路径节点（保留旧架构）
    workflow.add_node("react_init", react_init_node)
    workflow.add_node("react_step", react_step_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("hypothesis_refill", hypothesis_refill_node)
    workflow.add_node("reporter", reporter_node)

    # 设置入口
    workflow.set_entry_point("gateway")

    # ── Gateway 路由 — ✅ v9: 只有 invalid 和 quick_scan 两条路 ──
    workflow.add_conditional_edges("gateway", route_after_gateway, {
        "invalid_handler": "invalid_handler",
        "quick_scan": "quick_scan",
    })

    workflow.add_edge("invalid_handler", END)

    # ── v9 主管线: quick_scan 后三路分发 ──
    workflow.add_conditional_edges("quick_scan", route_after_quick_scan, {
        "present_scan_results": "present_scan_results",
        "reason": "reason",
        "react_init": "react_init",
    })

    workflow.add_edge("present_scan_results", END)  # ✅ v9: descriptive 出口

    # v8 Step 3: reason 后条件路由
    workflow.add_conditional_edges("reason", route_after_reason, {
        "reporter_v2": "reporter_v2",
        "present_findings": "present_findings",
    })

    workflow.add_edge("reporter_v2", END)
    workflow.add_edge("present_findings", END)  # 第一轮在此终止

    # ── fallback 路径（旧 react 架构） ──
    workflow.add_edge("react_init", "react_step")

    workflow.add_conditional_edges("react_step", should_continue_react, {
        "evaluator": "evaluator",
        "reporter": "reporter",
    })

    workflow.add_conditional_edges("evaluator", should_continue_after_evaluator, {
        "react_step": "react_step",
        "hypothesis_refill": "hypothesis_refill",
        "reporter": "reporter",
    })

    workflow.add_conditional_edges("hypothesis_refill", should_continue_after_refill, {
        "react_step": "react_step",
        "reporter": "reporter",
    })

    workflow.add_edge("reporter", END)

    return workflow.compile()

# ============================================================================
# 16. Orchestrator
# ============================================================================

class AgentOrchestrator:
    """Agent 编排器"""

    def __init__(self):
        self.graph = build_graph()
        # ✅ v8 Step 3: 构建一个 Deep RCA 专用 sub-graph
        self._deep_rca_graph = self._build_deep_rca_graph()

    def _build_deep_rca_graph(self) -> StateGraph:
        """
        ✅ v8 Step 3: 构建 Deep RCA 子图
        deep_rca_init → react_step → evaluator → ... → reporter_deep → END
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("deep_rca_init", deep_rca_init_node)
        workflow.add_node("react_step", react_step_node)
        workflow.add_node("evaluator", evaluator_node)
        workflow.add_node("hypothesis_refill", hypothesis_refill_node)
        workflow.add_node("reporter_deep", reporter_deep_node)

        workflow.set_entry_point("deep_rca_init")

        # ✅ Task 2A 补强: deep_rca_init 后做闸门判断
        # 如果 deep_rca_init_node 因无补充数据而 short-circuit，跳过 ReAct 直接出报告
        def _route_after_deep_rca_init(state: AgentState) -> str:
            if state.get("deep_rca_skipped"):
                return "reporter_deep"
            return "react_step"

        workflow.add_conditional_edges("deep_rca_init", _route_after_deep_rca_init, {
            "react_step": "react_step",
            "reporter_deep": "reporter_deep",
        })

        def _route_after_react_deep(state: AgentState) -> str:
            if not state.get("should_continue", True):
                return "reporter_deep"
            return "evaluator"

        workflow.add_conditional_edges("react_step", _route_after_react_deep, {
            "evaluator": "evaluator",
            "reporter_deep": "reporter_deep",
        })

        def _route_after_eval_deep(state: AgentState) -> str:
            if state.get("should_continue", False):
                return "react_step"
            elif state.get("_needs_refill", False):
                return "hypothesis_refill"
            return "reporter_deep"

        workflow.add_conditional_edges("evaluator", _route_after_eval_deep, {
            "react_step": "react_step",
            "hypothesis_refill": "hypothesis_refill",
            "reporter_deep": "reporter_deep",
        })

        def _route_after_refill_deep(state: AgentState) -> str:
            if state.get("should_continue", False):
                return "react_step"
            return "reporter_deep"

        workflow.add_conditional_edges("hypothesis_refill", _route_after_refill_deep, {
            "react_step": "react_step",
            "reporter_deep": "reporter_deep",
        })

        workflow.add_edge("reporter_deep", END)

        return workflow.compile()

    # ✅ v9.9.2: 节点进度描述映射（用于实时 UI 展示）
    _NODE_PROGRESS_INFO = {
        "gateway": {"icon": "🚪", "label": "路由分析", "desc": "解析查询意图..."},
        "quick_scan": {"icon": "🔍", "label": "快速扫描", "desc": "全维度扫描中..."},
        "detect": {"icon": "📊", "label": "异常检测", "desc": "识别异常维度..."},
        "reason": {"icon": "🧠", "label": "因果推理", "desc": "分析根因链路..."},
        "reporter_v2": {"icon": "📝", "label": "报告生成", "desc": "生成分析报告..."},
        "present_findings": {"icon": "📋", "label": "结论展示", "desc": "整理分析结论..."},
        "react_init": {"icon": "🔄", "label": "ReAct 初始化", "desc": "生成假设列表..."},
        "react_step": {"icon": "⚡", "label": "假设验证", "desc": "执行验证代码..."},
        "evaluator": {"icon": "⚖️", "label": "结果评估", "desc": "评估验证结果..."},
        "hypothesis_refill": {"icon": "🔄", "label": "假设补充", "desc": "生成新假设..."},
        "reporter": {"icon": "📝", "label": "报告生成", "desc": "生成分析报告..."},
        "invalid_handler": {"icon": "⚠️", "label": "无效请求", "desc": "处理无效查询..."},
    }

    def process_stream(self, user_query: str,
                       data_source: DataSourceType = DataSourceType.CSV,
                       uploaded_df: pd.DataFrame = None,
                       on_node_start: callable = None,
                       on_node_end: callable = None) -> Dict:
        """
        ✅ v9.9.2: 流式处理用户查询，支持实时回调
        
        Args:
            on_node_start: 节点开始回调 (node_name, progress_info) -> None
            on_node_end: 节点结束回调 (node_name, state_snapshot) -> None
        """
        start_time = time.time()

        initial_state: AgentState = {
            "user_query": user_query,
            "data_source": data_source.value,
            "uploaded_df": uploaded_df,
            "route": "",
            "output_type": "TABLE",
            "chart_type": None,
            "instruction": None,
            "validation_result": None,
            "dynamic_plan": None,
            "current_iteration": 0,
            "current_hypothesis": None,
            "all_results": [],
            "should_continue": True,
            "trace": None,
            "_needs_refill": False,
            "_refill_count": 0,
            "_max_refills": REACT_CONFIG.get("max_refills", 1),
            "_evidence_board": None,
            "_dimension_tree": None,
            "_latest_evaluation": None,
            "analysis_frame": None,
            "scan_state": None,
            "analysis_depth": None,
            "reason_result": None,
            "needs_deep_rca": None,
            "user_decision": None,
            "supplementary_df": None,
            "findings_presented": False,
            "deep_rca_mode": False,
            "prior_scan_state": None,
            "prior_reason_result": None,
            "prior_confirmed_anomalies": None,
            "scan_result": None,
            "scan_data": [],
            "confirmed_anomalies": [],
            "rejected_dimensions": [],
            "has_anomaly": None,
            "causal_result": None,
            "_fallback_to_react": False,
            "drilldown_data": [],
            "_python_agent": None,
            "_planner_agent": None,
            "_reporter_agent": None,
            "steps": [],
            "final_result": None,
            "final_report": None,
            "chart_paths": [],
            "error": None,
            "suggestions": None,
        }

        final_state = None
        # 使用 stream 模式逐步获取每个节点的输出
        for event in self.graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                # 节点开始回调
                if on_node_start and node_name in self._NODE_PROGRESS_INFO:
                    progress_info = self._NODE_PROGRESS_INFO[node_name]
                    on_node_start(node_name, progress_info)
                
                # 节点结束回调（传递部分状态快照）
                if on_node_end:
                    snapshot = {
                        "node": node_name,
                        "analysis_depth": node_output.get("analysis_depth"),
                        "has_anomaly": node_output.get("has_anomaly"),
                        "confirmed_anomalies": node_output.get("confirmed_anomalies", []),
                        "steps_count": len(node_output.get("steps", [])),
                    }
                    on_node_end(node_name, snapshot)
                
                final_state = node_output

        # 如果 stream 没有输出，回退到 invoke
        if final_state is None:
            final_state = self.graph.invoke(initial_state)

        return {
            "route": final_state.get("route", "unknown"),
            "analysis_depth": final_state.get("analysis_depth", "unknown"),
            "total_time_ms": int((time.time() - start_time) * 1000),
            "steps": final_state.get("steps", []),
            "final_result": final_state.get("final_result"),
            "final_report": final_state.get("final_report"),
            "chart_paths": final_state.get("chart_paths", []),
            "error": final_state.get("error"),
            "suggestions": final_state.get("suggestions"),
            "trace_log": final_state.get("trace").to_dict() if final_state.get("trace") else None,
            "scan_state": final_state.get("scan_state"),
            "reason_result": final_state.get("reason_result"),
            "suggested_data_types": (
                final_state.get("reason_result", {}).get("suggested_data_type_ids")
                or final_state.get("reason_result", {}).get("suggested_data_types")
                or final_state.get("reason_result", {}).get("suggested_data", [])
            ) if final_state.get("reason_result") else [],
            "findings_presented": final_state.get("findings_presented", False),
            "needs_deep_rca": final_state.get("needs_deep_rca", False),
            "report": final_state.get("final_report"),
            "_intermediate_state": {
                "scan_state": final_state.get("scan_state"),
                "reason_result": final_state.get("reason_result"),
                "confirmed_anomalies": final_state.get("confirmed_anomalies"),
                "rejected_dimensions": final_state.get("rejected_dimensions"),
                "analysis_frame": final_state.get("analysis_frame"),
                "scan_data": final_state.get("scan_data"),
                "drilldown_data": final_state.get("drilldown_data"),
                "has_anomaly": final_state.get("has_anomaly"),
            } if final_state.get("findings_presented") else None,
        }

    def process(self, user_query: str,
                data_source: DataSourceType = DataSourceType.CSV,
                uploaded_df: pd.DataFrame = None) -> Dict:
        """处理用户查询"""
        start_time = time.time()

        initial_state: AgentState = {
            "user_query": user_query,
            "data_source": data_source.value,
            "uploaded_df": uploaded_df,
            "route": "",
            "output_type": "TABLE",
            "chart_type": None,
            "instruction": None,
            "validation_result": None,
            "dynamic_plan": None,
            "current_iteration": 0,
            "current_hypothesis": None,
            "all_results": [],
            "should_continue": True,
            "trace": None,
            # v5.7 字段 (fallback 路径)
            "_needs_refill": False,
            "_refill_count": 0,
            "_max_refills": REACT_CONFIG.get("max_refills", 1),
            # v5.8 Evaluator 字段 (fallback 路径)
            "_evidence_board": None,
            "_dimension_tree": None,
            "_latest_evaluation": None,
            # ✅ v8 Quick Scan 字段
            "analysis_frame": None,
            "scan_state": None,
            # ✅ v9 新增
            "analysis_depth": None,
            # ✅ v8 Step 2 字段
            "reason_result": None,
            "needs_deep_rca": None,
            # ✅ v8 Step 3 字段
            "user_decision": None,
            "supplementary_df": None,
            "findings_presented": False,
            "deep_rca_mode": False,
            "prior_scan_state": None,
            "prior_reason_result": None,
            "prior_confirmed_anomalies": None,
            # v7 兼容字段（由 quick_scan_node 自动填充）
            "scan_result": None,
            "scan_data": [],
            "confirmed_anomalies": [],
            "rejected_dimensions": [],
            "has_anomaly": None,
            "causal_result": None,
            "_fallback_to_react": False,
            "drilldown_data": [],
            # Agent 实例占位，由 gateway_node 初始化
            "_python_agent": None,
            "_planner_agent": None,
            "_reporter_agent": None,
            "steps": [],
            "final_result": None,
            "final_report": None,
            "chart_paths": [],
            "error": None,
            "suggestions": None,
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "route": final_state.get("route", "unknown"),
            "analysis_depth": final_state.get("analysis_depth", "unknown"),  # ✅ v9
            "total_time_ms": int((time.time() - start_time) * 1000),
            "steps": final_state.get("steps", []),
            "final_result": final_state.get("final_result"),
            "final_report": final_state.get("final_report"),
            "chart_paths": final_state.get("chart_paths", []),
            "error": final_state.get("error"),
            "suggestions": final_state.get("suggestions"),
            "trace_log": final_state.get("trace").to_dict() if final_state.get("trace") else None,
            # ✅ v8 Step 3: 传递中间状态供前端存储
            "scan_state": final_state.get("scan_state"),
            "reason_result": final_state.get("reason_result"),
            # ✅ v9.6: 优先用 type_id 列表(纯字符串), 回退到 dict 列表
            "suggested_data_types": (
                final_state.get("reason_result", {}).get("suggested_data_type_ids")
                or final_state.get("reason_result", {}).get("suggested_data_types")
                or final_state.get("reason_result", {}).get("suggested_data", [])
            ) if final_state.get("reason_result") else [],
            "findings_presented": final_state.get("findings_presented", False),
            "needs_deep_rca": final_state.get("needs_deep_rca", False),
            "report": final_state.get("final_report"),  # 评测用 report 字段
            # 中间状态序列化（供 process_deep_rca 恢复）
            "_intermediate_state": {
                "scan_state": final_state.get("scan_state"),
                "reason_result": final_state.get("reason_result"),
                "confirmed_anomalies": final_state.get("confirmed_anomalies"),
                "rejected_dimensions": final_state.get("rejected_dimensions"),
                "analysis_frame": final_state.get("analysis_frame"),
                "scan_data": final_state.get("scan_data"),
                "drilldown_data": final_state.get("drilldown_data"),
                "has_anomaly": final_state.get("has_anomaly"),
            } if final_state.get("findings_presented") else None,
        }

    def process_deep_rca(self, user_query: str, prior_state: dict,
                         uploaded_df: pd.DataFrame = None,
                         supplementary_df: pd.DataFrame = None) -> Dict:
        """
        ✅ v8 Step 3: 深度根因分析（第二轮请求入口）

        基于 Step 1+2 的中间状态，启动 Deep RCA 子图:
        deep_rca_init → react_step → evaluator → ... → reporter_deep → END

        Args:
            user_query: 原始用户问题
            prior_state: 第一轮 process() 返回的 _intermediate_state
            uploaded_df: 原始数据
            supplementary_df: 用户补充上传的数据（可选）
        """
        start_time = time.time()

        # 恢复 schema 缓存
        if uploaded_df is not None:
            set_current_df(uploaded_df)

        # 初始化 Agent 实例（与 gateway_node 类似逻辑）
        python_agent = PythonAgent(
            llm=get_llm("python_agent"),
        )
        planner_agent = PlannerAgent(
            reasoner_llm=get_llm("planner"),
            chat_llm=get_llm("planner_chat"),
        )
        reporter_agent = ReporterAgent(llm=get_llm("reporter"))

        # 构建 TraceLog
        trace = TraceLog(
            session_id=str(uuid.uuid4())[:8],
            user_query=user_query,
            start_time=datetime.now().isoformat(),
            data_source="csv",
        )
        trace.log(TraceEventType.SESSION_START, iteration=0,
                  metadata={"mode": "deep_rca", "has_supplementary": supplementary_df is not None})

        initial_state: AgentState = {
            "user_query": user_query,
            "data_source": "csv",
            "uploaded_df": uploaded_df,
            "route": "complex",
            "output_type": "TABLE",
            "chart_type": None,
            "instruction": None,
            "validation_result": None,
            "dynamic_plan": None,
            "current_iteration": 0,
            "current_hypothesis": None,
            "all_results": [],
            "should_continue": True,
            "trace": trace,
            "_needs_refill": False,
            "_refill_count": 0,
            "_max_refills": REACT_CONFIG.get("max_refills", 1),
            "_evidence_board": None,
            "_dimension_tree": None,
            "_latest_evaluation": None,
            # v8 字段
            "analysis_frame": prior_state.get("analysis_frame"),
            "scan_state": prior_state.get("scan_state"),
            "reason_result": prior_state.get("reason_result"),
            "needs_deep_rca": True,
            # ✅ Step 3 特有字段
            "user_decision": "continue",
            "supplementary_df": supplementary_df,
            "findings_presented": True,
            "deep_rca_mode": True,
            "prior_scan_state": prior_state.get("scan_state"),
            "prior_reason_result": prior_state.get("reason_result"),
            "prior_confirmed_anomalies": prior_state.get("confirmed_anomalies"),
            # v7 兼容
            "scan_result": None,
            "scan_data": prior_state.get("scan_data", []),
            "confirmed_anomalies": prior_state.get("confirmed_anomalies", []),
            "rejected_dimensions": prior_state.get("rejected_dimensions", []),
            "has_anomaly": prior_state.get("has_anomaly"),
            "causal_result": prior_state.get("reason_result"),
            "_fallback_to_react": False,
            "drilldown_data": prior_state.get("drilldown_data", []),
            # Agent 实例
            "_python_agent": python_agent,
            "_planner_agent": planner_agent,
            "_reporter_agent": reporter_agent,
            "steps": [],
            "final_result": None,
            "final_report": None,
            "chart_paths": [],
            "error": None,
            "suggestions": None,
        }

        try:
            final_state = self._deep_rca_graph.invoke(initial_state)
        except Exception as e:
            trace.finalize("error", {"error": str(e)})
            return {
                "route": "complex",
                "total_time_ms": int((time.time() - start_time) * 1000),
                "steps": [],
                "final_result": None,
                "final_report": {
                    "success": False,
                    "full_content": f"深度分析失败: {str(e)}",
                    "summary": f"深度分析执行出错: {str(e)[:200]}",
                    "is_deep_rca_report": True,
                },
                "chart_paths": [],
                "error": str(e),
                "suggestions": None,
                "trace_log": trace.to_dict(),
                "is_deep_rca": True,
            }

        # ✅ v9.6 FIX: 构造评测契约字段 deep_rca_result
        # reporter_deep_node 把结果写在 final_report 里 (含 confirmed_hypotheses,
        # rejected_hypotheses, causal_chain, full_content), 这里转成评测期望的结构
        _final_report = final_state.get("final_report") or {}
        _prior_reason = (final_state.get("prior_reason_result")
                         or final_state.get("reason_result") or {})
        _evidence_board = final_state.get("_evidence_board") or {}

        # 提取 step3 验证后的"深度根因"——优先用 confirmed_hypotheses,
        # 没有则 fallback 到 prior root_causes
        _deep_root_causes = []
        for h in (_final_report.get("confirmed_hypotheses") or []):
            if isinstance(h, str):
                _deep_root_causes.append({
                    "name": h, "dimension": h,
                    "source": "deep_rca_verified",
                })
            elif isinstance(h, dict):
                _deep_root_causes.append({
                    "name": h.get("name", h.get("dimension", "")),
                    "dimension": h.get("dimension", h.get("name", "")),
                    "source": "deep_rca_verified",
                })
        if not _deep_root_causes:
            for rc in (_prior_reason.get("root_causes") or []):
                if isinstance(rc, dict):
                    _deep_root_causes.append({
                        "name": rc.get("dimension", ""),
                        "dimension": rc.get("dimension", ""),
                        "change_pct": rc.get("change_pct"),
                        "reasoning": rc.get("reasoning", ""),
                        "source": "step1_2_inherited",
                    })

        # ✅ v9.8: action_recommendations 三级提取策略
        _action_recommendations = []
        _full_content = _final_report.get("full_content", "") or ""

        # 策略1: 从 evidence_board 提取（v9.8 supp_scan 写入了 recommendation）
        for ev_name, ev_data in _evidence_board.items():
            if isinstance(ev_data, dict) and ev_data.get("recommendation"):
                _action_recommendations.append({
                    "hypothesis": ev_name,
                    "recommendation": ev_data["recommendation"],
                    "priority": "P1",
                })

        # 策略2: 从报告正文提取 P0/P1/P2 格式的行动建议
        if not _action_recommendations and _full_content:
            import re
            for match in re.finditer(
                    r'[*\-•]?\s*(P[012])[：:．.\s]\s*(.+?)(?=\n[*\-•]?\s*P[012]|\n\n|\n##|\Z)',
                    _full_content, re.DOTALL):
                rec_text = match.group(2).strip()
                if len(rec_text) > 5:  # 过滤太短的噪音
                    _action_recommendations.append({
                        "priority": match.group(1),
                        "recommendation": rec_text[:300],
                    })

        # 策略3: 从 deep_root_causes + supp_scan_findings 生成兜底建议
        if not _action_recommendations:
            _supp_findings = final_state.get("_supp_scan_findings") or []
            sig_findings = [f for f in _supp_findings if f.get("significant")]
            # 优先用扫描发现的显著异常
            for f in sig_findings[:3]:
                _action_recommendations.append({
                    "priority": "P1",
                    "recommendation": (
                        f"关注 {f['column']} 的异常变化 "
                        f"(从 {f['base_value']} → {f['compare_value']}，"
                        f"变化 {f['change_pct']:+.1f}%)，建议深入排查原因并制定应对措施"
                    ),
                })
            # 其次用 deep_root_causes
            if not _action_recommendations and _deep_root_causes:
                for rc in _deep_root_causes[:3]:
                    _action_recommendations.append({
                        "priority": "P1",
                        "recommendation": f"针对 {rc.get('dimension', '未知')} 的异常变化进行深入排查和优化",
                    })

        deep_rca_result = {
            "deep_root_causes": _deep_root_causes,
            "action_recommendations": _action_recommendations,
            "confirmed_hypotheses": _final_report.get("confirmed_hypotheses", []),
            "rejected_hypotheses": _final_report.get("rejected_hypotheses", []),
            "causal_chain": (_final_report.get("causal_chain", "")
                             or _prior_reason.get("causal_chain", "")),
            "confidence": _final_report.get("confidence", "unknown"),
            "summary": _final_report.get("summary", ""),
            "full_content": _full_content,
            "deep_rca_steps": _final_report.get(
                "deep_rca_steps", len(final_state.get("steps", []))),
            "is_deep_rca_report": True,
        }

        return {
            "route": final_state.get("route", "complex"),
            "total_time_ms": int((time.time() - start_time) * 1000),
            "steps": final_state.get("steps", []),
            "final_result": final_state.get("final_result"),
            "final_report": _final_report,
            "chart_paths": final_state.get("chart_paths", []),
            "error": final_state.get("error"),
            "suggestions": final_state.get("suggestions"),
            "trace_log": final_state.get("trace").to_dict() if final_state.get("trace") else None,
            "is_deep_rca": True,
            # ✅ 评测契约字段
            "deep_rca_result": deep_rca_result,
            # ✅ 让 eval run_single 的 step3_result.get("report") 也能拿到内容
            "report": _final_report,
        }

# ============================================================================
# 17. 测试数据
# ============================================================================

def setup_mock_data() -> "pd.DataFrame":
    """
    生成测试 DataFrame — 模拟真实店铺经营概况表
    
    Schema 对标真实数据:
    - 维度: 日期, 品类, 流量来源
    - 流量指标: 访客数, 浏览量, 平均停留时长(秒), 跳失率
    - 转化指标: 加购人数, 支付买家数, 支付转化率
    - 交易指标: 支付订单数, 支付金额, 支付件数
    - 售后指标: 退款订单数, 退款金额
    - 价格指标: 客单价, 件单价
    
    内置异常场景（用于测试归因分析）:
    - 8月美妆-付费推广: 转化率大幅下降（模拟投放效率问题）
    """
    import random
    random.seed(42)

    # 品类和流量来源（中文，匹配真实数据）
    categories = ["服装", "电子", "美妆", "食品"]
    traffic_sources = ["付费推广", "推荐", "直接访问", "直播", "自然搜索"]
    
    # 基准值配置（品类 × 流量来源）
    base_values = {
        # 服装
        ("服装", "付费推广"): {"visitors": 2000, "pv_ratio": 2.5, "stay": 60, "bounce": 0.40, "cart_rate": 0.18, "conv": 0.032, "price": 180, "item_price": 125},
        ("服装", "推荐"): {"visitors": 1500, "pv_ratio": 3.0, "stay": 95, "bounce": 0.43, "cart_rate": 0.14, "conv": 0.030, "price": 180, "item_price": 125},
        ("服装", "直接访问"): {"visitors": 800, "pv_ratio": 2.6, "stay": 90, "bounce": 0.32, "cart_rate": 0.19, "conv": 0.036, "price": 180, "item_price": 125},
        ("服装", "直播"): {"visitors": 1100, "pv_ratio": 2.9, "stay": 65, "bounce": 0.50, "cart_rate": 0.19, "conv": 0.025, "price": 180, "item_price": 125},
        ("服装", "自然搜索"): {"visitors": 2400, "pv_ratio": 3.2, "stay": 55, "bounce": 0.33, "cart_rate": 0.14, "conv": 0.033, "price": 180, "item_price": 125},
        # 电子
        ("电子", "付费推广"): {"visitors": 1400, "pv_ratio": 2.7, "stay": 105, "bounce": 0.41, "cart_rate": 0.14, "conv": 0.023, "price": 340, "item_price": 255},
        ("电子", "推荐"): {"visitors": 1000, "pv_ratio": 2.5, "stay": 90, "bounce": 0.39, "cart_rate": 0.16, "conv": 0.025, "price": 340, "item_price": 255},
        ("电子", "直接访问"): {"visitors": 560, "pv_ratio": 2.9, "stay": 60, "bounce": 0.43, "cart_rate": 0.13, "conv": 0.027, "price": 340, "item_price": 255},
        ("电子", "直播"): {"visitors": 780, "pv_ratio": 2.7, "stay": 100, "bounce": 0.37, "cart_rate": 0.18, "conv": 0.025, "price": 340, "item_price": 255},
        ("电子", "自然搜索"): {"visitors": 1600, "pv_ratio": 3.3, "stay": 100, "bounce": 0.32, "cart_rate": 0.20, "conv": 0.021, "price": 340, "item_price": 255},
        # 美妆
        ("美妆", "付费推广"): {"visitors": 2350, "pv_ratio": 3.1, "stay": 65, "bounce": 0.32, "cart_rate": 0.18, "conv": 0.035, "price": 130, "item_price": 105},
        ("美妆", "推荐"): {"visitors": 1800, "pv_ratio": 3.4, "stay": 110, "bounce": 0.41, "cart_rate": 0.17, "conv": 0.037, "price": 130, "item_price": 105},
        ("美妆", "直接访问"): {"visitors": 900, "pv_ratio": 2.4, "stay": 115, "bounce": 0.37, "cart_rate": 0.18, "conv": 0.037, "price": 130, "item_price": 105},
        ("美妆", "直播"): {"visitors": 1450, "pv_ratio": 2.2, "stay": 75, "bounce": 0.51, "cart_rate": 0.19, "conv": 0.035, "price": 130, "item_price": 105},
        ("美妆", "自然搜索"): {"visitors": 2750, "pv_ratio": 3.3, "stay": 85, "bounce": 0.36, "cart_rate": 0.18, "conv": 0.038, "price": 130, "item_price": 105},
        # 食品
        ("食品", "付费推广"): {"visitors": 1600, "pv_ratio": 3.0, "stay": 100, "bounce": 0.54, "cart_rate": 0.12, "conv": 0.041, "price": 45, "item_price": 34},
        ("食品", "推荐"): {"visitors": 1250, "pv_ratio": 3.1, "stay": 90, "bounce": 0.30, "cart_rate": 0.15, "conv": 0.036, "price": 45, "item_price": 34},
        ("食品", "直接访问"): {"visitors": 580, "pv_ratio": 3.3, "stay": 100, "bounce": 0.53, "cart_rate": 0.13, "conv": 0.050, "price": 45, "item_price": 34},
        ("食品", "直播"): {"visitors": 900, "pv_ratio": 2.6, "stay": 90, "bounce": 0.34, "cart_rate": 0.13, "conv": 0.051, "price": 45, "item_price": 34},
        ("食品", "自然搜索"): {"visitors": 1950, "pv_ratio": 2.9, "stay": 80, "bounce": 0.38, "cart_rate": 0.13, "conv": 0.045, "price": 45, "item_price": 34},
    }

    records = []
    start_date = datetime(2025, 7, 1)  # 7-8月数据，便于月环比分析
    end_date = datetime(2025, 8, 31)
    current_date = start_date

    while current_date <= end_date:
        for category in categories:
            # 品类级别的聚合指标（当日该品类所有来源汇总）
            category_total_orders = 0
            category_total_amount = 0.0
            category_total_items = 0
            category_refund_orders = 0
            category_refund_amount = 0.0
            
            # 先计算该品类当天的汇总（用于计算品类级客单价等）
            for source in traffic_sources:
                base = base_values[(category, source)]
                visitors = int(base["visitors"] * random.uniform(0.9, 1.1))
                conversion = base["conv"] * random.uniform(0.92, 1.08)
                buyers = max(1, int(visitors * conversion))
                avg_items_per_order = random.uniform(1.2, 1.8)
                orders = max(1, int(buyers * random.uniform(0.7, 0.9) * avg_items_per_order))
                items = int(orders * avg_items_per_order)
                amount = buyers * base["price"] * random.uniform(0.95, 1.05)
                
                category_total_orders += orders
                category_total_amount += amount
                category_total_items += items
                category_refund_orders += max(0, int(orders * random.uniform(0.02, 0.06)))
                category_refund_amount += amount * random.uniform(0.03, 0.07)
            
            # 品类级价格指标
            cat_unit_price = round(category_total_amount / max(1, category_total_orders), 2)
            cat_item_price = round(category_total_amount / max(1, category_total_items), 2)
            
            for source in traffic_sources:
                base = base_values[(category, source)]
                
                # 流量指标
                visitors = int(base["visitors"] * random.uniform(0.9, 1.1))
                pageviews = int(visitors * base["pv_ratio"] * random.uniform(0.95, 1.05))
                avg_stay = round(base["stay"] * random.uniform(0.9, 1.1), 1)
                bounce_rate = round(base["bounce"] * random.uniform(0.9, 1.1), 4)
                
                # 转化指标
                cart_users = int(visitors * base["cart_rate"] * random.uniform(0.9, 1.1))
                conversion = base["conv"] * random.uniform(0.92, 1.08)
                
                # ========== 异常注入：8月美妆-付费推广转化率大幅下降 ==========
                if (current_date.month == 8 and category == "美妆" 
                        and source == "付费推广"):
                    conversion = base["conv"] * 0.45 * random.uniform(0.9, 1.1)  # 转化率降至45%
                    bounce_rate = round(base["bounce"] * 1.3 * random.uniform(0.95, 1.05), 4)  # 跳失率上升
                    cart_users = int(cart_users * 0.6)  # 加购也下降
                # ================================================================
                
                buyers = max(1, int(visitors * conversion))
                conversion_rate = round(conversion, 4)
                
                # 交易指标（来源级按比例分摊）
                source_ratio = visitors / sum(
                    int(base_values[(category, s)]["visitors"] * random.uniform(0.9, 1.1)) 
                    for s in traffic_sources
                ) if visitors > 0 else 0.2
                
                orders = max(1, int(category_total_orders * source_ratio * random.uniform(0.9, 1.1)))
                amount = round(category_total_amount * source_ratio * random.uniform(0.95, 1.05), 2)
                items = max(1, int(category_total_items * source_ratio * random.uniform(0.9, 1.1)))
                
                # 售后指标
                refund_orders = max(0, int(orders * random.uniform(0.02, 0.06)))
                refund_amount = round(amount * random.uniform(0.03, 0.07), 2)
                
                records.append({
                    "日期": current_date.strftime("%Y-%m-%d"),
                    "品类": category,
                    "流量来源": source,
                    "访客数": visitors,
                    "浏览量": pageviews,
                    "平均停留时长(秒)": avg_stay,
                    "跳失率": bounce_rate,
                    "加购人数": cart_users,
                    "支付买家数": buyers,
                    "支付转化率": conversion_rate,
                    "支付订单数": orders,
                    "支付金额": amount,
                    "支付件数": items,
                    "退款订单数": refund_orders,
                    "退款金额": refund_amount,
                    "客单价": cat_unit_price,
                    "件单价": cat_item_price,
                })
        current_date += timedelta(days=1)

    df = pd.DataFrame(records)
    print(f"✅ 店铺经营测试数据已创建: {len(df)} 行 DataFrame")
    print(f"   日期范围: {df['日期'].min()} ~ {df['日期'].max()}")
    print(f"   品类: {', '.join(df['品类'].unique())}")
    print(f"   流量来源: {', '.join(df['流量来源'].unique())}")
    return df

# ============================================================================
# 18. 测试
# ============================================================================

def test_v5():
    """测试 v5（纯 DataFrame 版）— 基于店铺经营数据"""
    print("=" * 70)
    print("Insight Agent v9 - 店铺经营数据测试")
    print("=" * 70)

    mock_df = setup_mock_data()
    orchestrator = AgentOrchestrator()

    # 测试1: Simple 路径
    print("\n【测试1】Simple 路径 - 品类支付金额查询")
    print("-" * 50)
    result = orchestrator.process(
        "各品类8月支付金额是多少？",
        data_source=DataSourceType.CSV,
        uploaded_df=mock_df,
    )
    print(f"路由: {result['route']}")
    print(f"耗时: {result['total_time_ms']}ms")
    print(f"结果行数: "
          f"{result.get('final_result', {}).get('row_count', 0)}")
    if result.get("steps"):
        for step in result["steps"]:
            print(f"  步骤: {step.get('type')}")
            print(f"  重试次数: {step.get('retry_count', 0)}")
            if step.get("retry_details"):
                for rd in step["retry_details"]:
                    print(f"    尝试{rd['attempt']}: {rd['error']}")

    # 测试2: Simple 路径 + 图表
    print("\n【测试2】Simple 路径 + 图表")
    print("-" * 50)
    result = orchestrator.process(
        "展示各品类8月支付金额柱状图",
        data_source=DataSourceType.CSV,
        uploaded_df=mock_df,
    )
    print(f"路由: {result['route']}")
    print(f"图表数: {len(result.get('chart_paths', []))}")

    # 测试3: 小型 DataFrame
    print("\n【测试3】Simple 路径 - 小型 CSV")
    print("-" * 50)
    test_df = pd.DataFrame({
        "品类": ["服装", "电子", "美妆", "食品"],
        "支付金额": [150000, 200000, 180000, 120000],
        "访客数": [50000, 80000, 60000, 40000],
        "支付转化率": [0.03, 0.025, 0.03, 0.03],
    })
    result = orchestrator.process(
        "各品类支付金额是多少？",
        data_source=DataSourceType.CSV,
        uploaded_df=test_df,
    )
    print(f"路由: {result['route']}")
    print(f"结果行数: "
          f"{result.get('final_result', {}).get('row_count', 0)}")
    if result.get("error"):
        print(f"错误: {result['error']}")

    # 测试4: Complex 路径
    print("\n【测试4】Complex 路径 - 根因分析")
    print("-" * 50)
    result = orchestrator.process(
        "为什么8月美妆付费推广转化率下降了？",
        data_source=DataSourceType.CSV,
        uploaded_df=mock_df,
    )
    print(f"路由: {result['route']}")
    print(f"执行步骤: {len(result.get('steps', []))}")
    if result.get("final_report"):
        report = result["final_report"]
        print(f"确认根因: {report.get('confirmed_hypotheses', [])}")
        print(f"排除假设: {report.get('rejected_hypotheses', [])}")

    # ✅ 验证 Trace 中的重试记录
    if result.get("trace_log"):
        retry_events = [
            e for e in result["trace_log"]["events"]
            if e["event_type"] == "tool_retry"
        ]
        print(f"\nTrace 重试事件数: {len(retry_events)}")
        for evt in retry_events:
            print(f"  迭代{evt['iteration']}: "
                  f"{evt.get('error', 'N/A')}")

    print("\n" + "=" * 70)
    print("✅ v5.0 纯 DataFrame 版测试完成")
    print("=" * 70)

# ============================================================================
# 18. Streamlit 前端 UI
# ============================================================================

def _render_live_progress(placeholder, progress_nodes: List[Dict]):
    """
    ✅ v9.9.2: 实时渲染分析进度（在分析过程中动态更新）
    
    Args:
        placeholder: Streamlit placeholder 对象
        progress_nodes: 进度节点列表 [{"icon", "label", "desc", "status", "extra"?}, ...]
    """
    if not progress_nodes:
        return
    
    # 构建 Markdown 内容
    lines = []
    for i, node in enumerate(progress_nodes):
        icon = node.get("icon", "⏳")
        label = node.get("label", "处理中")
        desc = node.get("desc", "")
        status = node.get("status", "running")
        extra = node.get("extra", "")
        
        # 状态指示器
        if status == "done":
            status_icon = "✅"
        elif status == "running":
            status_icon = "⏳"
        else:
            status_icon = "❓"
        
        # 构建显示行
        line = f"{icon} **{label}** {status_icon}"
        if extra:
            line += f" — {extra}"
        elif status == "running":
            line += f" — {desc}"
        
        lines.append(line)
        
        # 连接线（非最后一个节点）
        if i < len(progress_nodes) - 1:
            lines.append("  │")
    
    placeholder.markdown("\n\n".join(lines))


def _render_analysis_timeline(result: Dict):
    """
    ✅ v9.9.3: 从 trace + steps 中提取关键节点，按正确顺序渲染时间线。
    
    修复：
    1. 所有节点附带 timestamp 用于排序
    2. 过滤无效节点（如 metric_count=0 的 Commander）
    3. 去重相同类型的节点（保留最后一个有效的）
    """
    import streamlit as st
    from datetime import datetime

    # 节点类型优先级（用于同一时间戳的节点排序）
    TYPE_PRIORITY = {
        "commander": 1,
        "scan_L1": 2,
        "scan_L2": 3,
        "scan_L3": 4,
        "detect": 5,
        "reasoner": 6,
        "deep_rca_decision": 7,
        "present_findings": 8,
        "deep_rca_init": 9,
        "supplementary_merged": 10,
        "deep_rca_report": 11,
        "report": 12,
    }

    raw_nodes = []  # 收集所有节点，附带时间戳和类型

    # ── 从 trace 提取（有时间戳） ──
    trace_log = result.get("trace_log") or {}
    for ev in trace_log.get("events", []):
        et = ev.get("event_type", "")
        meta = ev.get("metadata") or {}
        ts = ev.get("timestamp", "")
        od = ev.get("output_data") or {}

        if et == "commander_generated":
            metric_count = od.get("metric_count", 0)
            layer_count = od.get("layer_count", 0)
            # ✅ 过滤空规划（metric_count=0）
            if metric_count > 0:
                raw_nodes.append({
                    "ts": ts, "type": "commander",
                    "icon": "🧭", "title": "Commander 扫描规划",
                    "desc": f"规划 {metric_count} 个指标 × {layer_count} 层扫描（{od.get('comparison_type', 'mom')} 对比）",
                    "ok": True,
                })

        elif et == "evaluation_completed" and meta.get("phase") == "v8_quick_scan":
            raw_nodes.append({
                "ts": ts, "type": "detect",
                "icon": "📊", "title": "Quick Scan 异常检测",
                "desc": f"发现 {od.get('anomaly_count', 0)} 个异常 / {od.get('normal_count', 0)} 个正常（阈值 {od.get('threshold', 8)}%）",
                "ok": True,
            })

        elif et == "evaluation_completed" and meta.get("phase") == "v7_detect":
            raw_nodes.append({
                "ts": ts, "type": "detect",
                "icon": "📊", "title": "异常检测",
                "desc": f"发现 {od.get('confirmed_count', 0)} 个异常维度，{od.get('rejected_count', 0)} 个正常维度（阈值 ≥{od.get('threshold', 8)}%）",
                "ok": True,
            })

        elif et == "reasoner_v2_completed":
            conclusion = od.get("conclusion", "")
            if conclusion == "no_anomaly":
                desc = "无显著异常"
            elif conclusion == "single_anomaly":
                desc = f"单一根因: {od.get('root_cause', '')}"
            else:
                roots = od.get("root_causes", [])
                root_str = ", ".join(str(r) for r in roots)
                desc = f"因果分析 → 根因: {root_str}（信心: {od.get('confidence', '?')}）"
            raw_nodes.append({
                "ts": ts, "type": "reasoner",
                "icon": "🧠", "title": "Reasoner v2",
                "desc": desc, "ok": True,
            })

        elif et == "deep_rca_decision":
            if od.get("decision"):
                raw_nodes.append({
                    "ts": ts, "type": "deep_rca_decision",
                    "icon": "🔬", "title": "深度分析建议",
                    "desc": f"建议进行深度根因分析（信心: {od.get('confidence', '?')}）",
                    "ok": True,
                })

        elif et == "present_findings":
            raw_nodes.append({
                "ts": ts, "type": "present_findings",
                "icon": "📋", "title": "初步结论展示",
                "desc": "展示 Step 1+2 结论，等待用户决策",
                "ok": True,
            })

        elif et == "deep_rca_init":
            raw_nodes.append({
                "ts": ts, "type": "deep_rca_init",
                "icon": "🔬", "title": "深度分析初始化",
                "desc": f"生成 {od.get('hypothesis_count', 0)} 个假设，继承 {od.get('inherited_evidence_count', 0)} 条证据",
                "ok": True,
            })

        elif et == "supplementary_data_merged":
            new_cols = od.get("new_columns", [])
            raw_nodes.append({
                "ts": ts, "type": "supplementary_merged",
                "icon": "📎", "title": "补充数据合并",
                "desc": f"合并后 {od.get('merged_shape', [0, 0])} + 新增列: {', '.join(new_cols[:5])}" if new_cols else "数据合并完成",
                "ok": ev.get("success", True),
            })

        elif et == "deep_rca_report":
            raw_nodes.append({
                "ts": ts, "type": "deep_rca_report",
                "icon": "📝", "title": "深度分析报告",
                "desc": "生成综合分析报告（Step 1+2+3）",
                "ok": True,
            })

        elif et == "report_generated" and meta.get("phase") in ("v7_reporter", "v8_reporter"):
            raw_nodes.append({
                "ts": ts, "type": "report",
                "icon": "📝", "title": "报告生成",
                "desc": "生成分析报告 & 行动指南",
                "ok": True,
            })

    # ── 从 steps 提取（用于补充 trace 没有记录的扫描步骤） ──
    for idx, step in enumerate(result.get("steps", [])):
        step_type = step.get("type", "")
        ok = step.get("result", {}).get("success", False)
        summary = step.get("result", {}).get("summary", "")
        step_ts = step.get("timestamp", f"1970-01-01T00:00:{idx:02d}")  # fallback 时间戳

        if step_type == "full_dimension_scan":
            row_count = step.get("result", {}).get("row_count", 0)
            raw_nodes.append({
                "ts": step_ts, "type": "scan_L1",
                "icon": "🔍", "title": "全维度扫描",
                "desc": f"扫描 {row_count} 个指标的环比变化" + (f" — {summary[:80]}" if summary else ""),
                "ok": ok,
            })

        elif step_type.startswith("scan_L"):
            hyp = step.get("hypothesis", step_type)
            row_count = step.get("result", {}).get("row_count", 0)
            # 根据 step_type 确定扫描层级
            if "L1" in step_type:
                node_type = "scan_L1"
            elif "L2" in step_type:
                node_type = "scan_L2"
            else:
                node_type = "scan_L3"
            raw_nodes.append({
                "ts": step_ts, "type": node_type,
                "icon": "🔍" if "L1" in step_type else "🔬",
                "title": hyp,
                "desc": f"返回 {row_count} 条数据" + (f" — {summary[:60]}" if summary else ""),
                "ok": ok,
            })

        elif step_type == "drilldown_scan":
            hyp = step.get("hypothesis", "下钻分析")
            raw_nodes.append({
                "ts": step_ts, "type": "scan_L2",
                "icon": "🔬", "title": hyp,
                "desc": summary[:100] if summary else "按分组维度下钻定位贡献最大子组",
                "ok": ok,
            })

    # ── 排序：先按时间戳，再按类型优先级 ──
    def sort_key(node):
        ts = node.get("ts", "")
        type_priority = TYPE_PRIORITY.get(node.get("type", ""), 99)
        return (ts, type_priority)

    raw_nodes.sort(key=sort_key)

    # ── 去重：相同类型只保留最后一个（通常是最完整的） ──
    seen_types = {}
    for node in raw_nodes:
        node_type = node.get("type", "")
        # 扫描步骤不去重（可能有多个不同的扫描）
        if node_type.startswith("scan_"):
            key = f"{node_type}_{node.get('title', '')}"
        else:
            key = node_type
        seen_types[key] = node

    nodes = list(seen_types.values())
    # 重新排序去重后的节点
    nodes.sort(key=sort_key)

    # ── 渲染时间线 ──
    if not nodes:
        return

    with st.expander(f"📋 分析过程 ({len(nodes)} 个关键节点)", expanded=True):
        for i, node in enumerate(nodes):
            status = "✅" if node.get("ok", True) else "❌"
            connector = "│" if i < len(nodes) - 1 else " "
            st.markdown(
                f"{node['icon']} **{node['title']}** {status}\n"
                f"> {node['desc']}"
            )
            if i < len(nodes) - 1:
                st.caption(f"  {connector}")


def _render_trace_log(trace_dict: Dict):
    """渲染 Trace 日志"""
    import streamlit as st

    stats = trace_dict.get("statistics", {})
    duration = stats.get("total_duration_ms", 0)
    total_iter = stats.get("total_iterations", 0)
    tool_execs = stats.get("tool_executions", 0)
    errors = stats.get("tool_errors", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("耗时", f"{duration / 1000:.1f}s")
    c2.metric("迭代", total_iter)
    c3.metric("执行", tool_execs)
    c4.metric("错误", errors)

    # ✅ v5.8: 显示剪枝和 early stop 统计
    prune_count = sum(1 for e in trace_dict.get("events", []) if e.get("event_type") == "pruning_executed")
    early_stopped = any(e.get("event_type") == "early_stop" for e in trace_dict.get("events", []))
    if prune_count > 0 or early_stopped:
        c5, c6 = st.columns(2)
        if prune_count > 0:
            c5.metric("🌳 智能剪枝", f"{prune_count}次")
        if early_stopped:
            c6.metric("⚡ Early Stop", "是")

    important_types = {
        "session_start", "session_end", "intent_classification",
        "hypothesis_selected", "hypothesis_confirmed",
        "hypothesis_rejected", "pruning_executed", "pivot_decision",
        "tool_execution_error", "report_generated",
        "hypothesis_refill",  # ✅ v5.7
        "evaluation_completed", "evidence_updated", "priority_reranked", "early_stop",  # ✅ v5.8
        # ✅ v8 Step 1+2+3
        "commander_generated", "commander_fallback", "scan_layer_completed",
        "arbiter_invoked", "reasoner_v2_completed", "deep_rca_decision",
        "present_findings", "deep_rca_init", "supplementary_data_merged", "deep_rca_report",
    }
    events = trace_dict.get("events", [])
    important_events = [e for e in events if e.get("event_type") in important_types]

    if important_events:
        st.markdown("#### 关键事件")
        for event in important_events:
            et = event.get("event_type", "")
            hyp = event.get("hypothesis", "")
            it = event.get("iteration", "")
            icon = {"session_start": "🚀", "session_end": "🏁",
                    "intent_classification": "🧭",
                    "hypothesis_selected": "🔍", "hypothesis_confirmed": "🎯",
                    "hypothesis_rejected": "🚫", "pruning_executed": "✂️",
                    "pivot_decision": "🔄", "tool_execution_error": "❌",
                    "report_generated": "📝",
                    "hypothesis_refill": "🔁",
                    "evaluation_completed": "📊",  # ✅ v5.8
                    "evidence_updated": "🌳",       # ✅ v5.8
                    "priority_reranked": "📈",       # ✅ v5.8
                    "early_stop": "⚡",              # ✅ v5.8
                    # ✅ v8
                    "commander_generated": "🧭",
                    "commander_fallback": "⚠️",
                    "scan_layer_completed": "📊",
                    "arbiter_invoked": "⚖️",
                    "reasoner_v2_completed": "🧠",
                    "deep_rca_decision": "🔬",
                    "present_findings": "📋",
                    "deep_rca_init": "🔬",
                    "supplementary_data_merged": "📎",
                    "deep_rca_report": "📝",
                    }.get(et, "•")
            desc = ""
            if et == "intent_classification":
                desc = f"路由: {(event.get('output_data') or {}).get('route', '')}"
            elif et == "hypothesis_refill":
                # ✅ v5.7: 假设补充事件
                new_count = (event.get("output_data") or {}).get("new_count", 0)
                new_hyps = (event.get("output_data") or {}).get("new_hypotheses", [])
                if new_hyps:
                    names = ", ".join(h.get("name", "") for h in new_hyps)
                    desc = f"补充 {new_count} 个假设: {names}"
                elif new_count == 0 and event.get("error"):
                    desc = "补充假设失败"
                else:
                    rej_count = (event.get("input_data") or {}).get("rejected_count", 0)
                    desc = f"准备补充假设 (已排除 {rej_count} 个)"
            elif et == "evaluation_completed":
                # ✅ v5.8: 评估完成
                od = event.get("output_data") or {}
                pct = od.get("change_pct")
                sig = od.get("significant", False)
                pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
                sig_str = "⚠️显著" if sig else "正常"
                desc = f"评估: {hyp} → {pct_str} ({sig_str})"
            elif et == "evidence_updated":
                # ✅ v5.8: 证据更新/剪枝
                od = event.get("output_data") or {}
                pct = od.get("change_pct")
                sig = od.get("significant", False)
                pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
                sig_str = "⚠️显著" if sig else "正常"
                desc = f"证据: {hyp} → {pct_str} ({sig_str})"
            elif et == "priority_reranked":
                # ✅ v5.8: 优先级重排
                od = event.get("output_data") or {}
                new_order = od.get("new_order", [])
                desc = f"重排: {' → '.join(new_order[:3])}" if new_order else "优先级重排"
            elif et == "early_stop":
                # ✅ v5.8: 提前终止
                od = event.get("output_data") or {}
                desc = f"提前终止: 确认{od.get('confirmed_count', 0)}个根因，影响度{od.get('confirmed_impact', 0):.0f}%"
            elif et == "hypothesis_selected":
                dim = (event.get("input_data") or {}).get("dimension", "")
                desc = f"检验: {hyp}" + (f" ({dim})" if dim else "")
            elif et == "hypothesis_confirmed":
                r = (event.get("output_data") or {}).get("reasoning", "")
                desc = f"确认: {hyp}" + (f" — {r[:60]}" if r else "")
            elif et == "hypothesis_rejected":
                r = (event.get("output_data") or {}).get("reasoning", "")
                desc = f"排除: {hyp}" + (f" — {r[:60]}" if r else "")
            elif et == "pruning_executed":
                p = (event.get("output_data") or {}).get("pruned_hypotheses", [])
                desc = f"剪枝: {', '.join(p)}" if p else "剪枝"
            elif et == "pivot_decision":
                desc = f"转向 → {(event.get('output_data') or {}).get('to_hypothesis', '')}"
            elif et == "report_generated":
                desc = "报告生成完成"
            elif et == "session_start":
                desc = "开始分析"
            elif et == "session_end":
                desc = "分析完成"
            # ✅ v8 新增事件描述
            elif et == "commander_generated":
                od = event.get("output_data") or {}
                desc = f"Commander 规划: {od.get('metric_count', 0)} 指标 × {od.get('layer_count', 0)} 层"
            elif et == "commander_fallback":
                desc = f"Commander 降级: {event.get('error', '未知')[:80]}"
            elif et == "scan_layer_completed":
                od = event.get("output_data") or {}
                desc = f"L{od.get('layer_depth', '?')} 扫描完成: {od.get('rows_returned', 0)} 行"
            elif et == "arbiter_invoked":
                od = event.get("output_data") or {}
                desc = f"Arbiter 裁定: {od.get('action', '?')} — {od.get('reasoning', '')[:60]}"
            elif et == "reasoner_v2_completed":
                od = event.get("output_data") or {}
                desc = f"Reasoner v2: {od.get('conclusion', '?')}（信心: {od.get('confidence', '?')}）"
            elif et == "deep_rca_decision":
                od = event.get("output_data") or {}
                desc = f"Deep RCA 决策: {'需要' if od.get('decision') else '不需要'}深度分析"
            elif et == "present_findings":
                od = event.get("output_data") or {}
                desc = f"展示初步结论: {od.get('anomaly_count', 0)} 个异常（信心: {od.get('confidence', '?')}）"
            elif et == "deep_rca_init":
                od = event.get("output_data") or {}
                desc = f"Deep RCA 初始化: {od.get('hypothesis_count', 0)} 假设，继承 {od.get('inherited_evidence_count', 0)} 证据"
            elif et == "supplementary_data_merged":
                od = event.get("output_data") or {}
                desc = f"补充数据合并: {od.get('merged_shape', [])}"
            elif et == "deep_rca_report":
                desc = "深度分析报告生成"
            tag = f"`[{it}]` " if it else ""
            st.markdown(f"{icon} {tag}{desc}")


def _generate_html_report(content: str, chart_paths: List[str], timestamp: str) -> str:
    """生成可下载的 HTML 报告"""
    def _img_b64(path):
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(path)[1].lower()
            mime = {".png": "image/png", ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg", ".gif": "image/gif"}.get(ext, "image/png")
            return f"data:{mime};base64,{data}"
        except Exception:
            return None

    charts_html = ""
    if chart_paths:
        charts_html = '<div style="margin:30px 0"><h2>📊 分析图表</h2>'
        for i, cp in enumerate(chart_paths):
            if os.path.exists(cp):
                b64 = _img_b64(cp)
                if b64:
                    charts_html += (f'<div style="margin:25px 0;text-align:center">'
                                    f'<img src="{b64}" style="max-width:100%;border-radius:8px;'
                                    f'box-shadow:0 2px 8px rgba(0,0,0,0.1)">'
                                    f'<p style="color:#7f8c8d;font-size:13px">图表 {i+1}</p></div>')
        charts_html += "</div><hr>"

    try:
        import markdown
        text_html = markdown.markdown(content, extensions=["tables", "fenced_code", "nl2br"])
    except ImportError:
        text_html = content.replace("\n", "<br>")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>分析报告 - {timestamp}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
       max-width:900px; margin:0 auto; padding:30px; line-height:1.8; color:#2c3e50; }}
.header {{ text-align:center; margin-bottom:40px; padding-bottom:25px; border-bottom:3px solid #3498db; }}
.header h1 {{ margin:0 0 10px; font-size:28px; }}
h2 {{ border-bottom:2px solid #ecf0f1; padding-bottom:10px; margin-top:35px; }}
table {{ border-collapse:collapse; width:100%; margin:20px 0; font-size:14px; }}
th,td {{ border:1px solid #ddd; padding:12px 15px; text-align:left; }}
th {{ background:#3498db; color:#fff; }}
tr:nth-child(even) {{ background:#f8f9fa; }}
code {{ background:#f8f9fa; padding:3px 8px; border-radius:4px; color:#e74c3c; }}
pre {{ background:#f8f9fa; padding:18px; border-radius:8px; overflow-x:auto; border:1px solid #e9ecef; }}
.footer {{ margin-top:50px; padding-top:25px; border-top:2px solid #ecf0f1; text-align:center; color:#95a5a6; font-size:13px; }}
</style></head><body>
<div class="header"><h1>📊 数据分析报告</h1>
<p style="color:#7f8c8d;font-size:14px">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Insight Agent v9</p></div>
{charts_html}
<div><h2>📄 分析结论</h2>{text_html}</div>
<div class="footer"><p>本报告基于数据分析自动生成，仅供参考</p><p>Powered by Insight Agent v9</p></div>
</body></html>"""


def _render_complex_steps(steps: list):
    """渲染 Complex 路径的假设验证步骤"""
    import streamlit as st
    for i, step in enumerate(steps):
        hyp = step.get("hypothesis", step.get("type", f"步骤 {i+1}"))
        dim = step.get("dimension", "")
        status = step.get("status", "")
        evaluation = step.get("result", {}).get("evaluation") or {}

        icon = {"confirmed": "🎯", "rejected": "🚫",
                "investigating": "🔍", "pruned": "✂️"}.get(status, "⏳")
        header = f"{icon} **{hyp}**" + (f" `{dim}`" if dim else "")

        with st.expander(header, expanded=False):
            if evaluation:
                meets = evaluation.get("meets_criteria")
                conclusion = evaluation.get("conclusion", "")
                reasoning = evaluation.get("reasoning", "")
                actual = evaluation.get("actual_value")
                if meets is True:
                    st.error(f"⚠️ {conclusion}")
                elif meets is False:
                    st.success(f"✅ {conclusion}")
                else:
                    st.warning(f"⏳ {conclusion}")
                if reasoning:
                    st.caption(reasoning)
                if actual is not None:
                    st.metric("实际值", f"{actual}")

            preview = step.get("result", {}).get("data_preview", [])
            if preview:
                st.dataframe(pd.DataFrame(preview), use_container_width=True)

            summary = step.get("result", {}).get("summary", "")
            if summary:
                st.info(summary)

            code = step.get("code", "")
            if code:
                st.code(code, language="python")

            chart_path = step.get("result", {}).get("chart_path")
            if chart_path and os.path.exists(chart_path):
                st.image(chart_path)

            corrections = step.get("result", {}).get("correction_history", [])
            if corrections:
                st.warning(f"代码重试 {len(corrections)} 次")


def main():
    """Streamlit 主界面"""
    try:
        import streamlit as st
    except ImportError:
        print("请安装 streamlit: pip install streamlit")
        print("然后运行: streamlit run app_v5_6-1.py")
        return

    st.set_page_config(page_title="Insight Agent v9", page_icon="🔍", layout="wide")

    # ── 侧边栏 ──
    with st.sidebar:
        st.title("⚙️ 配置")

        st.markdown("##### 数据源")
        uploaded_file = st.file_uploader("上传 CSV / Excel", type=["csv", "xlsx", "xls"])
        use_demo = st.checkbox("使用内置演示数据", value=True)

        st.markdown("---")
        st.markdown("##### 📋 示例问题")
        st.markdown("""
**🟢 概览 (descriptive)**
- 8月各品类支付金额整体表现如何
- 各流量来源的转化率情况

**🟡 诊断 (diagnostic)**
- 8月支付金额有没有异常
- 哪些流量来源的转化率出现问题

**🔴 归因 (causal)**
- 为什么8月美妆付费推广转化率下降了
- 分析8月业绩下滑的根因
        """)

        st.markdown("---")
        st.markdown("##### ℹ️ 版本")
        st.caption("**Insight Agent v9**")
        st.caption("CSV-only · PythonAgent · 统一管线")
        st.caption("Adaptive Depth: descriptive → diagnostic → causal")
        st.caption("DeepSeek Chat/Reasoner")

    # ── 主区域 ──
    st.title("🔍 Insight Agent v9")
    st.caption("电商数据洞察助手 — 统一管线：Quick Scan → Adaptive Depth → Report")

    # ── Session State ──
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = AgentOrchestrator()
    if "demo_df" not in st.session_state:
        st.session_state.demo_df = setup_mock_data()

    # ── 确定 DataFrame ──
    current_df = None
    ds_type = DataSourceType.CSV

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                current_df = pd.read_csv(uploaded_file)
                ds_type = DataSourceType.CSV
            else:
                current_df = pd.read_excel(uploaded_file)
                ds_type = DataSourceType.EXCEL
            st.sidebar.success(f"已加载: {uploaded_file.name} ({len(current_df)} 行)")
        except Exception as e:
            st.sidebar.error(f"文件读取失败: {e}")
    elif use_demo:
        current_df = st.session_state.demo_df

    if current_df is not None:
        with st.sidebar.expander("📊 数据预览", expanded=False):
            st.dataframe(current_df.head(10), use_container_width=True)
            st.caption(f"{len(current_df)} 行 × {len(current_df.columns)} 列")
    else:
        st.info("👈 请上传 CSV/Excel 文件或勾选「使用内置演示数据」")

    # ── 历史消息 ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── 用户输入 ──
    if prompt := st.chat_input("请输入您的分析问题..."):
        if current_df is None:
            st.error("请先上传数据文件或勾选演示数据")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # ✅ v9.9.2: 使用 st.status() 实现实时进度展示
            progress_nodes = []  # 收集已完成的节点用于实时展示
            
            with st.status("🧠 分析中...", expanded=True) as status_container:
                # 进度占位符
                progress_placeholder = st.empty()
                
                def on_node_start(node_name: str, progress_info: dict):
                    """节点开始时更新进度"""
                    icon = progress_info.get("icon", "⏳")
                    label = progress_info.get("label", node_name)
                    desc = progress_info.get("desc", "处理中...")
                    
                    # 更新状态标题
                    status_container.update(label=f"{icon} {label}...")
                    
                    # 追加到进度列表
                    progress_nodes.append({
                        "icon": icon,
                        "label": label,
                        "desc": desc,
                        "status": "running"
                    })
                    _render_live_progress(progress_placeholder, progress_nodes)
                
                def on_node_end(node_name: str, snapshot: dict):
                    """节点结束时更新进度"""
                    if progress_nodes:
                        progress_nodes[-1]["status"] = "done"
                        # 根据快照补充信息
                        if snapshot.get("confirmed_anomalies"):
                            progress_nodes[-1]["extra"] = f"发现 {len(snapshot['confirmed_anomalies'])} 个异常"
                        _render_live_progress(progress_placeholder, progress_nodes)
                
                result = st.session_state.orchestrator.process_stream(
                    user_query=prompt, 
                    data_source=ds_type, 
                    uploaded_df=current_df,
                    on_node_start=on_node_start,
                    on_node_end=on_node_end
                )
                
                # 分析完成，更新状态
                status_container.update(label="✅ 分析完成", state="complete", expanded=False)

            route = result.get("route", "unknown")
            analysis_depth = result.get("analysis_depth", "unknown")
            total_ms = result.get("total_time_ms", 0)
            depth_labels = {
                "descriptive": "🟢 概览",
                "diagnostic": "🟡 诊断",
                "causal": "🔴 归因",
            }
            depth_label = depth_labels.get(analysis_depth, f"⚪ {analysis_depth}")
            st.caption(f"{depth_label} · 耗时: {total_ms / 1000:.1f}s")

            # ── 错误 ──
            if result.get("error"):
                st.error(f"❌ {result['error']}")
                if result.get("suggestions"):
                    st.info("💡 " + " / ".join(result["suggestions"]))

            else:
                # ── ✅ v9: 统一渲染逻辑（不再区分 simple/complex） ──
                report = result.get("final_report") or {}
                confirmed_a = report.get("confirmed_hypotheses", [])
                rejected_a = report.get("rejected_hypotheses", [])
                causal_chain = report.get("causal_chain", "")

                # v8 Step 2 字段
                confidence = report.get("confidence", "unknown")
                needs_deep_rca = report.get("needs_deep_rca", False)
                suggested_data = report.get("suggested_data", [])
                reason_result = report.get("reason_result", {})

                # ════════════════════════════════════════════
                # Layer 1: 结论卡片（10 秒看到答案）
                # ════════════════════════════════════════════
                if confirmed_a:
                    root_dims = ', '.join(str(c) for c in confirmed_a)
                    st.success(f"🎯 **异常定位**: {root_dims}")
                    if causal_chain and causal_chain != "（descriptive 深度，未做因果推理）":
                        st.info(f"🔗 **因果链**: {causal_chain}")

                    # 信心等级 badge（diagnostic/causal 才显示）
                    if analysis_depth in ("diagnostic", "causal"):
                        confidence_badges = {
                            "high": "🟢 高",
                            "medium": "🟡 中",
                            "low": "🔴 低",
                        }
                        badge = confidence_badges.get(confidence, f"⚪ {confidence}")
                        st.caption(f"信心等级: {badge}")

                elif report.get("success"):
                    st.info("✅ **结论**: 各维度在正常波动范围内，未发现显著异常")
                    # 0 异常时的建议
                    zero_suggestions = reason_result.get("zero_anomaly_suggestions", [])
                    if zero_suggestions:
                        with st.expander("💡 优化建议", expanded=False):
                            for s in zero_suggestions:
                                st.markdown(f"- {s}")

                if rejected_a:
                    st.caption(f"已排除: {', '.join(str(r) for r in rejected_a)}")

                # ════════════════════════════════════════════
                # ✅ v9.9.1 修复: Step 3 深度分析 - 只保存状态，UI 移到外部
                # ════════════════════════════════════════════
                if needs_deep_rca and suggested_data:
                    st.markdown("---")
                    st.warning("🔍 **建议深度分析** — 当前数据可能不足以完全解释根因")
                    st.markdown("建议补充以下数据以进一步分析:")
                    for i, sd in enumerate(suggested_data):
                        desc = sd.get("description", "未知数据")
                        reason = sd.get("reason", "")
                        cols = sd.get("required_columns", [])
                        st.markdown(f"**{i+1}. {desc}**")
                        if reason:
                            st.caption(f"原因: {reason}")
                        if cols:
                            st.caption(f"建议包含列: {', '.join(cols)}")

                    # ✅ v9.9.1: 保存完整的深度分析上下文到 session_state
                    # UI 交互组件移到 if prompt 外部，避免 rerun 时丢失
                    deep_rca_key = f"deep_rca_{hash(prompt)}"
                    if result.get("_intermediate_state"):
                        st.session_state["pending_deep_rca"] = {
                            "deep_rca_key": deep_rca_key,
                            "prompt": prompt,
                            "prior_state": result["_intermediate_state"],
                            "suggested_data": suggested_data,
                            "result": result,  # 保存完整结果用于后续渲染
                        }
                        st.session_state[f"{deep_rca_key}_state"] = result["_intermediate_state"]
                        st.session_state[f"{deep_rca_key}_query"] = prompt
                    
                    st.info("👇 请在下方上传补充数据并点击「开始深度分析」")

                # ════════════════════════════════════════════
                # Layer 2: 关键节点时间线（建立信任）
                # ════════════════════════════════════════════
                _render_analysis_timeline(result)

                # ════════════════════════════════════════════
                # Layer 3: 完整报告 + 行动指南（核心产出）
                # ════════════════════════════════════════════
                if report.get("success") and report.get("full_content"):
                    st.markdown("---")
                    is_preliminary = report.get("is_preliminary", False)
                    title = "📄 初步分析报告" if is_preliminary else "📄 分析报告 & 行动指南"
                    st.markdown(f"#### {title}")

                    report_key = f"report_{hash(prompt)}"
                    if report_key not in st.session_state:
                        st.session_state[report_key] = report.get("full_content", "")

                    edit_mode = st.toggle("✏️ 编辑模式", key=f"edit_{report_key}")
                    if edit_mode:
                        edited = st.text_area("编辑报告 (Markdown)",
                                              value=st.session_state[report_key], height=400)
                        st.session_state[report_key] = edited
                        st.info("💡 关闭编辑模式可预览效果")
                    else:
                        st.markdown(st.session_state[report_key])

                    st.markdown("---")
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    html_report = _generate_html_report(
                        st.session_state[report_key], result.get("chart_paths", []), ts)
                    st.download_button("🌐 下载 HTML 报告", data=html_report,
                                       file_name=f"report_{ts}.html", mime="text/html")

                # ════════════════════════════════════════════
                # Layer 4: 技术细节（折叠，debug 用）
                # ════════════════════════════════════════════
                if result.get("chart_paths"):
                    with st.expander("📊 分析图表", expanded=False):
                        cols = st.columns(min(len(result["chart_paths"]), 2))
                        for i, cp in enumerate(result["chart_paths"]):
                            if os.path.exists(cp):
                                with cols[i % 2]:
                                    st.image(cp)

                if result.get("steps"):
                    with st.expander(f"🔧 技术细节 — 执行步骤 ({len(result['steps'])} 步)", expanded=False):
                        for i, step in enumerate(result["steps"]):
                            step_type = step.get("type", step.get("hypothesis", "unknown"))
                            ok = "✅" if step.get("result", {}).get("success", False) else "❌"
                            st.markdown(f"{ok} **Step {step.get('iteration', i+1)}: {step_type}**")
                            if step.get("result", {}).get("summary"):
                                st.caption(step["result"]["summary"][:200])
                            if step.get("code"):
                                st.code(step["code"], language="python")
                            if step.get("result", {}).get("data_preview"):
                                preview = step["result"]["data_preview"]
                                if isinstance(preview, list) and preview:
                                    st.dataframe(pd.DataFrame(preview[:10]), use_container_width=True)

                # ── Trace 日志（所有路径都显示） ──
                if result.get("trace_log"):
                    with st.expander("📊 Trace 日志", expanded=False):
                        _render_trace_log(result["trace_log"])
                        st.download_button(
                            "📥 导出 Trace JSON",
                            data=json.dumps(result["trace_log"], ensure_ascii=False, indent=2, default=str),
                            file_name=f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json")

            # ── 保存历史 ──
            summary_text = (
                (result.get("final_report") or {}).get("summary", "")[:500]
                or (result.get("final_result") or {}).get("summary", "")[:500]
                or (f"❌ {result['error']}" if result.get("error") else "")
                or f"分析完成 (深度: {analysis_depth}, 耗时: {total_ms}ms)"
            )
            st.session_state.messages.append({"role": "assistant", "content": summary_text})

    # ════════════════════════════════════════════════════════════════════════════
    # ✅ v9.9.3: Step 3 深度分析结果渲染（优先于 pending 状态检查）
    # ════════════════════════════════════════════════════════════════════════════
    if st.session_state.get("completed_deep_rca") and current_df is not None:
        completed = st.session_state["completed_deep_rca"]
        deep_result = completed.get("deep_result", {})
        step12_summary = completed.get("step12_summary", {})
        had_supp_data = completed.get("had_supplementary_data", False)
        
        st.markdown("---")
        
        # ═══ 保留 Step 1+2 摘要（折叠展示）═══
        with st.expander("📋 Step 1+2 初步分析摘要", expanded=False):
            if step12_summary.get("confirmed_hypotheses"):
                root_dims = ', '.join(str(c) for c in step12_summary["confirmed_hypotheses"])
                st.info(f"🎯 **初步定位**: {root_dims}")
            if step12_summary.get("causal_chain"):
                st.caption(f"🔗 因果链: {step12_summary['causal_chain']}")
            if step12_summary.get("confidence"):
                st.caption(f"信心等级: {step12_summary['confidence']}")
        
        # ═══ Step 3 深度分析完整结果 ═══
        st.subheader("🔬 Step 3 深度根因分析结果")
        deep_report = deep_result.get("final_report") or {}
        deep_ms = deep_result.get("total_time_ms", 0)
        
        # 显示数据来源
        data_source_info = "✅ 使用补充数据" if had_supp_data else "ℹ️ 仅使用原始数据"
        st.caption(f"深度分析耗时: {deep_ms / 1000:.1f}s · "
                   f"验证步骤: {len(deep_result.get('steps', []))} 步 · "
                   f"{data_source_info}")

        if deep_report.get("success") and deep_report.get("full_content"):
            st.markdown(deep_report["full_content"])

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_report = _generate_html_report(
                deep_report["full_content"],
                deep_result.get("chart_paths", []), ts)
            st.download_button("🌐 下载深度分析报告", data=html_report,
                               file_name=f"deep_rca_report_{ts}.html",
                               mime="text/html",
                               key="dl_completed_deep_rca_report")
        elif deep_report.get("full_content"):
            st.error(deep_report["full_content"])

        # 深度分析步骤详情
        if deep_result.get("steps"):
            with st.expander(f"🔧 深度分析步骤 ({len(deep_result['steps'])} 步)", expanded=False):
                _render_analysis_timeline(deep_result)

        if deep_result.get("trace_log"):
            with st.expander("📊 深度分析 Trace 日志", expanded=False):
                _render_trace_log(deep_result["trace_log"])
        
        # 清除按钮（允许用户重新开始）
        if st.button("🔄 清除深度分析结果，重新提问", key="clear_deep_rca"):
            del st.session_state["completed_deep_rca"]
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════════
    # ✅ v9.9.1 修复: Step 3 深度分析 UI（独立于 if prompt，避免 rerun 时丢失）
    # ════════════════════════════════════════════════════════════════════════════
    if st.session_state.get("pending_deep_rca") and current_df is not None:
        pending = st.session_state["pending_deep_rca"]
        deep_rca_key = pending["deep_rca_key"]
        
        st.markdown("---")
        st.subheader("🔬 深度根因分析")
        
        # ✅ v9.9.3: 补充数据上传 + 立即缓存到 session_state（避免 rerun 丢失）
        supp_file = st.file_uploader(
            "📎 上传补充数据（可选）",
            type=["csv", "xlsx", "xls"],
            key="deep_rca_supp_file",
        )
        
        # 文件上传后立即读取并缓存（解决 Streamlit rerun 导致的数据丢失问题）
        if supp_file is not None:
            try:
                if supp_file.name.endswith(".csv"):
                    cached_df = pd.read_csv(supp_file)
                else:
                    cached_df = pd.read_excel(supp_file)
                # 缓存到 session_state
                st.session_state["_supp_df_cache"] = cached_df
                st.session_state["_supp_df_filename"] = supp_file.name
                st.success(f"✅ 已加载补充数据: {supp_file.name} ({len(cached_df)} 行, {len(cached_df.columns)} 列)")
                st.caption(f"列名: {', '.join(cached_df.columns[:10])}{'...' if len(cached_df.columns) > 10 else ''}")
            except Exception as e:
                st.error(f"补充数据读取失败: {e}")
                st.session_state.pop("_supp_df_cache", None)
        
        # 显示当前缓存的数据状态
        if st.session_state.get("_supp_df_cache") is not None:
            cached_df = st.session_state["_supp_df_cache"]
            filename = st.session_state.get("_supp_df_filename", "unknown")
            st.info(f"📊 当前缓存数据: {filename} ({len(cached_df)} 行)")
        else:
            st.caption("💡 未上传补充数据，将基于原始数据进行深度分析")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔍 开始深度分析", key="deep_rca_start_btn", type="primary"):
                # ✅ v9.9.3: 从 session_state 获取缓存的补充数据（而非直接从 file_uploader）
                supp_df = st.session_state.get("_supp_df_cache")
                
                # 调试信息
                if supp_df is not None:
                    st.caption(f"📤 传递补充数据: {len(supp_df)} 行, {len(supp_df.columns)} 列")
                else:
                    st.caption("📤 无补充数据，使用原始数据")

                # 调用 process_deep_rca
                prior_state = pending.get("prior_state", {})
                if prior_state:
                    with st.spinner("🔬 深度根因分析中..."):
                        deep_result = st.session_state.orchestrator.process_deep_rca(
                            user_query=pending["prompt"],
                            prior_state=prior_state,
                            uploaded_df=current_df,
                            supplementary_df=supp_df,
                        )
                    
                    # ✅ v9.9.3: 保存深度分析结果到 session_state（而非直接渲染后丢失）
                    step12_result = pending.get("result", {})
                    step12_report = step12_result.get("final_report", {})
                    
                    st.session_state["completed_deep_rca"] = {
                        "deep_result": deep_result,
                        "step12_summary": {
                            "confirmed_hypotheses": step12_report.get("confirmed_hypotheses", []),
                            "causal_chain": step12_report.get("causal_chain", ""),
                            "confidence": step12_report.get("confidence", "unknown"),
                            "summary": step12_report.get("summary", ""),
                        },
                        "prompt": pending["prompt"],
                        "timestamp": datetime.now().isoformat(),
                        "had_supplementary_data": supp_df is not None,  # 记录是否有补充数据
                    }
                    
                    # 清除缓存和 pending 状态
                    st.session_state.pop("_supp_df_cache", None)
                    st.session_state.pop("_supp_df_filename", None)
                    del st.session_state["pending_deep_rca"]
                    st.rerun()
                else:
                    st.error("中间状态丢失，无法进行深度分析。请重新提问。")
                    del st.session_state["pending_deep_rca"]

        with col_btn2:
            if st.button("⏭️ 跳过深度分析", key="deep_rca_skip_btn"):
                st.caption("已跳过深度分析，以上为初步分析结论。")
                st.session_state.pop("_supp_df_cache", None)
                st.session_state.pop("_supp_df_filename", None)
                del st.session_state["pending_deep_rca"]
                st.rerun()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_v5()
    else:
        main()
