#!/usr/bin/env python3
"""
consistency_check.py — 单平台单表数据一致性校验 (v3)
======================================================
对所有 12 个 case 的数据进行自洽性校验, 适配 v3 单平台单表 MVP 格式
(店铺经营概况.csv, 中文列名).

校验规则:
  1. 支付转化率 ≈ 支付买家数 / 访客数 (来源粒度, 同行)
  2. 客单价     ≈ 支付金额 / 支付订单数 (日×品类粒度)
  3. 件单价     ≈ 支付金额 / 支付件数 (日×品类粒度)
  4. 交易指标在同一天同一品类的不同流量来源行中应完全一致 (R01 粒度)
  5. 数据范围合理性 (无负值)

用法:
  python consistency_check.py --data-dir ../datasets
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np


PRIMARY_FILE = "店铺经营概况.csv"

# 交易指标列 (日 × 品类 粒度, 同一天同一品类不同来源行应一致)
TRADE_COLS = [
    "支付订单数", "支付金额", "支付件数",
    "退款订单数", "退款金额", "客单价", "件单价",
]


class ConsistencyCheckerV3:
    """v3: 单平台单表 中文列名 自洽性校验"""

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance

    def check_case(self, case_dir: str) -> dict:
        fpath = os.path.join(case_dir, PRIMARY_FILE)
        if not os.path.exists(fpath):
            return {"status": "skip", "reason": f"{PRIMARY_FILE} 不存在", "violations": []}

        df = pd.read_csv(fpath)
        violations = []

        required = {"日期", "品类", "流量来源", "访客数", "支付买家数",
                    "支付转化率", "支付订单数", "支付金额", "客单价"}
        missing = required - set(df.columns)
        if missing:
            return {"status": "fail",
                    "violations": [{"rule": "missing_columns",
                                    "description": f"缺失列: {sorted(missing)}",
                                    "severity": "critical"}]}

        # 规则 1: 支付转化率 ≈ 支付买家数 / 访客数 (来源粒度)
        uv = df["访客数"].clip(lower=1)
        calc_rate = df["支付买家数"] / uv
        diff1 = (calc_rate - df["支付转化率"]).abs() / df["支付转化率"].clip(lower=0.001)
        bad1 = diff1[diff1 > self.tolerance]
        if len(bad1) > 0:
            violations.append({
                "rule": "支付转化率_自洽",
                "description": f"支付转化率 ≠ 支付买家数/访客数 (>{self.tolerance:.0%})",
                "bad_count": int(len(bad1)),
                "total_count": int(len(df)),
                "max_diff": round(float(bad1.max()), 4),
                "severity": "warning",
            })

        # 规则 2: 客单价 ≈ 支付金额 / 支付订单数 (日×品类, 先去重)
        td = df[["日期", "品类"] + [c for c in TRADE_COLS if c in df.columns]].drop_duplicates().copy()
        if "支付订单数" in td.columns and "客单价" in td.columns:
            calc_avg = td["支付金额"] / td["支付订单数"].clip(lower=1)
            diff2 = (calc_avg - td["客单价"]).abs() / td["客单价"].clip(lower=1)
            bad2 = diff2[diff2 > self.tolerance]
            if len(bad2) > 0:
                violations.append({
                    "rule": "客单价_自洽",
                    "description": f"客单价 ≠ 支付金额/支付订单数 (>{self.tolerance:.0%})",
                    "bad_count": int(len(bad2)),
                    "total_count": int(len(td)),
                    "max_diff": round(float(bad2.max()), 4),
                    "severity": "error",
                })

        # 规则 3: 件单价 ≈ 支付金额 / 支付件数
        if "件单价" in td.columns and "支付件数" in td.columns:
            calc_item = td["支付金额"] / td["支付件数"].clip(lower=1)
            diff3 = (calc_item - td["件单价"]).abs() / td["件单价"].clip(lower=1)
            bad3 = diff3[diff3 > self.tolerance]
            if len(bad3) > 0:
                violations.append({
                    "rule": "件单价_自洽",
                    "description": f"件单价 ≠ 支付金额/支付件数 (>{self.tolerance:.0%})",
                    "bad_count": int(len(bad3)),
                    "total_count": int(len(td)),
                    "max_diff": round(float(bad3.max()), 4),
                    "severity": "error",
                })

        # 规则 4: 交易指标在同一天同一品类下应跨来源一致
        trade_cols_present = [c for c in TRADE_COLS if c in df.columns]
        if trade_cols_present:
            grp = df.groupby(["日期", "品类"])[trade_cols_present].nunique()
            not_unique = int((grp > 1).any(axis=1).sum())
            if not_unique > 0:
                violations.append({
                    "rule": "交易指标_跨来源一致",
                    "description": "同日同品类不同流量来源的交易指标不一致 (R01 应为日×品类粒度)",
                    "bad_count": not_unique,
                    "total_count": int(len(grp)),
                    "severity": "error",
                })

        # 规则 5: 数据范围合理性
        neg_cols = []
        for c in ["访客数", "支付买家数", "支付订单数", "支付金额"]:
            if c in df.columns and (df[c] < 0).any():
                neg_cols.append(c)
        if neg_cols:
            violations.append({
                "rule": "数据范围",
                "description": f"负值列: {neg_cols}",
                "severity": "critical",
            })

        status = "pass" if not violations else (
            "fail" if any(v["severity"] in ("error", "critical") for v in violations)
            else "warning"
        )
        return {
            "status": status,
            "violations": violations,
            "shape": list(df.shape),
            "date_range": [str(df["日期"].min()), str(df["日期"].max())],
            "categories": sorted(df["品类"].unique().tolist()),
            "sources": sorted(df["流量来源"].unique().tolist()),
        }

    def check_supplementary(self, supp_dir: str) -> dict:
        """补充数据基本完整性"""
        violations = []
        if not os.path.isdir(supp_dir):
            return {"status": "skip", "violations": []}

        for csv_file in Path(supp_dir).glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                violations.append({
                    "rule": "read_error",
                    "description": f"{csv_file.name} 读取失败: {e}",
                    "severity": "error",
                })
                continue
            if len(df) == 0:
                violations.append({
                    "rule": "空文件",
                    "description": f"{csv_file.name} 为空",
                    "severity": "error",
                })
                continue
            all_nan_cols = [c for c in df.columns if df[c].isna().all()]
            if all_nan_cols:
                violations.append({
                    "rule": "全NaN列",
                    "description": f"{csv_file.name} 全 NaN 列: {all_nan_cols}",
                    "severity": "warning",
                })

        status = "pass" if not violations else (
            "fail" if any(v["severity"] in ("error", "critical") for v in violations)
            else "warning"
        )
        return {"status": status, "violations": violations}


def main():
    parser = argparse.ArgumentParser(description="v3 单平台单表数据一致性校验")
    parser.add_argument("--data-dir", default="datasets", help="数据目录")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="自洽公式允许的相对误差")
    args = parser.parse_args()

    checker = ConsistencyCheckerV3(tolerance=args.tolerance)
    data_dir = Path(args.data_dir)

    print(f"\n{'=' * 60}")
    print(f"  单平台单表数据一致性校验 v3")
    print(f"  数据目录: {data_dir}")
    print(f"  主文件: {PRIMARY_FILE}")
    print(f"  容差: {args.tolerance:.0%}")
    print(f"{'=' * 60}\n")

    all_pass = True
    case_ids = sorted([d.name for d in data_dir.iterdir()
                       if d.is_dir() and d.name.startswith("C")])

    for cid in case_ids:
        case_dir = str(data_dir / cid)
        result = checker.check_case(case_dir)

        icon = {"pass": "✅", "warning": "⚠️", "fail": "❌", "skip": "⏭️"}[result["status"]]
        shape_str = f"{result.get('shape', ['?','?'])}"
        print(f"  {icon} {cid}: {result['status']}  ({shape_str[1:-1]})")

        for v in result.get("violations", []):
            sev_icon = {"warning": "⚠️", "error": "❌", "critical": "🔴"}.get(v["severity"], "?")
            print(f"     {sev_icon} {v['description']}")
            if "bad_count" in v and "total_count" in v:
                print(f"        {v['bad_count']}/{v['total_count']} 行, "
                      f"最大差异: {v.get('max_diff', '?')}")

        supp_dir = str(data_dir / cid / "supplementary")
        if os.path.isdir(supp_dir):
            supp_result = checker.check_supplementary(supp_dir)
            if supp_result["status"] != "skip":
                s_icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}[supp_result["status"]]
                print(f"     📦 补充数据: {s_icon}")
                for v in supp_result.get("violations", []):
                    print(f"        {v['description']}")

        if result["status"] == "fail":
            all_pass = False

    print(f"\n{'=' * 60}")
    if all_pass:
        print(f"  ✅ 所有 case 通过一致性校验")
    else:
        print(f"  ❌ 部分 case 存在一致性问题")
    print(f"{'=' * 60}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
