#!/usr/bin/env python3
"""
eval_e2e.py — 端到端评测运行器 (v6 / app_v9)
=============================================
一键运行 Complex 路径完整评测:
  1. 数据一致性校验 (v3 单平台单表)
  2. Step 1+2 评测 (全部 12 case, 含 analysis_depth 评测)
  3. Step 3 评测 (B 组 5 case, 需 --run-step3)
  4. 汇总报告生成

用法:
  # Mock 模式 (管线验证)
  python eval_e2e.py

  # 真实 Agent (app_v9 统一管线)
  python eval_e2e.py --agent-module app_v9

  # 含 Step 3
  python eval_e2e.py --agent-module app_v9 --run-step3

  # 指定 case
  python eval_e2e.py --agent-module app_v9 --question C01 C08

  # 重新生成数据后评测
  python eval_e2e.py --regenerate --seed 123 --agent-module app_v9
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list, desc: str) -> int:
    """运行子命令并打印结果"""
    print(f"\n{'─' * 60}")
    print(f"  🚀 {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Complex 路径端到端评测")
    parser.add_argument("--eval-dir", default=None,
                        help="评测根目录 (默认: 脚本所在目录的上级)")
    parser.add_argument("--agent-module", default=None,
                        help="Agent 模块路径")
    parser.add_argument("--question", nargs="*", default=None,
                        help="指定 case ID")
    parser.add_argument("--run-step3", action="store_true",
                        help="启用 Step 3 深度分析评测")
    parser.add_argument("--regenerate", action="store_true",
                        help="重新生成评测数据")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (仅 --regenerate 时生效)")
    parser.add_argument("--skip-consistency", action="store_true",
                        help="跳过一致性校验")
    parser.add_argument("--output-dir", default=None,
                        help="评测结果输出目录")
    parser.add_argument("--delay", type=float, default=0,
                        help="题间延迟秒数")
    args = parser.parse_args()

    # 确定路径
    script_dir = Path(__file__).parent
    eval_dir = Path(args.eval_dir) if args.eval_dir else script_dir.parent
    data_dir = eval_dir / "complex_data_v9"
    eval_json = eval_dir / "eval_cases_v5-1.json"
    output_dir = args.output_dir or str(eval_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"\n{'#' * 60}")
    print(f"  Complex 路径端到端评测 (v6 / 单平台单表 / app_v9)")
    print(f"{'#' * 60}")
    print(f"  评测目录: {eval_dir}")
    print(f"  数据目录: {data_dir}")
    print(f"  评测集:   {eval_json}")
    print(f"  Agent:    {'🤖 ' + args.agent_module if args.agent_module else '⚠️ MockAgent'}")
    print(f"  Step 3:   {'✅ 启用' if args.run_step3 else '⏭️ 跳过'}")
    print(f"  输出:     {output_dir}")

    # ── Phase 0: 数据生成 (可选) ──
    if args.regenerate:
        ret = run_cmd([
            sys.executable, str(script_dir / "generate_test_data.py"),
            "--output-dir", str(data_dir),
            "--eval-json", str(eval_json),
            "--seed", str(args.seed),
        ], "重新生成评测数据")
        if ret != 0:
            print("\n  ❌ 数据生成失败")
            sys.exit(1)

    # 检查数据是否存在
    if not eval_json.exists():
        print(f"\n  ❌ 评测集文件不存在: {eval_json}")
        print(f"  请先运行: python generate_test_data.py --output-dir {data_dir} --eval-json {eval_json}")
        sys.exit(1)

    # ── Phase 1: 一致性校验 ──
    if not args.skip_consistency:
        ret = run_cmd([
            sys.executable, str(script_dir / "consistency_check.py"),
            "--data-dir", str(data_dir),
        ], "跨表数据一致性校验")
        # 一致性校验 warning 不阻断

    # ── Phase 2: 运行评测 ──
    eval_cmd = [
        sys.executable, str(script_dir / "eval_complex_v7-3.py"),
        "--eval-file", str(eval_json),
        "--data-dir", str(data_dir),
        "--output-dir", output_dir,
    ]
    if args.agent_module:
        eval_cmd.extend(["--agent-module", args.agent_module])
    if args.question:
        eval_cmd.extend(["--question"] + args.question)
    if args.run_step3:
        eval_cmd.extend(["--run-step3", "--step3-data-dir", str(data_dir)])
    if args.delay > 0:
        eval_cmd.extend(["--delay", str(args.delay)])

    ret = run_cmd(eval_cmd, "Complex 路径评测 v6")

    # ── Phase 3: 汇总 ──
    summary_path = os.path.join(output_dir, "eval_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        o = summary.get("overall", {})
        print(f"\n{'#' * 60}")
        print(f"  📊 评测完成")
        print(f"{'#' * 60}")
        print(f"  通过率: {o.get('pass_rate', 0):.0%} ({o.get('pass_count', 0)}/{o.get('total', 0)})")
        print(f"  均分:   {o.get('avg_score', 0):.2%}")
        print(f"  A组:    {o.get('group_a_avg', 0):.2%} ({o.get('group_a_count', 0)} 题)")
        print(f"  B组:    {o.get('group_b_avg', 0):.2%} ({o.get('group_b_count', 0)} 题)")
        weakest = o.get("weakest_dimension")
        if weakest:
            print(f"  短板:   {weakest} ({o.get('weakest_dimension_score', 0):.2%})")
        print(f"\n  完整报告: {output_dir}/eval_summary.json")
        print(f"  文本报告: {output_dir}/eval_report.txt")
    else:
        print(f"\n  ⚠️ 未找到评测汇总文件: {summary_path}")

    sys.exit(ret)


if __name__ == "__main__":
    main()
