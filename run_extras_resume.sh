#!/usr/bin/env bash
# 续跑：应用退出后重建追加实验（阶段 7 尾 → 9）。第 6 步（均值池化抽取）已完成，不再重跑。
# 各脚本按层断点续跑：已完成的层会跳过，不重算。
set -u
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
CH="bert-base-chinese chinese-roberta-wwm-ext chinese-macbert-base"
bak() { cp results/*.json /d/paper_backup/results_json/ 2>/dev/null; }
stage() { echo "=== $* | $(date '+%H:%M:%S') | C盘可用 $(df -h /c | awk 'NR==2{print $4}')"; }

stage "7/9（续）均值池化主线探针 ×3 模型"
for m in $CH; do
  $PY src/run_first_probe.py --model "${m}_mean" >> "results/first_probe_${m}_mean.log" 2>&1 || echo "WARN: $m 均值主线失败"
  bak
done

stage "8/9 均值池化深度控制 + MDL ×3 模型"
for m in $CH; do
  $PY src/run_control.py --control depth --model "${m}_mean" >> "results/control_depth_${m}_mean.log" 2>&1 || echo "WARN: $m 均值深度控制失败"
  bak
  $PY src/run_mdl.py --model "${m}_mean" >> "results/mdl_${m}_mean.log" 2>&1 || echo "WARN: $m 均值 MDL 失败"
  bak
done

stage "9/9 均值池化距离控制（BERT 抽查）"
$PY src/run_control.py --control distance --model bert-base-chinese_mean >> results/control_distance_bert-base-chinese_mean.log 2>&1 || echo "WARN: BERT 均值距离控制失败"
bak

echo "ALL_EXTRAS_DONE"
