#!/usr/bin/env bash
# 均值池化的距离控制：三个模型串行，逐层断点续跑（已完成的层跳过）。
# 目的：把"均值池化 × 距离 selectivity"补齐到三模型，闭合论文第 5 节局限⑤。
set -u
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
CH="bert-base-chinese chinese-roberta-wwm-ext chinese-macbert-base"
bak() { cp results/*.json /d/paper_backup/results_json/ 2>/dev/null; }
stage() { echo "=== $* | $(date '+%H:%M:%S') | C盘可用 $(df -h /c | awk 'NR==2{print $4}')"; }

for m in $CH; do
  stage "距离控制（均值池化）：$m"
  $PY src/run_control.py --control distance --model "${m}_mean" >> "results/control_distance_${m}_mean.log" 2>&1 || echo "WARN: $m 均值距离控制失败"
  bak
done

stage "汇总与绘图"
$PY src/summarize.py
$PY src/plot_results.py
bak

echo "ALL_DISTANCE_MEAN_DONE"
