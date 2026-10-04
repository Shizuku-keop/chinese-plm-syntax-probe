#!/usr/bin/env bash
# 多种子鲁棒性：三个中文单语模型 × seed 1,2，各跑主线探针 + 深度控制 + MDL。
# 逐层断点续跑（run_first_probe/run_control/run_mdl 都按层写检查点），
# 输出为 results/*_seed<N>.json，**不覆盖** seed 0 的主结果。最后自动汇总。
set -u
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
MODELS="bert-base-chinese chinese-roberta-wwm-ext chinese-macbert-base"
bak() { cp results/*.json /d/paper_backup/results_json/ 2>/dev/null; }
stage() { echo "=== $* | $(date '+%m-%d %H:%M:%S') | C盘可用 $(df -h /c | awk 'NR==2{print $4}')"; }

for s in 1 2; do
  for m in $MODELS; do
    stage "seed=$s 主线探针：$m"
    $PY src/run_first_probe.py --model "$m" --seed "$s" --tag "_seed$s" \
      >> "results/first_probe_${m}_seed${s}.log" 2>&1 || echo "WARN: $m seed$s 主线探针失败"
    stage "seed=$s 深度控制：$m"
    $PY src/run_control.py --control depth --model "$m" --seed "$s" --tag "_seed$s" \
      >> "results/control_depth_${m}_seed${s}.log" 2>&1 || echo "WARN: $m seed$s 深度控制失败"
    stage "seed=$s MDL：$m"
    $PY src/run_mdl.py --model "$m" --seed "$s" --tag "_seed$s" \
      >> "results/mdl_${m}_seed${s}.log" 2>&1 || echo "WARN: $m seed$s MDL 失败"
    bak
  done
done

stage "多种子汇总"
$PY src/summarize_seeds.py || echo "WARN: 多种子汇总失败"
bak

echo "ALL_SEEDS_DONE"
