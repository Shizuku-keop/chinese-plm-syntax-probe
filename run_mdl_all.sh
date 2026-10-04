#!/usr/bin/env bash
# 全部模型的 MDL 扫描（bert-base-chinese 已完成的层会断点续跑跳过）。
# 等 CPU 空闲后启动：bash run_mdl_all.sh
set -u
cd "$(dirname "$0")"
for m in bert-base-chinese chinese-macbert-base chinese-roberta-wwm-ext; do
  echo "=== $(date '+%F %T') 开始 $m ===" >> "results/mdl_${m}.log"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python src/run_mdl.py --model "$m" \
    >> "results/mdl_${m}.log" 2>&1
  echo "=== $(date '+%F %T') 结束 $m (exit=$?) ===" >> "results/mdl_${m}.log"
done
for m in bert-base-chinese chinese-macbert-base chinese-roberta-wwm-ext; do
  echo "ALL_MDL_DONE $(date '+%F %T')" >> "results/mdl_${m}.log"
done
echo "ALL_MDL_DONE"
