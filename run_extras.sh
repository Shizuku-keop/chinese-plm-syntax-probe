#!/usr/bin/env bash
# 追加实验：①mBERT 单语 vs 多语对照；②均值池化消融（三模型）。
# 串行执行，每阶段写日志 + 备份 JSON 到 D 盘。进度看 stdout 的阶段标记。
set -u
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
CH="bert-base-chinese chinese-roberta-wwm-ext chinese-macbert-base"
bak() { cp results/*.json /d/paper_backup/results_json/ 2>/dev/null; }
stage() { echo "=== $* | $(date '+%H:%M:%S') | C盘可用 $(df -h /c | awk 'NR==2{print $4}')"; }

stage "1/9 下载 mBERT"
$PY src/download_model.py AI-ModelScope/bert-base-multilingual-cased > results/download_mbert.log 2>&1 || echo "WARN: 下载失败"
tail -2 results/download_mbert.log 2>/dev/null

stage "2/9 抽取 mBERT 表示（first 池化）"
$PY src/extract.py models/bert-base-multilingual-cased --pooling first > results/extract_mbert.log 2>&1 || echo "WARN: mBERT 抽取失败"
tail -2 results/extract_mbert.log 2>/dev/null

stage "3/9 mBERT 主线探针"
$PY src/run_first_probe.py --model bert-base-multilingual-cased >> results/first_probe_mbert.log 2>&1 || echo "WARN: mBERT 主线失败"
bak

stage "4/9 mBERT 控制（depth + distance）"
for c in depth distance; do
  $PY src/run_control.py --control "$c" --model bert-base-multilingual-cased >> "results/control_${c}_mbert.log" 2>&1 || echo "WARN: mBERT $c 控制失败"
  bak
done

stage "5/9 mBERT MDL"
$PY src/run_mdl.py --model bert-base-multilingual-cased >> results/mdl_mbert.log 2>&1 || echo "WARN: mBERT MDL 失败"
bak

stage "6/9 均值池化表示抽取 ×3 模型"
for m in $CH; do
  $PY src/extract.py "models/$m" --pooling mean >> "results/extract_${m}_mean.log" 2>&1 || echo "WARN: $m 均值抽取失败"
  tail -1 "results/extract_${m}_mean.log" 2>/dev/null
done

stage "7/9 均值池化主线探针 ×3 模型"
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
