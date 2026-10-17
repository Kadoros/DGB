#!/usr/bin/env bash
# 1-a.0 baseline + 1-a.1 encoder comparison (frozen encoder, 224 px, 3 seeds).
# Baseline "rawpatch + mlp" is a stand-in for the lab's existing MLP pipeline (not in this repo).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RES=224
run() {  # encoder head
  [ -f cache/$1_$RES/info.json ] || $PY -m scripts.cache_features --encoder $1 --res $RES
  for s in 0 1 2; do
    [ -f runs/$1_${RES}_$2_s$s/metrics.json ] || $PY -m scripts.train_probe --encoder $1 --head $2 --res $RES --seed $s
  done
}
run rawpatch mlp
for e in resnet50 dinov2_b14 endovit_b16 endofm_b16; do
  run $e linear
  [ "${KEEP_CACHE:-0}" = 1 ] || rm -f cache/${e}_$RES/feats.npy   # ~3-7 GB each
done
$PY -m scripts.summarize --out results/1a1_encoders.md
