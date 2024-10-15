#!/bin/bash

set -e
set -x

CONFIG=$1
CKPT=$2
OUT=${CONFIG##*/}
OUT=${OUT%.*}

if [ -z $CONFIG ] || [ -z $CKPT ]; then
    echo "Usage: $0 <config> <ckpt> <out>"
    exit 1
fi

if [ ! -f work_dirs/$OUT.pkl ]; then
    python tools/test.py $CONFIG $CKPT --out work_dirs/$OUT.pkl
fi
if [ ! -f work_dirs/$OUT-train.pkl ]; then
    python tools/test.py $CONFIG $CKPT --out work_dirs/$OUT-train.pkl --dataset train
fi
python tools/analysis_tools/analyze_results.py $CONFIG work_dirs/$OUT.pkl --out-dir work_dirs/analy-${OUT}
python tools/analysis_tools/analyze_results.py $CONFIG work_dirs/$OUT-train.pkl --out-dir work_dirs/analy-${OUT}-train --dataset train
python tools/analysis_tools/confusion_matrix.py $CONFIG work_dirs/$OUT.pkl --show-path work_dirs/analy-${OUT}/confmat.png --include-values
python tools/analysis_tools/confusion_matrix.py $CONFIG work_dirs/$OUT-train.pkl --show-path work_dirs/analy-${OUT}-train/confmat.png --include-values