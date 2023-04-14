#!/usr/bin/env bash

ALGO="insightface"
NUM_ITERATIONS="GhostFaceNets/GhostFaceNet_W1.3_S1_ArcFace.h5"
EPOCH="26"
DATA="data/"
MODELSDIR="/home/ai-04/Downloads/megaface-testsuite/megaface/GhostFaceNets/"

ROOT="/home/ai-04/Downloads/megaface-testsuite/megaface/"

FEATUREOUT="$ROOT/feature_out/$ALGO"
FEATUREOUTCLEAN="$ROOT/feature_out_clean/$ALGO"

CUDA_VISIBLE_DEVICES=3 python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model "$MODELSDIR/$ALGO/$NUM_ITERATIONS" --megaface-data "$DATA" --output "$FEATUREOUT"
python -u remove_noises.py --algo "$ALGO" --megaface-data "$DATA" --feature-dir-input "$FEATUREOUT" --feature-dir-out "$FEATUREOUTCLEAN"

DEVKIT="/home/ai-04/Downloads/megaface-testsuite/megaface/devkit/experiments/"

cd "$DEVKIT"
export LD_LIBRARY_PATH="/usr/local/lib/opencv2.4/"

python -u run_experiment.py "$FEATUREOUTCLEAN/megaface" "$FEATUREOUTCLEAN/facescrub" _"$ALGO".bin "$ROOT/$ALGO/" -s 1000000 -p ../templatelists/facescrub_features_list.json

python -u run_experiment.py "$FEATUREOUT/megaface" "$FEATUREOUT/facescrub" _"$ALGO".bin "$ROOT/$ALGO-noisy/" -s 1000000 -p ../templatelists/facescrub_features_list.json
cd -