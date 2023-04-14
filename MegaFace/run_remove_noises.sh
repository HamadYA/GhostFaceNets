#!/usr/bin/env bash

ALGO="insightface"
NUM_ITERATIONS="GhostFaceNets/GhostFaceNet_W1.3_S1_ArcFace.h5"
EPOCH="26"
DATA="data/"
MODELSDIR="/home/ai-04/Downloads/megaface-testsuite/megaface/GhostFaceNets/"

ROOT="/home/ai-04/Downloads/megaface-testsuite/megaface/"

# FEATUREOUT="$ROOT/feature_out/$ALGO-$EPOCH"
# FEATUREOUTCLEAN="$ROOT/feature_out_clean/$ALGO-$EPOCH"

FEATUREOUT="$ROOT/feature_out/$ALGO"
FEATUREOUTCLEAN="$ROOT/feature_out_clean/$ALGO"

python -u remove_noises.py --algo "$ALGO" --megaface-data "$DATA" --feature-dir-input "$FEATUREOUT" --feature-dir-out "$FEATUREOUTCLEAN"