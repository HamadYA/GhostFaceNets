#!/usr/bin/env bash

DEVKIT="devkit/experiments"
ALGO="insightface" #ms1mv2
ROOT="/home/ai-04/Downloads/megaface-testsuite/megaface/"

echo $LD_LIBRARY_PATH
cd "$DEVKIT"
LD_LIBRARY_PATH="/usr/local/lib/opencv2.4/" python -u run_experiment.py "$ROOT/feature_out/megaface" "$ROOT/feature_out/facescrub" _"$ALGO".bin "$ROOT/results/" -s 1000000 -p ../templatelists/facescrub_features_list.json
cd -

exit 0
