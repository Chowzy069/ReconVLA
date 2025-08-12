#!/bin/bash

set -e
trap "kill 0" EXIT

timestamp=$(date +%m%d_%H%M)

### ====== public port list (server & client consistent)======
PORTSLIST=(9097 9098)  # you can add multiple ports, like (9077 9078 9079)

### ====== Step 1: start multi-port Server ======
echo "[INFO] launching flask_server.py..."
# set project path
export PYTHONPATH=/reconvla/reconvla:$PYTHONPATH
# set gpu list
export CUDA_VISIBLE_DEVICES=0,1
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#PORTSLIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=${PORTSLIST[$IDX]}
    echo "[INFO] [SERVER] Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./serve/flask_server.py \
        --model-path /project/reconvla/checkpoints/checkpoint \
        --action_stat ./statistics.yaml \
        --port $port \
        --double_instruction True \
        &
done



wait