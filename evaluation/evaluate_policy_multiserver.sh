#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export EGL_VISIBLE_DEVICES=0
trap "kill 0" EXIT
PORTSLIST=(9097)
export PYTHONPATH=/reconvla/calvin:$PYTHONPATH
export PYTHONPATH=/reconvla/calvin/calvin_models:$PYTHONPATH
export PYTHONPATH=/reconvla/calvin/calvin_env:$PYTHONPATH
export PYTHONPATH=/reconvla/calvin/calvin/calvin_env/tacto:$PYTHONPATH
EVAL_LOG_DIR="/reconvla/calvin/calvin_models/calvin_agent/evaluation/log"
CHUNKS=${#PORTSLIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    python ./evaluate_policy_multiserver.py \
        --dataset_path /data/task_ABC_D \
        --question_file  ./question.json\
        --eval_log_dir $EVAL_LOG_DIR \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --port ${PORTSLIST[$IDX]} \
        --save_dir ./video \
        --save_name result \
        --custom_model \
        &
done


wait