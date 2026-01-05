#!/bin/bash

export PYTHONPATH=.

for EXP_NAME in "$@"; do
    echo "=================================================="
    echo "Processing experiment: $EXP_NAME"
    echo "=================================================="
    
    CONFIG="configs/finetune/${EXP_NAME}.py"
    if [ ! -f "$CONFIG" ]; then
        echo "Error: Config file $CONFIG not found!"
        continue
    fi

    WORK_DIR="work_dirs/${EXP_NAME}"
    if [ ! -d "$WORK_DIR" ]; then
        echo "Error: Work directory $WORK_DIR not found!"
        continue
    fi

    shopt -s nullglob
    CHECKPOINTS=("$WORK_DIR"/iter_*.pth)
    shopt -u nullglob

    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "No checkpoints found in $WORK_DIR"
        continue
    fi

    for CKPT_PATH in "${CHECKPOINTS[@]}"; do
        FILENAME=$(basename "$CKPT_PATH")
        ITER=${FILENAME#iter_}
        ITER=${ITER%.pth}
        # mkdir /mnt/hdfs/jixie/old/${EXP_NAME}
        mkdir -p "/mnt/hdfs/jixie/old/${EXP_NAME}"
        OUTPUT_DIR="/mnt/hdfs/jixie/old/${EXP_NAME}/${EXP_NAME}_${ITER}"
        
        echo "--------------------------------------------------"
        echo "Running inference for checkpoint: $CKPT_PATH"
        echo "Output directory: $OUTPUT_DIR"
        
        mkdir -p "$OUTPUT_DIR"
        
        accelerate launch --main_process_port 12333 scripts/evaluation/gen_eval.py "$CONFIG" \
            --checkpoint "$CKPT_PATH" \
            --batch_size 4 \
            --output "$OUTPUT_DIR" \
            --height 512 --width 512 \
            --seed 42
            
    done
done
