#!/bin/bash

export PYTHONPATH=.

# Loop over all arguments provided (experiment names)
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

    # Enable nullglob to handle case with no matches
    shopt -s nullglob
    CHECKPOINTS=("$WORK_DIR"/iter_*.pth)
    shopt -u nullglob

    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "No checkpoints found in $WORK_DIR"
        continue
    fi

    for CKPT_PATH in "${CHECKPOINTS[@]}"; do
        # Extract iteration number from filename (e.g., iter_1000.pth)
        FILENAME=$(basename "$CKPT_PATH")
        ITER=${FILENAME#iter_}
        ITER=${ITER%.pth}
        
        # Use a structured output directory
        OUTPUT_DIR="${WORK_DIR}/inference/iter_${ITER}"
        
        echo "--------------------------------------------------"
        echo "Running inference for checkpoint: $CKPT_PATH"
        echo "Output directory: $OUTPUT_DIR"
        
        # Ensure output directory exists (gen_eval.py does this too, but good practice)
        mkdir -p "$OUTPUT_DIR"
        
        # Run inference
        # Using the finetune config allows the model to load the correct pretrained weights defined in the config.
        accelerate launch scripts/evaluation/gen_eval.py "$CONFIG" \
            --checkpoint "$CKPT_PATH" \
            --batch_size 4 \
            --output "$OUTPUT_DIR" \
            --height 512 --width 512 \
            --seed 42
            
    done
done
