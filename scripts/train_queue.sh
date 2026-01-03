#!/bin/bash

export PYTHONPATH=.

# Loop over all arguments provided (experiment names)
for EXP_NAME in "$@"; do
    echo "=================================================="
    echo "Starting training for experiment: $EXP_NAME"
    echo "=================================================="

    CONFIG="configs/finetune/${EXP_NAME}.py"
    
    if [ ! -f "$CONFIG" ]; then
        echo "Error: Config file $CONFIG not found!"
        continue
    fi
    
    # Run training
    # scripts/train_ddp.sh accepts config file as first argument
    echo "Running training command: bash scripts/train_ddp.sh $CONFIG"
    bash scripts/train_ddp.sh "$CONFIG"
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo "Training failed for $EXP_NAME with exit code $TRAIN_EXIT_CODE. Skipping inference."
        continue
    fi
    
    echo "Training completed successfully for $EXP_NAME. Starting inference..."
    
    # Run inference
    # inference.sh expects experiment name(s) as arguments
    bash inference.sh "$EXP_NAME"
    
    echo "Completed processing for $EXP_NAME"
    echo ""
done
