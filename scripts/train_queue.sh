#!/bin/bash

set -e

export PYTHONPATH=.

for EXP_NAME in "$@"; do
    CONFIG="configs/finetune/${EXP_NAME}.py"

    if [ ! -f "$CONFIG" ]; then
        echo "Error: Config file $CONFIG not found!"
        continue
    fi

    echo "Training: $EXP_NAME"
    bash scripts/train_ddp.sh "$CONFIG"

    echo "Inference: $EXP_NAME"
    bash inference.sh "$EXP_NAME"
done
