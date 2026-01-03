I will help you restore and improve your scripts.

### 1. Update `inference.sh`
I will modify `inference.sh` to accept experiment names as arguments and automatically process corresponding checkpoints.
- **Dynamic Arguments**: Use a loop `for EXP_NAME in "$@"` to handle multiple experiments (e.g., `bash inference.sh exp_1 exp_2`).
- **Config Detection**: Automatically use the config file at `configs/finetune/${EXP_NAME}.py`. This ensures the correct model settings (like the `l` variant used in `des_360prompt`) are used, and avoids the need to manually specify a base model path since it's defined in the finetune config.
- **Checkpoint Discovery**: Automatically find all `iter_*.pth` checkpoints in `work_dirs/${EXP_NAME}/` instead of using a hardcoded list.
- **Output Management**: Save inference results to `work_dirs/${EXP_NAME}/inference/iter_${ITER}` to keep things organized.

### 2. Create `scripts/train_queue.sh`
I will create a new script `scripts/train_queue.sh` to implement the training queue.
- **Sequential Execution**: Iterate through the provided experiment names.
- **Training**: Call `bash scripts/train_ddp.sh configs/finetune/${EXP_NAME}.py` for each experiment.
- **Automatic Inference**: Immediately after training completes, call `bash inference.sh ${EXP_NAME}` to run inference on the newly trained checkpoints.

### Verification
- I will verify the scripts by checking their syntax and logic.
- Since I cannot run full training/inference (due to resource/time constraints), I will dry-run the logic or check file paths to ensure they match the project structure.
