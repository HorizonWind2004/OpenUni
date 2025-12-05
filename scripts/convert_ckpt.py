# coding=utf-8
# Copyright 2024 OpenUni Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import numpy as np
import torch
import argparse
from xtuner.model.utils import guess_load_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OpenUni checkpoint format")
    parser.add_argument('--config', help='config file path.', default='configs/models/openuni_config.py')
    parser.add_argument("--checkpoint", type=str, default=None, required=True, help='input checkpoint path')
    parser.add_argument('--output', help='output file path.', default='openuni_converted.pth')
    parser.add_argument('--format', choices=['safetensors', 'pytorch'], default='pytorch', 
                        help='output format (safetensors or pytorch)')

    args = parser.parse_args()
    
    if args.checkpoint is None:
        print("Error: --checkpoint argument is required")
        exit(1)
    
    if args.checkpoint is not None:
        print(f"Load checkpoint: {args.checkpoint}", flush=True)
        if os.path.isdir(args.checkpoint):
            checkpoint = guess_load_checkpoint(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, weights_only=False)
    
    print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if args.format == 'safetensors':
        try:
            from safetensors.torch import save_file
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                save_file(checkpoint['state_dict'], args.output)
            elif isinstance(checkpoint, dict):
                save_file(checkpoint, args.output)
            else:
                print("Error: Checkpoint format not supported for safetensors")
                exit(1)
            print(f"Checkpoint saved as safetensors to: {args.output}")
        except ImportError:
            print("Error: safetensors not installed. Please install with: pip install safetensors")
            exit(1)
    else:
        torch.save(checkpoint, args.output)
        print(f"Checkpoint saved as PyTorch format to: {args.output}")
    
    print("Conversion completed successfully!")
