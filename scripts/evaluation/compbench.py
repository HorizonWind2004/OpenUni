"""
OpenUni T2I CompBench inference script.

Reads CompBench-style metadata JSON (e.g., SRUM/val_comp.json), generates one
image per prompt using OpenUni, and writes images directly under the output
directory with filename "<Question_id>.png". Additionally, it writes an
output.jsonl mapping image_file -> original_prompt for downstream VLM scoring.

Usage example (multi-GPU with accelerate):
  accelerate launch --num_processes 4 scripts/evaluation/compbench_infer.py \
    configs/openuni_7b.py \
    --checkpoint work_dirs/exp1/iter_5000.pth \
    --batch_size 4 \
    --data /home/jixie/SRUM/val_comp.json \
    --output compbench_openuni_results \
    --cfg_scale 4.5 \
    --num_steps 20 \
    --height 512 --width 512 \
    --seed 42
"""

import json
import os
import copy
import torch
import argparse
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from einops import rearrange


class CompBenchDataset(Dataset):
    def __init__(self, data_path):
        """Load CompBench metadata JSON."""
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        # Expect list of objects with keys: Question, Category, Question_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data[idx])
        text = data_dict.get('Question', '')
        qid = data_dict.get('Question_id', idx)
        category = data_dict.get('Category', 'unknown')
        
        data_dict.update(idx=idx, text=text, qid=qid, category=category)
        return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int, help='Images generated per forward pass')
    parser.add_argument('--data', default='/home/jixie/SRUM/val_comp.json', type=str)
    parser.add_argument('--output', default='compbench_openuni_results', type=str)
    parser.add_argument("--cfg_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--overwrite", action='store_true', help='Overwrite existing images')

    args = parser.parse_args()

    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)

    config = Config.fromfile(args.config)

    print(f'Device: {accelerator.device}', flush=True)

    dataset = CompBenchDataset(data_path=args.data)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x
    )

    model = BUILDER.build(config.model)
    if args.base is not None:
        print(f"Load base checkpoint: {args.base}", flush=True)
        state_dict = guess_load_checkpoint(args.base)
        info = model.load_state_dict(state_dict, strict=False)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=accelerator.device)
    model = model.to(model.dtype)
    model.eval()

    dataloader = accelerator.prepare(dataloader)

    print(f'Number of batches: {len(dataloader)}', flush=True)

    if args.cfg_prompt is None:
        cfg_prompt = model.prompt_template['CFG']
    else:
        cfg_prompt = model.prompt_template['GENERATION'].format(input=args.cfg_prompt.strip())
    cfg_prompt = model.prompt_template['INSTRUCTION'].format(input=cfg_prompt)
    if model.prompt_template.get('IMG_START_TOKEN_FOR_GENERATION', True):
        cfg_prompt += model.prompt_template['IMG_START_TOKEN']

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)
        jsonl_path = os.path.join(args.output, 'output.jsonl')
        jsonl_fp = open(jsonl_path, 'a', encoding='utf-8')
    else:
        jsonl_fp = None

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        prompts = []
        qids = []
        for data_sample in data_samples:
            prompt = copy.deepcopy(data_sample['text'].strip())
            prompt = model.prompt_template['GENERATION'].format(input=prompt)
            prompt = model.prompt_template['INSTRUCTION'].format(input=prompt)
            if model.prompt_template.get('IMG_START_TOKEN_FOR_GENERATION', True):
                prompt += model.prompt_template['IMG_START_TOKEN']
            prompts.append(prompt)
            qids.append(data_sample['qid'])

        # Check if all images in this batch already exist
        all_exist = True
        if not args.overwrite:
            for qid in qids:
                filename = f"{qid}.png"
                out_path = os.path.join(args.output, filename)
                if not os.path.exists(out_path):
                    all_exist = False
                    break
        else:
            all_exist = False

        if all_exist:
            accelerator.print(f"Batch {batch_idx}: all images exist, skipping generation")
            # Still write JSONL mappings if main process
            if accelerator.is_main_process:
                for data_sample in data_samples:
                    qid = data_sample['qid']
                    filename = f"{qid}.png"
                    record = {
                        'image_file': filename,
                        'original_prompt': data_sample['text'],
                    }
                    jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_fp.flush()
            continue

        # Prepare input for CFG: [cond_prompts, uncond_prompts]
        prompts = prompts + len(prompts) * [cfg_prompt]

        inputs = model.tokenizer(
            prompts, add_special_tokens=True, return_tensors='pt', padding=True
        ).to(accelerator.device)

        images = model.generate(
            **inputs,
            progress_bar=False,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            generator=generator,
            height=args.height,
            width=args.width
        )

        # images shape: (batch_size, C, H, W)
        images = torch.clamp(127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        images = rearrange(images, 'b c h w -> b h w c')

        # Save images only on main process
        if accelerator.is_main_process:
            for image, data_sample in zip(images, data_samples):
                qid = data_sample['qid']
                filename = f"{qid}.png"
                out_path = os.path.join(args.output, filename)
                
                Image.fromarray(image).save(out_path)

                # Write mapping to JSONL
                record = {
                    'image_file': filename,
                    'original_prompt': data_sample['text'],
                }
                jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_fp.flush()

    if accelerator.is_main_process and jsonl_fp is not None:
        jsonl_fp.close()

    accelerator.print("Done CompBench inference for OpenUni.")
