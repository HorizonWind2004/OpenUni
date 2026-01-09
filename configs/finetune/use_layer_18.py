from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from xtuner.engine.runner import TrainLoop

from src.datasets.collate_functions import collate_func_img2img
from src.datasets.image2image.edit_datasets import Des360PromptReconstructionDataset
from src.optimisers.custom_adamw import CustomAdamW


with read_base():
    from ..models.openuni_l_internvl3_2b_sana_1_6b_512_hf import model
    from ..datasets.internvl3_2b_512.processors import (image_size, pad_index,
                                                        prompt_template,
                                                        tokenizer)


dataset = dict(type=Des360PromptReconstructionDataset,
               data_path='/mnt/hdfs/jixie/flux_img',
               image_size=image_size,
               prompt_template=prompt_template,
               tokenizer=tokenizer)

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_img2img, pad_index=pad_index))


model.num_queries = 256
model.use_activation_checkpointing = False
model.freeze_transformer = True
model.pretrained_pth = 'checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.pth'
model.limit_image_attention_layers = None
model.use_certain_layer = [18]


accumulative_counts = 4
max_iters = 5000
lr = 1e-6
betas = (0.9, 0.95)
weight_decay = 0.05
max_norm = 1.0
warmup_ratio = 0.01

save_steps = 1000
save_total_limit = 10


optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=CustomAdamW, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)


param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=warmup_ratio * max_iters),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=False,
        begin=warmup_ratio * max_iters,
        end=max_iters)
]


train_cfg = dict(type=TrainLoop, max_iters=max_iters)


default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=True)
log_processor = dict(by_epoch=False)
