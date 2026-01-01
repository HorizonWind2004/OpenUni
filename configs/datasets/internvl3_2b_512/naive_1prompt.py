from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import collate_func_img2img
from src.datasets.image2image.edit_datasets import FolderReconstructionDataset


with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index


dataset = dict(type=FolderReconstructionDataset,
               data_path='/mnt/hdfs/jixie/flux_img',
               image_size=image_size,
               prompt_template=prompt_template,
               tokenizer=tokenizer)

train_dataloader = dict(
    batch_size=30,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_img2img,
                    pad_index=pad_index)
)
