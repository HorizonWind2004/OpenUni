import torch
import numpy as np
import io
import random
import glob
import os
from PIL import Image
from einops import rearrange
from src.datasets.utils import crop2square
from src.datasets.text2image.caption_datasets import CaptionDataset
from src.datasets.image2image.consts import get_recon_prompt_list


class ImageEditDataset(CaptionDataset):
    def _process_image(self, image):
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        return pixel_values

    def _process_text(self, text):
        prompt_template = self.prompt_template
        image_tokens = prompt_template['IMG_START_TOKEN'] + \
                       prompt_template['IMG_CONTEXT_TOKEN'] * self.image_length + \
                       prompt_template['IMG_END_TOKEN']
        prompt = f'{image_tokens}\n{text}'
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        if self.prompt_template.get('IMG_START_TOKEN_FOR_GENERATION', True):
            prompt += prompt_template['IMG_START_TOKEN']
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', **self.tokenizer_kwargs)[0]

        return dict(input_ids=input_ids)

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            source_image = self._read_image(data_sample['source_image']).convert('RGB')
            target_image = self._read_image(data_sample['target_image']).convert('RGB')
            prompt = self._read_json(data_sample['annotation'])[self.cap_source]

            pixel_values_src = self._process_image(source_image)
            pixel_values = self._process_image(target_image)

            data = self._process_text(prompt)

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder, image_file=data_sample['image'],
                type='image2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class ReconstructDataset(ImageEditDataset):
    def _process_image(self, image):
        image = crop2square(image)
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        return pixel_values

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            prompt = "Keep the image as it is."
            pixel_values = pixel_values_src = self._process_image(image)

            data = self._process_text(prompt)

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder, image_file=data_sample['image'],
                type='image2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class MidjourneyReconstructionDataset(ImageEditDataset):
    def __init__(self, 
                 data_path=None,
                 cache_dir=None,
                 max_samples=None,
                 *args, **kwargs):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.prompt_list = get_recon_prompt_list()
        self._load_data(data_path)
        super().__init__(data_path=data_path,*args, **kwargs)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            self.dataset = load_dataset(data_path, cache_dir=self.cache_dir)['train']
            print(f"Loaded {len(self.dataset)} samples from {data_path}")
            
            self.data_list = []
            for idx in range(len(self.dataset)):
                if self.max_samples and idx >= self.max_samples:
                    break
                self.data_list.append({
                    'image': idx, 
                })
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path} for reconstruction", flush=True)

    def _read_image(self, image_idx):
        try:
            sample = self.dataset[image_idx]
            image_data = sample['image']
            
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes']))
            elif hasattr(image_data, 'convert'):
                image = image_data
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                try:
                    image = Image.fromarray(np.array(image_data))
                except:
                    raise TypeError(f"Unknown type: {type(image_data)}")

            return image
        except Exception as e:
            print(f"Error reading image at index {image_idx}: {e}")
            raise

    def _process_image(self, image):
        image = crop2square(image)
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        return pixel_values

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            prompt = random.choice(self.prompt_list)
            pixel_values = pixel_values_src = self._process_image(image)

            data = self._process_text(prompt)

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder, image_file=data_sample['image'],
                type='image2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class BLIP3oReconstructionDataset(ImageEditDataset):
    def __init__(self, 
                 data_path=None,
                 cache_dir=None,
                 max_samples=None,
                 *args, **kwargs):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.prompt_list = get_recon_prompt_list()
        self._load_data(data_path)
        super().__init__(data_path=data_path,*args, **kwargs)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            data_files = glob.glob(self.data_path) 
            self.dataset = load_dataset("webdataset", data_files=data_files, cache_dir=self.cache_dir, split="train", num_proc=64)
            print(f"Loaded {len(self.dataset)} samples.")
            self.data_list = []
            for idx in range(len(self.dataset)):
                if self.max_samples and idx >= self.max_samples:
                    break
                self.data_list.append({
                    'image': idx, 
                })
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path} for reconstruction", flush=True)

    def _read_image(self, image_idx):
        try:
            sample = self.dataset[image_idx]
            image_data = sample['jpg']
            
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes']))
            elif hasattr(image_data, 'convert'):
                image = image_data
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                try:
                    image = Image.fromarray(np.array(image_data))
                except:
                    raise TypeError(f"Unknown type: {type(image_data)}")
            
            return image
        except Exception as e:
            print(f"Error reading image at index {image_idx}: {e}")
            raise

    def _process_image(self, image):
        image = crop2square(image)
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        return pixel_values

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            prompt = random.choice(self.prompt_list)
            pixel_values = pixel_values_src = self._process_image(image)

            data = self._process_text(prompt)

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder, image_file=data_sample['image'],
                type='image2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()




class BLIP3oSFTDataset(CaptionDataset):
    def __init__(self, 
                 data_path=None,
                 cache_dir=None,
                 max_samples=None,
                 *args, **kwargs):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self._load_data(data_path)
        super().__init__(data_path=data_path,*args, **kwargs)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            # print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            # self.dataset = load_dataset(data_path, cache_dir=self.cache_dir)['train']
            data_files = glob.glob(self.data_path) 
            self.dataset = load_dataset("webdataset", data_files=data_files, cache_dir=self.cache_dir, split="train", num_proc=64)
            print(f"Loaded {len(self.dataset)} samples.")
            # import pdb; pdb.set_trace()
            self.data_list = []
            for idx in range(len(self.dataset)):
                if self.max_samples and idx >= self.max_samples:
                    break
                self.data_list.append({
                    'image': idx, 
                })
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path} for reconstruction", flush=True)

    def _read_image(self, image_idx):
        try:
            sample = self.dataset[image_idx]
            image_data = sample['jpg']
            text_data = sample['txt']
            
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes']))
            elif hasattr(image_data, 'convert'):
                image = image_data
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                try:
                    image = Image.fromarray(np.array(image_data))
                except:
                    raise TypeError(f"Unknown image type: {type(image_data)}")
            
            return image, text_data
        
        except Exception as e:
            print(f"Error reading image at index {image_idx}: {e}")
            raise

    def _process_image(self, image):
        image = crop2square(image)
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        return pixel_values

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            image, prompt = self._read_image(data_sample['image'])
            pixel_values = self._process_image(image)

            data = self._process_text(prompt)

            data.update(
                pixel_values=pixel_values,
                image_dir=self.image_folder, image_file=data_sample['image'],
                type='text2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()

class Naive1PromptDataset(ReconstructDataset):
    def __init__(self, data_path, *args, **kwargs):
        self.data_path = data_path
        self._load_data(data_path)
        super().__init__(data_path=data_path, *args, **kwargs)

    def _load_data(self, data_path):
        self.data_list = []
        # Support common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
        
        for ext in extensions:
            # Recursive search
            files = glob.glob(os.path.join(data_path, ext))
            self.data_list.extend(files)
            
        self.data_list = sorted(list(set(self.data_list)))
        self.data_list = [{'image': path} for path in self.data_list]
        print(f"Loaded {len(self.data_list)} images from {data_path} for reconstruction", flush=True)

    def _read_image(self, image_path):
        return Image.open(image_path)


class FolderReconstructionDataset(Naive1PromptDataset):
    pass


class Des360PromptReconstructionDataset(Naive1PromptDataset):
    def __init__(self, data_path, *args, **kwargs):
        self.prompt_list = get_recon_prompt_list()
        super().__init__(data_path=data_path, *args, **kwargs)

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            prompt = random.choice(self.prompt_list)
            pixel_values = pixel_values_src = self._process_image(image)

            data = self._process_text(prompt)

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder, image_file=data_sample['image'],
                type='image2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()
