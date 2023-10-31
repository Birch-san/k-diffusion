import accelerate
from accelerate.state import AcceleratorState
import argparse
from pathlib import Path
import torch
from torch import distributed as dist, multiprocessing as mp, Tensor, FloatTensor
from torch.utils import data
from torchvision import transforms
from typing import Dict, Optional, Literal, Callable
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds
from os import makedirs
from contextlib import nullcontext

class Dataset(torch.utils.data.Dataset):

    def __init__(self):

        self.data = []

        paths = list(Path("/fsx/home-tmabraham/datasets/train").glob('**/*.JPEG'))[:100]


        for i, filepath in enumerate(paths):
            self.data.append(str(filepath))

        self.safe_image = Image.open(self.data[0]).convert("RGB")
        self.safe_image = transforms.ToTensor()(self.safe_image)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.data[idx]).convert("RGB")
            image = transforms.ToTensor()(image)
        except Exception as e:
            print("Fail to read image", self.data[idx])
            print(e)
            exit()


        return {"image":image, "path":self.data[idx]}

    def __len__(self):
        return len(self.data)

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

train_transforms = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.Normalize([0.5], [0.5]),
    ]
)
def preprocess_train(images):
    images = [train_transforms(image) for image in images]
    return images

def collate_fn(data):

    images = []
    paths = []
    for entry in data:

        image = entry["image"]
        if type(image) == int:
            if image == -1:
                continue
        images.append(image)
        paths.append(entry["path"])

    images = preprocess_train(images)
    images = torch.stack(images)
    return {"images": images, "paths": paths}

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=4,
                   help='the batch size')
    p.add_argument('--mixed-precision', type=str,
                   choices=['no', 'fp16', 'bf16', 'fp8'],
                   help='the mixed precision type')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--inference-out-wds-root', type=str, default="./shards",
                   help='[in inference-only mode] directory into which to output WDS .tar files')
    p.add_argument('--inference-out-wds-shard', type=int, default=0,
                   help="[in inference-only mode] the directory within the WDS dataset .tar to place each sample. this enables you to prevent key clashes if you were to tell multiple nodes to independently produce .tars and collect them together into a single dataset afterward (poor man's version of multi-node support).")

    args = p.parse_args()
    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    accelerator = accelerate.Accelerator(mixed_precision=args.mixed_precision)
    ensure_distributed()
    device = accelerator.device
    # device = "cuda"

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])


    images = Dataset()
    train_loader = DataLoader(images, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn,
                              pin_memory = True, num_workers = args.num_workers, drop_last= False)

    if accelerator.is_main_process:
        print("Length of dataset", len(images))

    model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    model.enable_xformers_memory_efficient_attention()
    model = torch.compile(model, fullgraph = True, mode = "max-autotune")
    model.to(device)


    train_loader = accelerator.prepare(train_loader)

    # if accelerator.is_main_process:
    from webdataset import ShardWriter
    makedirs(args.inference_out_wds_root, exist_ok=True)
    print(f'process {AcceleratorState().process_index}')
    print(f'num AcceleratorState().num_processes')
    sink_ctx = ShardWriter(f'{args.inference_out_wds_root}/%05d.tar', maxcount=10000)
    def sink_sample(sink: ShardWriter, ix: int, image, key) -> None:
        out: SinkOutput = {
            '__key__': f'{AcceleratorState().process_index}/{ix}',
            'img.pth': image,
            'txt': key
        }
        sink.write(out)
#else:
    #    sink_ctx = nullcontext()
    #    sink_sample: Callable[[Optional[ShardWriter], int, object], None] = lambda *_: ...

    count = 0
    for epoch in range(0,1):
        for batch in tqdm(train_loader):

            images = batch["images"].to(device)
            with torch.cuda.amp.autocast(), torch.no_grad():

                latents = model.encode(images.to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * model.config.scaling_factor

            #We set image_key and class_cond_key in the CONFIG
            #What we are storing is a random key, and the sample is a dict with two items
            #That's image_key, image and class_cond_key, label. For us, the label needs to be extracted from the path
            for ix, sample in enumerate(latents):
                class_key = batch["paths"][ix].split("/")[-2] #This should be the name of the folder, which is just a random ID for us
                sink_sample(sink_ctx, count, sample, class_key)
                count += 1
    sink_ctx.close()
    print("Done")

if __name__ == '__main__':
    main()