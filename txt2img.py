import argparse
from PIL import Image
import numpy as np
import os
import torch
from torch import autocast
import cv2
from matplotlib import pyplot as plt
from inpainting import StableDiffusionInpaintingPipeline
from torchvision import transforms
import clip
import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.segmask import get_seg_mask
from utils.GenerateEntity import get_Entity_with_mask
from utils.EntitywithCLIP import save_mask

auth_token = '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'

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintingPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=auth_token,
).to(device)



def predict(image_path, mask_path, prompt, outdir, nums, H, W):
    with autocast("cuda"):
        init_image = Image.open(image_path).convert("RGB").resize((H, W))
        mask = Image.open(mask_path).convert("RGB").resize((H, W))
        for i in tqdm(range(nums)):
            images = pipe(prompt = prompt, init_image=init_image, mask_image=mask, strength=0.8)["sample"]
            images[0].save(outdir + '{}.jpg'.format(str(i)))
    #return images[0]


'''
image_rank with clip
Calculate similarity
return dict{image(str): score(float)}
'''
def ClipRank(text, images):
    
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_inputs=[]
    text = clip.tokenize([text]).to(device)

    image_paths = sorted(glob.glob(images + '*.jpg'))
    #print(image_paths)
    for image_path in image_paths:
        img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_inputs.append(img)

    img = torch.concat(image_inputs, dim=0)
    with torch.no_grad():
        #image_features = model.encode_image(torch.stack(image_inputs))
        logits_per_image, logits_per_text = model(img, text)
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()
    dict_cilp = {}
    for img, score in zip(image_paths, probs[0]):
        dict_cilp[img] = score
    return dict_cilp


def PsnrRank(oimg, images):
    image_paths = glob.glob(images + '/*.jpg')
    dict_psnr = {}
    rimg = Image.open(oimg).convert('RGB').resize((512, 512))
    for fimg_path in image_paths:
        fimg = Image.open(fimg_path).convert('RGB').resize((512, 512))
        psnr = compare_psnr(np.array(rimg), np.array(fimg))
        dict_psnr[fimg_path] = psnr

    return dict_psnr


def load_flist(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            # print(np.genfromtxt(flist, dtype=np.str))
            # return np.genfromtxt(flist, dtype=np.str)
            try:
                return np.genfromtxt(flist, dtype=str)
            except:
                return [flist]
    return []

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a wooden table",
        help="rice"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="for_single_test/"
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--image_prompt",
        type=str,
        help="image to prompt with, must specify a mask",
        default='/home/lzr/projects/stable-diffusion-inpainting/COCO_test2014_000000302981.jpg'
    )
    parser.add_argument(
        "--mask_prompt",
        type=str,
        help="mask to prompt with, must specify image prompt",
        default='/home/lzr/projects/stable-diffusion-inpainting/mask/combine_mask.jpg'
    )

    parser.add_argument(
        "--nums",
        type=int,
        help="the number of generated images",
        default=16,
    )

    parser.add_argument(
        "--CLIP_threshold",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--entity_file",
        type=str,
        default='/home/lzr/projects/stable-diffusion-inpainting/entity/',
    )

    parser.add_argument(
        "--mask_file",
        type=str,
        default='/home/lzr/projects/stable-diffusion-inpainting/mask/',
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='image', # background , image
    )

    opt = parser.parse_args()


    # 测试单张图片代码
    # 应包含mask, image, prompt, outdir and nums for results
    descriptions_clip = 'a picture of a yellow luggage'
    get_seg_mask('COCO_test2014_000000302981.jpg', 'mask')
    name = get_Entity_with_mask('COCO_test2014_000000302981.jpg', opt.mask_file, opt.entity_file,)
    save_mask(descriptions_clip, opt.entity_file, opt.CLIP_threshold, name, opt.mask_file, opt.mode)
    predict(opt.image_prompt, opt.mask_prompt, opt.prompt, opt.outdir, opt.nums, opt.H, opt.W)
