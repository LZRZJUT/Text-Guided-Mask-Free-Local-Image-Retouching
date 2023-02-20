import clip
import glob
import torch
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


device = "cuda" if torch.cuda.is_available() else "cpu"

def ClipRank(text, images):
    
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_inputs=[]
    text = clip.tokenize([text]).to(device)

    image_paths = sorted(glob.glob(images + '/*.jpg'))
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


if __name__ == "__main__":
    path = 'for_single_test'
    psnr_rank = PsnrRank('COCO_test2014_000000369665.jpg', path)
    print(sorted(psnr_rank.items(), key = lambda kv:(kv[1], kv[0])))    
    clip_Rank = ClipRank("There are beautiful cherry trees under the snow-covered Mount Fuji", path)
    #print(sorted(clip_Rank.items(), key = lambda kv:(kv[1], kv[0]))) 
    dict_combine = {}
    for key, value in clip_Rank.items():
            dict_combine.update({key: clip_Rank.get(key) * psnr_rank.get(key)})   
    print(sorted(dict_combine.items(), key = lambda kv:(kv[1], kv[0]))) 