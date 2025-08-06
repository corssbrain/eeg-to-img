import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests 
import open_clip  
import json  
from huggingface_hub import snapshot_download 
from typing import List, Optional, Sequence, Tuple 


def setup() -> None:  
    # Configuration        
    PROXY_URL = "http://127.0.0.1:7890"
    clip_model_type = "ViT-H-14"
    PRETRAINED_WEIGHTS = "laion2b_s32b_b79k"
    PRECISION = "fp32"
     
    os.environ["http_proxy"] = PROXY_URL
    os.environ["https_proxy"] = PROXY_URL

    """Load model and paths specified in *data_config.json*.""" 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_clip, preprocess_train, feature_extractor = (
        open_clip.create_model_and_transforms(
            clip_model_type,
            pretrained=PRETRAINED_WEIGHTS,  # pulled from local HF cache
            precision=PRECISION,
            device=device,
        )
    )

    return model_clip, preprocess_train, feature_extractor, clip_model_type


def clip_text_encoder(model_clip, preprocess_train, text: Sequence[str], device: Sequence[str]) -> torch.Tensor:
    """
    Encode a batch of text strings with CLIP and return ℓ²-normalized features.

    Parameters
    ----------
    text : Sequence[str]
        List/tuple of text prompts.
    Returns
    -------
    torch.Tensor
        Normalized text feature matrix of shape [N, D].
    """
    text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(device)  # [N, 77]
    with torch.no_grad(): 
        text_feats = model_clip.encode_text(text_inputs)
    return F.normalize(text_feats, dim=-1).detach()


def clip_image_encoder(model_clip, preprocess_train, images: Sequence[str], batch_size: int = 20, device='cuda') -> torch.Tensor:
    """
    Encode a list of image paths with CLIP and return ℓ²-normalized features.

    Parameters
    ----------
    images : Sequence[str]
        Paths to RGB images on disk.
    batch_size : int, default 20
        Mini-batch size for GPU-friendly inference.

    Returns
    -------
    torch.Tensor
        Normalized image feature matrix of shape [N, D].
    """
    feats: List[torch.Tensor] = [] 
    for i in range(0, len(images), batch_size):
        paths_batch = images[i : i + batch_size]
        imgs_batch = [
            preprocess_train(Image.open(p).convert("RGB")) for p in paths_batch
        ]
        img_inputs = torch.stack(imgs_batch).to(device)

        with torch.no_grad():
            batch_feats = model_clip.encode_image(img_inputs)
            batch_feats = F.normalize(batch_feats, dim=-1)

        feats.append(batch_feats)

    return torch.cat(feats, dim=0)

 
def clip_text_image_encoder(
    text: Sequence[str],
    img: Sequence[str], 
    pictures = None,
    classes = None,
    train: bool = True,
    device = 'cuda', 
    clip_image_text_embedding='',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (or load) CLIP features for `text` and `img`.

    A cache file is used only when **both** `classes` and `pictures` are *None*.
    """
    
    model_clip, preprocess_train, feature_extractor, clip_model_type = setup()

    cache_file = (
        f"{clip_model_type}_features_train.pt"
        if train
        else f"{clip_model_type}_features_test.pt"
    )

    use_cache = classes is None and pictures is None and os.path.exists(cache_file)

    if use_cache:
        saved = torch.load(cache_file, weights_only=False)
        text_feats = saved["text_features"]
        img_feats = saved["img_features"]
    else: 
        text_feats = clip_text_encoder(model_clip, preprocess_train, text, device=device)
        img_feats = clip_image_encoder(model_clip, preprocess_train, img, device=device)

        if classes is None and pictures is None:  # cache only the generic split
            torch.save(
                {"text_features": text_feats.cpu(), "img_features": img_feats.cpu()},
                clip_image_text_embedding+cache_file,
            )

    return text_feats, img_feats
 

# if __name__ == "__main__": 
     
    # test_dataset = EEGDataset(subjects=["sub-01"], train=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
 
    # # ... or iterate through the DataLoader:
    # for batch_idx, batch in enumerate(test_loader):
    #     if batch_idx == 110:
    #         sample = batch
    #         break

    # (
    #     x_eeg,
    #     y_label,
    #     x_text,
    #     x_text_feat,
    #     x_img_feat,
    #     img_path,

    # ) = sample

    # import pdb;pdb.set_trace()

 