"""
This code embeds the Things dataset’s images and captions of the EEG recording
stimulations using the CLIP model.
"""

import os
from typing import List, Sequence, Tuple

import clip
import open_clip
import torch
from PIL import Image
from torch.nn import functional as F


def setup(clip_model_type="ViT-H-14", proxy_url = "http://127.0.0.1:7890", pretrained_weights = "laion2b_s32b_b79k", precision = "fp32") -> None:  
    
    # Configuration         
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url

    """Load model and paths specified in *data_config.json*."""  
    model_clip, preprocess_train, feature_extractor = (
        open_clip.create_model_and_transforms(
            clip_model_type,
            pretrained=pretrained_weights,  # pulled from local HF cache
            precision=precision,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
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
    clip_model_type="ViT-H-14", 
    proxy_url = "http://127.0.0.1:7890", 
    pretrained_weights = "laion2b_s32b_b79k", 
    precision = "fp32"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (or load) CLIP features for `text` and `img`.

    A cache file is used only when **both** `classes` and `pictures` are *None*.
    """
    
    model_clip, preprocess_train, feature_extractor, clip_model_type = setup(
        clip_model_type=clip_model_type, 
        proxy_url = proxy_url, 
        pretrained_weights=pretrained_weights, 
        precision = precision
    )

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
