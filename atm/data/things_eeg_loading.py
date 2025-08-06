"""
The aim of this code is to load raw EEG dataset of Things dataset.
"""

import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
 
def load_image_text_eeg(
    subjects,
    train: bool,
    classes=None,
    exclude_subject=None,
    pictures=None, 
    img_directory_training='',
    img_directory_test='',
):
    """
    Load EEG data and associated text/image prompts.

    Parameters
    ----------
    subjects
        Iterable of subject identifiers (directory names under *data_path*).
    train
        ``True`` to load training data, ``False`` for test data.
    classes
        Indices of visual classes to include (``None`` → include all).
    exclude_subject
        Subject identifier to skip (training-only).
    pictures
        Specific picture indices per class (parallel to *classes*). ``None`` to
        include all images within selected classes. 
    img_directory_training
        Root directory with training images, one folder per class.
    img_directory_test
        Root directory with test images, one folder per class.

    Returns
    -------
    data_tensor
        EEG data (shape depends on *train* flag).
    label_tensor
        Corresponding integer class labels.
    texts
        Text prompts describing images.
    images
        Absolute paths to chosen image files.
    times
        Time-points vector extracted from EEG file.
    ch_names
        EEG channel-name list from file header.
    """
    
    # Constants  
    SAMPLES_PER_CLASS_TRAIN = 10
    SAMPLES_PER_CLASS_TEST = 1
    N_CLASSES_TRAIN = 1654
    N_CLASSES_TEST = 200

    CONFIG_PATH = Path("data_config.json")
    with CONFIG_PATH.open(encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)

    data_path = config["data_path"]
    img_directory_training = config["img_directory_training"]
    img_directory_test = config["img_directory_test"] 
    clip_image_text_embedding = config["clip_image_text_embedding"]
 
    # Build text prompts  
    img_dir = img_directory_training if train else img_directory_test
    dirnames = _list_subdirs(img_dir)
    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    texts = [f"This picture is {d.split('_', 1)[-1]}" for d in dirnames]
 
    # Collect image paths    
    images = _collect_images(img_dir, classes, pictures)
 
    # Load EEG data  
    data_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    times: Optional[torch.Tensor] = None
    ch_names: Optional[Sequence[str]] = None

    for subject in subjects:
        if train and subject == exclude_subject:
            continue

        fname = (
            "preprocessed_eeg_training.npy"
            if train
            else "preprocessed_eeg_test.npy"
        )
        fpath = os.path.join(data_path, subject, fname)
        data_np = np.load(fpath, allow_pickle=True)

        eeg = torch.from_numpy(data_np["preprocessed_eeg_data"]).float()
        times = torch.from_numpy(data_np["times"])[50:]
        ch_names = data_np["ch_names"]

        if train:
            samples_per_class = SAMPLES_PER_CLASS_TRAIN
            n_classes = N_CLASSES_TRAIN
        else:
            samples_per_class = SAMPLES_PER_CLASS_TEST
            n_classes = N_CLASSES_TEST

        # ----------------------------------------------------------------- #
        # Slice EEG by class / picture                                      #
        # ----------------------------------------------------------------- #
        if classes is not None and pictures is not None:
            # One sample per (class, picture) pair
            for c_idx, p_idx in zip(classes, pictures):
                start = c_idx * samples_per_class + p_idx
                if start < len(eeg):
                    seg = eeg[start : start + 1]
                    data_list.append(seg)
                    label_list.append(torch.tensor([c_idx], dtype=torch.long))
        elif classes is not None:
            # All pictures for specified classes
            for c_idx in classes:
                start = c_idx * samples_per_class
                seg = eeg[start : start + samples_per_class]
                lab = torch.full(
                    (samples_per_class,), c_idx, dtype=torch.long
                )
                data_list.append(seg)
                label_list.append(lab)
        else:
            # All classes / pictures
            for c_idx in range(n_classes):
                start = c_idx * samples_per_class
                seg = eeg[start : start + samples_per_class]
                lab = torch.full(
                    (samples_per_class,), c_idx, dtype=torch.long
                )
                data_list.append(seg)
                label_list.append(lab)

    if not data_list:
        raise RuntimeError("No EEG data matched the requested criteria.")

    # --------------------------------------------------------------------- #
    # Stack tensors                                                         #
    # --------------------------------------------------------------------- #
    if train:
        data_tensor = torch.cat(data_list).view(
            -1, *data_list[0].shape[2:]  # drop sample & picture dims
        )
    else:
        data_tensor = torch.cat(data_list)

    label_tensor = torch.cat(label_list)

    if train:
        # Duplicate labels four times to match data augmentation
        label_tensor = label_tensor.repeat_interleave(4)

        # Re-index labels if a subset of classes is used
        if classes is not None:
            unique = torch.unique(label_tensor)
            remap = {int(v): i for i, v in enumerate(unique)}
            label_tensor = torch.tensor(
                [remap[int(v)] for v in label_tensor], dtype=torch.long
            )
 
    return (
        data_tensor,
        label_tensor,
        texts,
        images,
        times,       # type: ignore[return-value]
        ch_names,    # type: ignore[return-value]
        data_path, 
        clip_image_text_embedding,
    )
 

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _list_subdirs(path: str):
    """Return sorted list of immediate sub-directories in *path*."""
    return sorted(
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )


def _list_images(path: str):
    """Return sorted list of image files (png/jpg/jpeg) in *path*."""
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")
    return sorted(
        f
        for f in os.listdir(path)
        if f.lower().endswith(IMG_EXTENSIONS)
    )

def _collect_images(
    img_root: str,
    dir_indices=None,
    pic_indices=None,
):
    """
    Gather image paths under *img_root*.

    Parameters
    ----------
    img_root
        Root directory that contains one folder per class.
    dir_indices
        Class indices to include (None → include all).
    pic_indices
        Image indices to include within each class (None → include all).

    Returns
    -------
    List[str]
        Absolute paths to chosen images.
    """
    folders = _list_subdirs(img_root)

    if dir_indices is not None:
        folders = [folders[i] for i in dir_indices]

    image_paths: List[str] = []
    for idx, folder in enumerate(folders):
        folder_path = os.path.join(img_root, folder)
        imgs = _list_images(folder_path)

        if pic_indices is None:
            chosen = imgs
        else:
            if idx < len(pic_indices) and pic_indices[idx] < len(imgs):
                chosen = [imgs[pic_indices[idx]]]
            else:
                continue

        image_paths.extend(os.path.join(folder_path, img) for img in chosen)

    return image_paths

 
if __name__ == "__main__":   

    data, labels, text, img, times, ch_names, data_path, model_clip = load_data(
        subjects = ['sub-01'], 
        train=False, 
    )
    # data.shape = [200, 80, 63, 250]
    # labels.shape = [200] 
    # len(text) = 200
    # times.shape = 250
    # ch_names.shape = 63
    # import pdb;pdb.set_trace()

 