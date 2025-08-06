"""
The aim of this code is to load raw EEG dataset of Things dataset.
"""

import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch


from pathlib import Path
import os
import json
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


def load_image_text_eeg(
    subjects: Sequence[str],
    train: bool,
    classes: Optional[Sequence[int]] = None,
    exclude_subject: Optional[str] = None,
    pictures: Optional[Sequence[int]] = None,
    img_directory_training: str = "",
    img_directory_test: str = "",
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

    # ------------------------------------------------------------------ #
    # 1. Resolve paths and constants (use JSON defaults unless over‐
    #    ridden by the caller – mimics Dataset behaviour)                #
    # ------------------------------------------------------------------ #
    cfg = json.load(Path("./data/data_config.json").open(encoding="utf-8"))
    data_path = cfg["data_path"]
    clip_image_text_embedding = cfg["clip_image_text_embedding"]
    img_directory_training = (
        img_directory_training or cfg["img_directory_training"]
    )
    img_directory_test = img_directory_test or cfg["img_directory_test"]

    SAMPLES_PER_CLASS_TRAIN = 10
    SAMPLES_PER_CLASS_TEST = 1
    N_CLASSES_TRAIN = 1654
    N_CLASSES_TEST = 200

    # ------------------------------------------------------------------ #
    # 2. Build caption list                                               #
    # ------------------------------------------------------------------ #
    img_root = img_directory_training if train else img_directory_test
    dirnames = sorted(
        d
        for d in os.listdir(img_root)
        if os.path.isdir(os.path.join(img_root, d))
    )
    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    texts = [
        f"This picture is {d.split('_', 1)[-1]}"
        for d in dirnames
        if "_" in d
    ]

    # ------------------------------------------------------------------ #
    # 3. Collect image paths (100 % same logic as Dataset.load_data)      #
    # ------------------------------------------------------------------ #
    images: List[str] = []
    all_folders = sorted(
        d
        for d in os.listdir(img_root)
        if os.path.isdir(os.path.join(img_root, d))
    )

    def _all_imgs(folder_path: str) -> List[str]:
        return sorted(
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    if classes is not None and pictures is not None:
        for c_idx, p_idx in zip(classes, pictures):
            if c_idx < len(all_folders):
                folder_path = os.path.join(img_root, all_folders[c_idx])
                imgs = _all_imgs(folder_path)
                if p_idx < len(imgs):
                    images.append(os.path.join(folder_path, imgs[p_idx]))
    elif classes is not None:  # pictures is None
        for c_idx in classes:
            if c_idx < len(all_folders):
                folder_path = os.path.join(img_root, all_folders[c_idx])
                images.extend(
                    os.path.join(folder_path, img) for img in _all_imgs(folder_path)
                )
    else:  # classes is None → all images
        for folder in all_folders:
            folder_path = os.path.join(img_root, folder)
            images.extend(
                os.path.join(folder_path, img) for img in _all_imgs(folder_path)
            )

    # ------------------------------------------------------------------ #
    # 4. Load EEG and slice exactly like Dataset.load_data               #
    # ------------------------------------------------------------------ #
    data_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    times: Optional[torch.Tensor] = None
    ch_names: Optional[Sequence[str]] = None

    for subject in subjects:
        if train and subject == exclude_subject:
            continue

        fname = (
            "preprocessed_eeg_training.npy" if train else "preprocessed_eeg_test.npy"
        )
        fpath = os.path.join(data_path, subject, fname)
        npz = np.load(fpath, allow_pickle=True)

        eeg = torch.from_numpy(npz["preprocessed_eeg_data"]).float()
        times = torch.from_numpy(npz["times"]).detach()[50:]
        ch_names = npz["ch_names"]

        if train:
            samples_per_class = SAMPLES_PER_CLASS_TRAIN
            n_classes = N_CLASSES_TRAIN
        else:
            samples_per_class = SAMPLES_PER_CLASS_TEST
            n_classes = N_CLASSES_TEST

        # ------------ slicing logic (bug-compatible on purpose) -------- #
        if train:
            if classes is not None and pictures is not None:
                for c, p in zip(classes, pictures):
                    start = c * 1 + p                       # <- keep ×1
                    if start < len(eeg):
                        seg = eeg[start : start + 1]
                        data_list.append(seg)
                        label_list.append(torch.tensor([c], dtype=torch.long))
            elif classes is not None:
                for c in classes:
                    start = c * samples_per_class
                    seg = eeg[start : start + samples_per_class]
                    lab = torch.full((samples_per_class,), c, dtype=torch.long)
                    data_list.append(seg)
                    label_list.append(lab)
            else:
                for c in range(n_classes):
                    start = c * samples_per_class
                    seg = eeg[start : start + samples_per_class]
                    lab = torch.full((samples_per_class,), c, dtype=torch.long)
                    data_list.append(seg)
                    label_list.append(lab)
        else:  # test mode
            for c in range(n_classes):
                if classes is not None and c not in classes:
                    continue
                start = c * samples_per_class
                seg = eeg[start : start + samples_per_class]
                seg = torch.mean(seg.squeeze(0), 0)          # <- mean collapse ####################### <------ MEAN COLLAPSE
                lab = torch.full((samples_per_class,), c, dtype=torch.long)
                data_list.append(seg)
                label_list.append(lab)

    if not data_list:
        raise RuntimeError("No EEG data matched the requested criteria.")

    # ------------------------------------------------------------------ #
    # 5. Stack tensors exactly like Dataset.load_data                    #
    # ------------------------------------------------------------------ #
    if train:
        data_tensor = torch.cat(data_list, 0).view(
            -1, *data_list[0].shape[2:]
        )
    else:
        data_tensor = torch.cat(data_list, 0).view(
            -1, *data_list[0].shape
        )

    label_tensor = torch.cat(label_list, 0)

    if train:
        label_tensor = label_tensor.repeat_interleave(4)
        if classes is not None:
            unique_vals = []
            for v in label_tensor.tolist():
                if v not in unique_vals:
                    unique_vals.append(v)
            mapping = {v: i for i, v in enumerate(unique_vals)}
            label_tensor = torch.tensor([mapping[v] for v in label_tensor], dtype=torch.long)

    # ------------------------------------------------------------------ #
    # 6. Done                                                            #
    # ------------------------------------------------------------------ #
    return data_tensor, label_tensor, texts, images, times, ch_names, data_path, clip_image_text_embedding
 