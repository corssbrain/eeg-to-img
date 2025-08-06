"""
This code creates an EEGDataset which creates a PyTorch DataLoader.
It contains EEGDataset which first loads raw EEG of the Things dataset
from things_eeg_loading. It then embeds the text and image of the dataset
using clip_text_image_encoder from clip_embedding.
"""

import os
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

import sys
from pathlib import Path
 

class EEGDataset(Dataset):
    """
    EEGDataset wraps multi-modal EEG experiments
    (EEG ✚ text ✚ image) in a PyTorch-compatible interface.

    Parameters
    ----------
    exclude_subject : Optional[str]
        Subject ID to leave out.
    subjects : Sequence[str]
        List of subject IDs to load.
    train : bool
        If True, use train-mode sizing; otherwise test-mode.
    time_window : Sequence[float]
        `[start, end]` window (in seconds) for the EEG clip.
    classes : Optional[Sequence[int]]
        Subset of class indices to keep (if any).
    pictures : Optional[Sequence[int]]
        Subset of picture indices to keep (if any).
    val_size : Optional[int]
        Validation split size (handled elsewhere if needed).
    """

    def __init__(
        self,
        exclude_subject: Optional[str] = None,
        subjects: Sequence[str] = ("sub-01",),
        train: bool = True,
        time_window: Sequence[float] = (0.0, 1.0),
        classes: Optional[Sequence[int]] = None,
        pictures: Optional[Sequence[int]] = None,
        val_size: Optional[int] = None,
    ) -> None:
        device = 'cuda'
        # loading the raw data: eeg, image, and text
        from data.things_eeg_loading import load_image_text_eeg 
        # from things_eeg_loading import load_image_text_eeg  
        (
            raw_data,
            self.labels,
            self.text,
            self.img,
            self.times,
            self.ch_names,
            self.data_path,   
            clip_image_text_embedding, 
        ) = load_image_text_eeg(subjects=subjects, train=train)

        self.train = train
        self.subject_list: List[str] = os.listdir(self.data_path)
        self.subjects = subjects if subjects is not None else self.subject_list
        self.n_subjects = len(self.subjects)
        self.time_window = tuple(time_window)
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject
        self.val_size = val_size

        if not any(sub in self.subject_list for sub in self.subjects):
            raise ValueError("None of the requested subjects are available in data_path.")
 
        # Pre-compute EEG clips and CLIP features   
        self.data = self._extract_eeg(raw_data, self.time_window)

        from data.clip_embedding import clip_text_image_encoder 
        # from clip_embedding import clip_text_image_encoder 
        (
            self.text_features,
            self.img_features,
        ) = clip_text_image_encoder(self.text, self.img, pictures, classes, train, device, clip_image_text_embedding)
   
    # Private helpers     
    def _extract_eeg(self, eeg_data: torch.Tensor, window: Tuple[float, float]) -> torch.Tensor:
        """Crop raw EEG to `window` (start, end) in seconds."""
        start, end = window
        mask = (self.times >= start) & (self.times <= end)
        return eeg_data[..., mask]
 
    # PyTorch Dataset protocol 
    def __len__(self) -> int:  # noqa: D401  (one-liner is fine here)
        """Return total number of EEG epochs."""
        return self.data.shape[0]
 
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 10 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
   
        text = self.text[text_index]
        img = self.img[img_index]
        
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        
        return x, label, text, text_features, img, img_features

if __name__ == "__main__": 

    from torch.utils.data import Dataset, DataLoader

    test_dataset = EEGDataset(subjects=["sub-01"], train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
 
    # ... or iterate through the DataLoader:
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx == 110:
            sample = batch
            break

    (
        x_eeg,
        y_label_int,
        y_description,
        x_text_feat,
        img_path,
        x_img_feat,
    ) = sample

    import pdb;pdb.set_trace()