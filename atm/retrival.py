"""Train and evaluate EEG-based retrieval models.

This script trains an EEG encoder to align EEG features with corresponding
image and text embeddings. It supports both in-subject and cross-subject
training and reports top-1 / top-5 accuracy for various retrieval set sizes.

Requires:
    * PyTorch
    * data.dataloader_leaveone.EEGDataset
    * utils.{args,plots,utils}

"""

from __future__ import annotations

# std-lib imports 
import itertools
import os
import random
from typing import List, Tuple

# third-party imports 
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# local-app imports
from args import args_function
from data.dataloader_leaveone import EEGDataset
from utils.plots import plot_metrics
from utils.utils import clear_screen, extract_id_from_string

# model imports 
from model.retrival_atms import ATMS
from braindecode.models import (
    EEGNetv4,
    ATCNet,
    EEGConformer,
    EEGITNet,
    ShallowFBCSPNet,
)

 
# helpers 
def train_model(
    subject: str,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    text_feats_all: torch.Tensor,
    img_feats_all: torch.Tensor,
    alpha: float = 0.99,
) -> Tuple[float, float, torch.Tensor]:
    """One training epoch."""
    model.train()

    text_feats_all = text_feats_all.to(device).float()          # (n_cls, d)
    img_feats_all = img_feats_all[::10].to(device).float()      # one image / class

    total_loss = correct = total = 0
    feats_epoch: List[torch.Tensor] = []

    for eeg, labels, _, text_feats, _, img_feats in dataloader:
        eeg, labels = eeg.to(device), labels.to(device)
        text_feats, img_feats = text_feats.to(device).float(), img_feats.to(device).float()

        optimizer.zero_grad()

        subj_id = extract_id_from_string(subject)
        subj_ids = torch.full((eeg.size(0),), subj_id, dtype=torch.long, device=device)

        eeg_feats = model(eeg, subj_ids).float()                               # model([64, 63, 250], [64]) --> [64, 1024]
        feats_epoch.append(eeg_feats)
        logit_scale = model.logit_scale
    
        loss_img = model.loss_func(eeg_feats, img_feats, logit_scale)          # 6.8374 = ([64, 1024], [64, 1024], 2.6593)
        loss_txt = model.loss_func(eeg_feats, text_feats, logit_scale)         # 6.6043 = ([64, 1024], [64, 1024], 2.6593)
        loss = alpha * loss_img + (1 - alpha) * loss_txt                       # 6.8350 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        logits = logit_scale * eeg_feats @ img_feats_all.T                     # [64, 1654] = 2.6593 * [64, 1024]  @ [1024, 1654]
        preds = logits.argmax(dim=1)                                           # [64]

        total += preds.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, torch.cat(feats_epoch, dim=0)


def evaluate_model(
    subject: str,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    text_feats_all: torch.Tensor,
    img_feats_all: torch.Tensor,
    k: int = 200,
    alpha: float = 0.99,
) -> Tuple[float, float, float]:
    """
        Evaluate top-1 and top-5 accuracy on *k*-way retrieval.
    """
    model.eval()

    text_feats_all = text_feats_all.to(device).float()
    img_feats_all = img_feats_all.to(device).float()

    total_loss = correct = total = top5_correct = 0
    all_labels = set(range(text_feats_all.size(0)))

    with torch.no_grad():
        for eeg, labels, _, text_feats, _, img_feats in dataloader:
            eeg, labels = eeg.to(device), labels.to(device)
            text_feats, img_feats = text_feats.to(device).float(), img_feats.to(device).float()

            subj_id = extract_id_from_string(subject)
            subj_ids = torch.full((eeg.size(0),), subj_id, dtype=torch.long, device=device)
            eeg_feats = model(eeg, subj_ids)

            logit_scale = model.logit_scale
            loss_img = model.loss_func(eeg_feats, img_feats, logit_scale)
            loss_txt = model.loss_func(eeg_feats, text_feats, logit_scale)
            total_loss += (alpha * loss_img + (1 - alpha) * loss_txt).item()

            # “Here’s your brain signal (EEG). Now I’m going to give you a pile of k pictures — one is the correct match, the others are distractors. Can you find the right one?”
            for idx, lbl in enumerate(labels):
                neg_classes = list(all_labels - {lbl.item()})
                sel_classes = random.sample(neg_classes, k - 1) + [lbl.item()]
                sel_img_feats = img_feats_all[sel_classes]

                logits = logit_scale * eeg_feats[idx] @ sel_img_feats.T
                pred_lbl = sel_classes[logits.argmax().item()]
                 
                correct += pred_lbl == lbl.item()
                total += 1

                if k >= 50:  # only compute top-5 for larger polls
                    top5 = logits.topk(5).indices.tolist()
                    if lbl.item() in [sel_classes[i] for i in top5]:
                        top5_correct += 1

    avg_loss = total_loss / len(dataloader)
    topk_acc = correct / total
    top5_acc = top5_correct / total if k >= 50 else 0.0
    return avg_loss, topk_acc, top5_acc


def main_train_loop(
    subject: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    txt_train: torch.Tensor,
    txt_test: torch.Tensor,
    img_train: torch.Tensor,
    img_test: torch.Tensor,
    cfg,
):
    """Full training / evaluation loop for a single subject."""
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    v2_accs, v4_accs, v10_accs = [], [], []

    best_acc = 0.0
    best_epoch_info = {}

    clear_screen()
    for epoch in range(cfg.epochs):
        tr_loss, tr_acc, _ = train_model(
            subject,
            model,
            train_loader,
            optimizer,
            device,
            txt_train,
            img_train,
        )
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)

        te_loss, te_acc, top5_acc = evaluate_model(
            subject, model, test_loader, device, txt_test, img_test, k=200
        )
        _, v2_acc, _ = evaluate_model(subject, model, test_loader, device, txt_test, img_test, k=2)
        _, v4_acc, _ = evaluate_model(subject, model, test_loader, device, txt_test, img_test, k=4)
        _, v10_acc, _ = evaluate_model(subject, model, test_loader, device, txt_test, img_test, k=10)
        _, v50_acc, v50_top5 = evaluate_model(
            subject, model, test_loader, device, txt_test, img_test, k=50
        )
        _, v100_acc, v100_top5 = evaluate_model(
            subject, model, test_loader, device, txt_test, img_test, k=100
        )

        test_losses.append(te_loss)
        test_accs.append(te_acc)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "test_loss": te_loss,
                "test_acc": te_acc,
                "v2_acc": v2_acc,
                "v4_acc": v4_acc,
                "v10_acc": v10_acc,
            }

        print(
            f"Ep {epoch + 1:03}│ "
            f"trLoss {tr_loss:.3f}, trAcc {tr_acc:.2%} │ "
            f"teLoss {te_loss:.3f}, teAcc/top5 {te_acc:.2%}/{top5_acc:.2%} │ "
            f"Acc@k (2/4/10/50/100): {v2_acc:.2%}/{v4_acc:.2%}/"
            f"{v10_acc:.2%}/{v50_acc:.2%}/{v100_acc:.2%}"
        )

    plot_metrics(
        train_losses,
        test_losses,
        train_accs,
        test_accs,
        v2_accs,
        v4_accs,
        v10_accs,
        best_epoch_info, 
        save_path="retrival_results.png",
    )

# main
def main() -> None:
    """Parse arguments and launch subject-wise training."""
    args, device = args_function()

    # central registry of encoders
    ENCODER_REGISTRY = {
        "ATMS": ATMS,
        "EEGNetv4": EEGNetv4,
        "ATCNet": ATCNet,
        "EEGConformer": EEGConformer,
        "EEGITNet": EEGITNet,
        "ShallowFBCSPNet": ShallowFBCSPNet,
    }

    for subject in args.subjects:
        model = ENCODER_REGISTRY[args.encoder_type]().to(device)
        optimizer = AdamW(itertools.chain(model.parameters()), lr=args.lr)

        if args.insubject:
            train_ds = EEGDataset(args.data_path, subjects=[subject], train=True)
            test_ds = EEGDataset(args.data_path, subjects=[subject], train=False)
        else:
            train_ds = EEGDataset(
                args.data_path, exclude_subject=subject, subjects=args.subjects, train=True
            )
            test_ds = EEGDataset(
                args.data_path, exclude_subject=subject, subjects=args.subjects, train=False
            )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=True, num_workers=0, drop_last=True
        )

        main_train_loop(
            subject,
            model,
            train_loader,
            test_loader,
            optimizer,
            device,
            train_ds.text_features,
            test_ds.text_features,
            train_ds.img_features,
            test_ds.img_features,
            cfg=args,
        )

if __name__ == "__main__":
    main()