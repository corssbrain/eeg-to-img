import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm 
from data.dataloader_leaveone import EEGDataset

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
# from util import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
import itertools
import math
import re
from nn.subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from nn.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from nn.subject_layers.Embed import DataEmbedding
import numpy as np
from loss import ClipLoss
import argparse
from torch import nn
from torch.optim import AdamW


from args import Config
from model.retrival_atms import ATMS

 

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def train_model(sub, eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, config):
    eeg_model.train()
    text_features_all = text_features_all.to(device).float() # (n_cls, d)
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha=0.99
    features_list = []  # List to store features
    save_features= True
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):

        

        # eeg_data.shape
        # torch.Size([64, 63, 250])
        # (Pdb) labels
        # tensor([1645,  596, 1312,  247,  300,  566,  484, 1159, 1197,  494,  753, 1453,
        #         1330,  312, 1061, 1645,  479,  538,  676, 1651, 1595,  520, 1621, 1392,
        #         48, 1495,  653,  812,  761,   73,  752,  222,  191,  288,  239,  566,
        #         192, 1269,  835,  800, 1385, 1568, 1439,  625,  863, 1101,  245, 1096,
        #         1444, 1517, 1110, 1508,  866,  431, 1352, 1206, 1546,  809,  604,  959,
        #         202,  253,  101, 1559])
        # (Pdb) text
        # ('This picture is yak', 'This picture is glasses', 'This picture is snowball', 'This picture is carrot', 'This picture is chute', 'This picture is fudge', 'This picture is enchilada', 'This picture is rice', 'This picture is sand', 'This picture is eyeliner', 'This picture is juice', 'This picture is teacup', 'This picture is soup', 'This picture is clipper1', 'This picture is platypus', 'This picture is yak', 'This picture is eggplant', 'This picture is flip_flop', 'This picture is hearing_aid', 'This picture is zebra', 'This picture is water_fountain', 'This picture is fireplace', 'This picture is window', 'This picture is straightjacket', 'This picture is bacon', 'This picture is toothpick', 'This picture is handcuff', 'This picture is license_plate', 'This picture is kayak', 'This picture is baseball', 'This picture is jug', 'This picture is can_opener', 'This picture is bulletin_board', 'This picture is chimney', 'This picture is car_seat', 'This picture is fudge', 'This picture is bulletproof_vest', 'This picture is shovel', 'This picture is lollipop', 'This picture is lectern', 'This picture is stirrup', 'This picture is vest', 'This picture is tamale', 'This picture is grille', 'This picture is marker', 'This picture is prune', 'This picture is carousel', 'This picture is praying_mantis', 'This picture is tape_measure', 'This picture is trashcan', 'This picture is punch1', 'This picture is tractor', 'This picture is mascara', 'This picture is dog', 'This picture is sprinkler', 'This picture is sauce', 'This picture is turtle', 'This picture is leopard', 'This picture is gold', 'This picture is ottoman', 'This picture is button1', 'This picture is catfish', 'This picture is beer', 'This picture is vacuum')
        # (Pdb) text_features.shape
        # torch.Size([64, 1024])
        # (Pdb) img.shape
        # *** AttributeError: 'tuple' object has no attribute 'shape'
        # (Pdb) len(img)
        # 64
        # (Pdb) img_features.shape
        # torch.Size([64, 1024])



        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        batch_size = eeg_data.size(0)  
        subject_id = extract_id_from_string(sub)
        # eeg_data = eeg_data.permute(0, 2, 1)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
        # if not config.insubject:
        #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)     
        eeg_features = eeg_model(eeg_data, subject_ids).float()
 
        
        features_list.append(eeg_features)
        logit_scale = eeg_model.logit_scale
        
        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
        # loss = img_loss + text_loss
        # print("text_loss", text_loss)
        # print("img_loss", img_loss)
        loss = alpha * img_loss + (1 - alpha) * text_loss
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        
        # Compute the corresponding logits
        logits_img = logit_scale * eeg_features @ img_features_all.T
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        # logits_single = (logits_text + logits_img) / 2.0        
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)



def evaluate_model(sub, eeg_model, dataloader, device, text_features_all, img_features_all, k, config):
    eeg_model.eval()

    
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    # Get all unique classes
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = eeg_data.size(0) 
            subject_id = extract_id_from_string(sub)
            # eeg_data = eeg_data.permute(0, 2, 1)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            # if not config.insubject:
            #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)    
            # import pdb;pdb.set_trace()         
            # eeg_data.shape
            # torch.Size([1, 80, 63, 250])
            # (Pdb) subject_ids
            # tensor([1], device='cuda:0')
            # [64, 63, 250]
            # import pdb;pdb.set_trace()
            eeg_features = eeg_model(eeg_data, subject_ids)

        
            logit_scale = eeg_model.logit_scale 
            # print(eeg_features.type, text_features.type, img_features.type)
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
            loss = img_loss*alpha + text_loss*(1-alpha)
            
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                # First, select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                selected_text_features = text_features_all[selected_classes]
                
                if k==200:
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get the predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1
                    
                    # logits_single is the model's output, shape (n_batch, n_classes)
                    # label is the true label, shape (n_batch,)
                    # Get the indices of the top-5 predictions
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    # For k=50 or 100, select k classes for evaluation
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get the predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc

def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device, text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    # logger = wandb_logger(config) if logger else None
    # logger.watch(eeg_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    
    for epoch in range(config.epochs):
        # Train the model
        train_loss, train_accuracy, features_tensor = train_model(sub, eeg_model, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all, config=config)
        if (epoch +1) % 5 == 0:                    
            # Get the current time and format it as a string (e.g., '2024-01-17_15-30-00')                  
            if config.insubject==True:       
                os.makedirs(f"./models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)            
            else:                
                os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        # Evaluate the model
        test_loss, test_accuracy, top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all,k=200, config=config)
        _, v2_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 2, config=config)
        _, v4_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 4, config=config)
        _, v10_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 10, config=config)
        _, v50_acc, v50_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all,  k=50, config=config)
        _, v100_acc, v100_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all,  k=100, config=config)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        # "train_loss": train_loss,
        # "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc":v50_top5_acc,
        "v100_top5_acc": v100_top5_acc
        }

        results.append(epoch_results)
        # If the test accuracy of the current epoch is the best, save the model and related information
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()
            
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc":v2_acc,
                "v4_acc":v4_acc,
                "v10_acc":v10_acc
            }
        # logger.log({
        #     "Train Loss": train_loss,
        #     "Train Accuracy": train_accuracy,
        #     "Test Loss": test_loss,
        #     "Test Accuracy": test_accuracy,
        #     "v2 Accuracy": v2_acc,
        #     "v4 Accuracy": v4_acc,
        #     "v10 Accuracy": v10_acc,
        #     "Epoch": epoch
        # })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
  
    # # Load the best model weights
    # model.load_state_dict(best_model_weights)

    # # # Save the best model
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')

    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you've already calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information for annotation
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    # logger.finish()
    return results

import datetime

def main():
    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="/gpfs/radev/home/aa2793/project/aria/sota/dataset/EEG_Image_decode_local/Preprocessed_data_250Hz", help='Path to the preprocessed data')
    # parser.add_argument('--data_path', type=str, default="/root/autodl-tmp/THINGS/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')    
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--logger', type=bool, default=False, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='ATMS', help='Encoder type')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')    
    args = parser.parse_args()

    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    subjects = args.subjects        
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in subjects:
        eeg_model = globals()[args.encoder_type]()
        eeg_model.to(device)

        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=args.lr)

        if args.insubject:
            train_dataset = EEGDataset(args.data_path, subjects=[sub], train=True)
            test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=True)
            test_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, current_time, eeg_model, train_loader, test_loader, optimizer, device, 
                                  text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config=args, logger=args.logger)


        # Save results to a CSV file
        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)

        if args.insubject:
            results_file = f"{results_dir}/{args.encoder_type}_{sub}.csv"
        else:
            results_file = f"{results_dir}/{args.encoder_type}_cross_exclude_{sub}.csv"

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')

                
if __name__ == '__main__':
    main()
