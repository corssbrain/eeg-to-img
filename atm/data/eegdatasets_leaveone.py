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




proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_type = 'ViT-H-14'

model_clip, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type,
    pretrained = "laion2b_s32b_b79k",   # open_clip will pull from your HF cache
    precision  = "fp32",
    device     = device,
)

config_path = "data_config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

data_path = config["data_path"]
img_directory_training = config["img_directory_training"]
img_directory_test = config["img_directory_test"]





def load_data(subjects, train, classes, exclude_subject, pictures, data_path):
    
    data_list, label_list = [], []
    texts, images = [], [] 
    if train: directory = img_directory_training
    else: directory = img_directory_test
    
    dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    dirnames.sort()
    
    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    for dir in dirnames:
        
        try:
            idx = dir.index('_')
            description = dir[idx+1:]  
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue
            
        new_description = f"This picture is {description}"
        texts.append(new_description)

    if train:
        img_directory = img_directory_training  
    else:
        img_directory = img_directory_test
    
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()  

    if classes is not None and pictures is not None:
        images = []  
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []  
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []  
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else: 
        print("Error")
         
    for subject in subjects:
        if train:
            if subject == exclude_subject:  
                continue            
            # print("subject:", subject)    
            file_name = 'preprocessed_eeg_training.npy'

            file_path = os.path.join(data_path, subject, file_name)
            data = np.load(file_path, allow_pickle=True)
            
            preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()                
            times = torch.from_numpy(data['times']).detach()[50:]
            ch_names = data['ch_names']  

            n_classes = 1654  
            samples_per_class = 10  
            
            if classes is not None and pictures is not None:
                for c, p in zip(classes, pictures):
                    start_index = c * 1 + p
                    if start_index < len(preprocessed_eeg_data):  
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]  
                        labels = torch.full((1,), c, dtype=torch.long).detach()  
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)  

            elif classes is not None and pictures is None:
                for c in classes:
                    start_index = c * samples_per_class
                    preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                    labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()  
                    data_list.append(preprocessed_eeg_data_class)
                    label_list.append(labels)

            else:
                for i in range(n_classes):
                    start_index = i * samples_per_class 
                    preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class] 
                    labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  
                    data_list.append(preprocessed_eeg_data_class)
                    label_list.append(labels)

        else:
            if subject == exclude_subject or exclude_subject==None:  
                file_name = 'preprocessed_eeg_test.npy'
                file_path = os.path.join(data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']  
                n_classes = 200  # Each class contains 1 images
                
                samples_per_class = 1  

                for i in range(n_classes):
                    if classes is not None and i not in classes:  # If we've defined specific classes and the current class is not in the list, skip
                        continue
                    start_index = i * samples_per_class  # Update start_index for each class
                    preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class] 
                    labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                    preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0) 
                    data_list.append(preprocessed_eeg_data_class)
                    label_list.append(labels)  # Add labels to the label list
            else:
                continue 
    if train:          
        data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])         
        print("data_tensor", data_tensor.shape)
    else:           
        data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)    
    label_tensor = torch.cat(label_list, dim=0) 
    if train: 
        label_tensor = label_tensor.repeat_interleave(4)
        if classes is not None:
            unique_values = list(label_tensor.numpy())
            lis = []
            for i in unique_values:
                if i not in lis:
                    lis.append(i)
            unique_values = torch.tensor(lis)        
            mapping = {val.item(): index for index, val in enumerate(unique_values)}   
            label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
    else: 
        pass       

    return data_tensor, label_tensor, texts, images, times, ch_names




 








class EEGDataset():
    """
    subjects = 
        ['sub-01', 'sub-02', 'sub-05', 
        'sub-04', 'sub-03', 'sub-06', 
        'sub-07', 'sub-08', 'sub-09', 'sub-10'
    ]
    """
    def __init__(
        self, 
        data_path, 
        exclude_subject=None, 
        subjects=None, 
        train=True, 
        time_window=[0, 1.0], 
        classes=None, 
        pictures=None, 
        val_size=None
    ):
        
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject  
        self.val_size = val_size
        
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)
         
        self.data, self.labels, self.text, self.img, self.times, self.ch_names = load_data(
            self.subjects, 
            self.train, 
            self.classes, 
            self.exclude_subject, 
            self.pictures, 
            self.data_path
        )

        self.data = self.extract_eeg(
            self.data, 
            time_window
        )
 
        if self.classes is None and self.pictures is None: # True
            features_filename = (
                os.path.join(f"{model_type}_features_train.pt")
                if self.train
                else os.path.join(f"{model_type}_features_test.pt")
            ) 
             
            if os.path.exists(features_filename): # True if ViT-H-14_features_train.pt exits
                saved_features = torch.load(features_filename, weights_only=False)
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
            else:
                self.text_features = self.Textencoder(self.text)
                self.img_features = self.ImageEncoder(self.img)
                torch.save({
                    'text_features': self.text_features.cpu(),
                    'img_features': self.img_features.cpu(),
                }, features_filename)
        else:
            self.text_features = self.Textencoder(self.text)
            self.img_features = self.ImageEncoder(self.img)

    def extract_eeg(self, eeg_data, time_window): 
        start, end = time_window 
        indices = (self.times >= start) & (self.times <= end) 
        extracted_data = eeg_data[..., indices]  
        return extracted_data

    def Textencoder(self, text):     
        text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(device)   # [1654, 77] 
        with torch.no_grad(): 
            text_features = model_clip.encode_text(text_inputs) 
        text_features = F.normalize(text_features, dim=-1).detach() 
        return text_features
        
    def ImageEncoder(self,images):
        batch_size = 20  
        image_features_list = [] 
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

            with torch.no_grad():
                batch_image_features = model_clip.encode_image(image_inputs)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0) 
        return image_features
 
    
    
    def __getitem__(self, index): 
        x_eeg = self.data[index]
        y_label = self.labels[index]
        
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
        
        return x_eeg, y_label, text, text_features, img, img_features

    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same

if __name__ == "__main__": 
    data_path = data_path
    train_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=True) 
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    import pdb;pdb.set_trace() 
    (x_eeg, label, text, text_features, img_path, img_features) = train_dataset[110]

    from PIL import Image 
    image = Image.open(img_path) 
    image.save('test.png')
     

    from visi import plot_umap_eeg_features, plot_umap_img_features 
    # transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    fig_eeg, ax_eeg, Z_eeg, reducer_eeg, meta_eeg = plot_umap_eeg_features(train_dataset, pct_images=0.00, random_state=42)
    fig_img, ax_img, Z_img, reducer_img, meta_img = plot_umap_img_features(train_dataset, pct_images=0.00, random_state=42)
 

    












    fig_eeg, ax_eeg, Z_eeg, reducer_eeg, meta_eeg = plot_umap_eeg_features(test_loader, pct_images=0.05, random_state=42)
    plt.savefig('test_loader__plot_umap_img_features.png')


    # # Image features UMAP
    # fig_img, ax_img, Z_img, reducer_img, meta_img = plot_umap_img_features(train_dataset, pct_images=0.05, random_state=42)
    

    # EEG UMAP
    

    plt.show()  # if running in a script/notebook without auto-display


    
    fig1, ax1, Z_joint, reducer_joint, meta_joint = plot_umap_img_and_eeg(train_dataset, pct_images=0.05, random_state=42)
# fig2, ax2, Z_eeg,  reducer_eeg,  meta_eeg  = plot_umap_eeg(train_dataset,     pct_images=0.05, random_state=42)
# plt.show() 

    for (x_eeg, label, text, text_features, img_path, img_features) in train_dataset:  
        x_img = transform(Image.open(img_path)) 
        # x_eeg.shape = torch.Size([63, 250])
        # x_img.shape = torch.Size([3, 32, 32])
        # img_features.shape = torch.Size([1024])

        

    

 
    import pdb;pdb.set_trace() 

    

    # train_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=True)    
    # test_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=False)    
    # train_dataset = EEGDataset(data_path, train=True) 
    # test_dataset = EEGDataset(data_path, train=False) 
    
    
    
    
    # 100 Hz
    
    
    
    i = 80*1-1
    x, label, text, text_features, img, img_features  = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
            
    
        
    