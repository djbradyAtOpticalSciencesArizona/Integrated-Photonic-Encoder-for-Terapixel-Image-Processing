import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import os
import cv2 as cv

class SampleDataset(Dataset):
    def __init__(self, sample_dir, target_dir, transform=None):
        self.sample_dir = sample_dir
        self.target_dir = target_dir
        self.transform = transform
        self.samples = os.listdir(sample_dir)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        sample_path = os.path.join(self.sample_dir, self.samples[index])
        target_path = os.path.join(self.target_dir, self.samples[index].replace('.npy', '.png'))
        sample = np.load(sample_path)
        sample = sample.transpose((2,0,1))
        target = cv.imread(target_path, cv.IMREAD_GRAYSCALE).astype(np.float32)
        target = target/255.
        
        if self.transform is not None:
            augmentations = self.transform(sample=sample, target=target)
            sample = augmentations['sample']
            target = augmentations['target']
            
        return sample, target

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
    train_dir,
    train_targetdir,
    val_dir,
    val_targetdir,
    batch_size,
    num_workers=4,
    pin_memory=True,
    train_transform=None,
    val_transform=None,
):
    train_ds = SampleDataset(sample_dir=train_dir,
                             target_dir=train_targetdir,)
        
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=False,)
    
    val_ds = SampleDataset(sample_dir=val_dir,
                           target_dir=val_targetdir,)
        
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False,)
    
    return train_loader, val_loader

def check_accuracy(loader, model, device='cuda'):
    model.eval()
    
    loss = nn.MSELoss()
    loss_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).unsqueeze(1)
            preds = model(x)
            loss_value = loss(preds, y)
            loss_list.append(loss_value.detach().cpu().numpy())
    
    loss_avg = np.sum(loss_list)/len(loss_list)
    print(f'loss = {loss_avg}')
    
    model.train()
    
    return loss_avg

def save_predicitons_as_imgs(
    loader, model, folder = 'saved_predicted_images/', device='cuda'
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            preds = model(x)
            preds = preds.float()
        
        torchvision.utils.save_image(preds, f'{folder}/pred_{idx}.png')
        
        torchvision.utils.save_image(y.unsqueeze(1), f'{folder}/true_{idx}.png')
        
    model.train()

# xavier initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)