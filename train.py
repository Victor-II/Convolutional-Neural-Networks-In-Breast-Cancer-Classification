import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torcheval.metrics import BinaryAccuracy
import torchvision.transforms as T
import time
import os
from models import resnet50_model, densenet121_model, inceptionv3_model
import pandas as pd

def binary_accuracy(outputs, labels, device):
    metric = BinaryAccuracy().to(device)
    metric.update(outputs, labels)
    return metric.compute()

def create_dl(data, batch_size):
    data_loader = DataLoader(data, batch_size, shuffle=True)
    return data_loader

def train_single_epoch(model, data_loader, criterion, optimiser, device):
    train_loss, train_accuracy = 0.0, 0.0

    for i, (batch, labels) in enumerate(data_loader):
        batch, labels = batch.to(device), labels.float().to(device)
        outputs = model(batch)
        loss = criterion(outputs.logits.squeeze(), labels)
        accuracy = binary_accuracy(outputs.logits.squeeze(), labels, device)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss += loss
        train_accuracy += accuracy

    train_loss = train_loss / float(i)
    train_accuracy = train_accuracy / float(i)

    return train_loss, train_accuracy

def evaluate(model, data_loader, criterion, device):
    val_loss, val_accuracy = 0.0, 0.0
    with torch.no_grad():
        for i, (batch, labels) in enumerate(data_loader, start=1):
            batch, labels = batch.to(device), labels.float().to(device)

            outputs = model(batch)
            loss = criterion(outputs.logits.squeeze(), labels)
            accuracy = binary_accuracy(outputs.logits.squeeze(), labels, device)

            val_loss += loss
            val_accuracy += accuracy
        
        val_loss = val_loss / float(i)
        val_accuracy = val_accuracy / float(i)

    return val_loss, val_accuracy

def train(model, criterion, optimiser, scheduler, epochs, train_dl, val_dl, save_path, save_best=True, device='cuda'):
    train_losses, train_accuracys, val_losses, val_accuracys = [], [], [], []
    start = time.time()
    print('Training started...\n')
    for epoch in range(epochs):
        start_epoch = time.time()
        train_loss, train_accuracy = train_single_epoch(model, train_dl, criterion, optimiser, device)
        val_loss, val_accuracy = evaluate(model, val_dl, criterion, device)
        scheduler.step(val_loss)

        if save_best == True:
            best_loss = 100
            if val_loss < best_loss:
                best_loss = val_loss
                try:
                    os.remove(save_path)
                except:
                    pass
                print('\nSaving model state dict...', end=' ')
                torch.save(model.state_dict(), save_path)
                print(f'State dict saved at "{save_path}"')

        train_losses.append(train_loss.cpu().detach().numpy())
        train_accuracys.append(train_accuracy.cpu().detach().numpy())
        val_losses.append(val_loss.cpu().detach().numpy())
        val_accuracys.append(val_accuracy.cpu().detach().numpy())

        elapsed_epoch = time.time() - start_epoch
        print(f'Epoch : [{epoch+1}/{epochs}], {elapsed_epoch / 60.0:.2f} [min]  |  Loss: {train_loss:.5f}  |  Accuracy: {train_accuracy:.5f}  |  Val_loss: {val_loss:.5f}  |  Val_accuracy: {val_accuracy:.5f}')
    elapsed = time.time() - start
    print('---------------------------------------------------------------------')
    print(f'Finished training in {elapsed / 60.0:.2f} [min]')

    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracys,
        'val_loss': val_losses,
        'val_accuracy': val_accuracys
    }

    return history

def save_history(history, save_path):
    history = pd.DataFrame(history)
    history.to_csv(save_path)

if __name__ == '__main__':

    TRAIN_DIR = 'Path/To/BreakHis_Split/train'
    VAL_DIR = 'Path/To/BreakHis_Split/val'
    TEST_DIR = 'Path/To/BreakHis_Split/test'
    SAVE_STATE_DICT = 'Path/To/saved_models/'
    SAVE_HISTORY = 'Path/To/saved_historys/'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 16
    EPOCHS = 100

    LEARNING_RATE = 0.001
    CRITERION = nn.BCELoss()

    model, name = inceptionv3_model()
    model.to(DEVICE)

    if 'inceptionv3' in name:
        SIZE = (299, 299)
    else:
        SIZE = (224, 224)

    train_transforms = T.Compose([T.Resize(size=SIZE), T.RandomHorizontalFlip(), T.ToTensor()])
    val_transforms = T.Compose([T.Resize(size=SIZE), T.ToTensor()])

    train_ds = ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    val_ds = ImageFolder(root=VAL_DIR, transform=val_transforms)

    train_dl = create_dl(train_ds, BATCH_SIZE)
    val_dl = create_dl(val_ds, BATCH_SIZE)

    OPTIMIZER = Adam(model.parameters(), lr=LEARNING_RATE)
    LR_SCHEDULER = ReduceLROnPlateau(optimizer=OPTIMIZER, factor=0.25, patience=5, threshold=0.001, verbose=True)

    history = train(model,
                    CRITERION, 
                    OPTIMIZER, 
                    LR_SCHEDULER, 
                    EPOCHS, 
                    train_dl, 
                    val_dl, 
                    SAVE_STATE_DICT+name+'.pt',
                    save_best=False,
                    device=DEVICE)
    
    save_history(history, SAVE_HISTORY+name+'_history.csv')
    
    
