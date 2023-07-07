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
# from torchvision.models import resnet50, densenet121, inception_v3
import pandas as pd
import results

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

# def resnet50_model(weights=None):
#     model = resnet50(weights=weights)
#     model.fc = nn.Sequential(
#         nn.Dropout(0.5),
#         nn.Linear(in_features=2048, out_features=1),
#         nn.Sigmoid()
#     )
#     return model

# def inceptionv3_model(weights=None):
#     model = inception_v3(weights=weights)
#     model.fc = nn.Sequential(
#         nn.Linear(in_features=2048, out_features=1),
#         nn.Sigmoid()
#     )
#     return model

# def densenet121_model(weights=None):
#     model = densenet121(weights=weights)
#     model.classifier = nn.Sequential(
#         nn.Dropout(0.5),
#         nn.Linear(in_features=1024, out_features=1),
#         nn.Sigmoid()
#     )
#     return model

# def main(args):
    
#     model = inceptionv3_model()
#     model.to(args.get('device'))

#     train_transforms = T.Compose([T.Resize(size=(299, 299)), T.RandomHorizontalFlip(), T.ToTensor()])
#     val_transforms = T.Compose([T.Resize(size=(299, 299)), T.ToTensor()])

#     train_ds = ImageFolder(root=args.get('train_dir'), transform=train_transforms)
#     val_ds = ImageFolder(root=args.get('val_dir'), transform=val_transforms)

#     train_dl = create_dl(train_ds, args.get('batch_size'))
#     val_dl = create_dl(val_ds, args.get('batch_size'))

#     criterion = nn.BCELoss()
#     optimiser = Adam(model.parameters(), lr=args.get('learning_rate'))
#     scheduler = ReduceLROnPlateau(optimizer=optimiser, factor=0.25, patience=5, threshold=0.001, verbose=True)

#     history = train(model, criterion, optimiser, scheduler, args.get('epochs'), train_dl, val_dl, args.get('save_state_dict'), args.get('device'))

#     return history

if __name__ == '__main__':

    args = {
        'device': 'cuda',
        'train_dir': 'C:/Users/Victor/Desktop/LICENTA/BreakHis_Split/train',
        'test_dir': 'C:/Users/Victor/Desktop/LICENTA/BreakHis_Split/test',
        'val_dir': 'C:/Users/Victor/Desktop/LICENTA/BreakHis_Split/val',
        'save_state_dict': 'C:/Users/Victor/Desktop/COD_LICENTA/saved_models/densenet_model.pt',
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 1
    }

    model = inceptionv3_model()
    model.to(args.get('device'))

    train_transforms = T.Compose([T.Resize(size=(299, 299)), T.RandomHorizontalFlip(), T.ToTensor()])
    val_transforms = T.Compose([T.Resize(size=(299, 299)), T.ToTensor()])

    train_ds = ImageFolder(root=args.get('train_dir'), transform=train_transforms)
    val_ds = ImageFolder(root=args.get('val_dir'), transform=val_transforms)

    train_dl = create_dl(train_ds, args.get('batch_size'))
    val_dl = create_dl(val_ds, args.get('batch_size'))

    criterion = nn.BCELoss()
    optimiser = Adam(model.parameters(), lr=args.get('learning_rate'))
    scheduler = ReduceLROnPlateau(optimizer=optimiser, factor=0.25, patience=5, threshold=0.001, verbose=True)

    history = train(model, criterion, optimiser, scheduler, args.get('epochs'), train_dl, val_dl, args.get('save_state_dict'), save_best=False, device=args.get('device'))
    
    