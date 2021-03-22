from densenet import DenseNet

from tqdm.notebook import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import sklearn.metrics

def make_dataloader(dataset, batch_size):
    """Make and return a data loader."""
    num_samples = len(dataset)
    class_idxs = torch.Tensor([dataset[i][1] for i in range(num_samples)]).long()
    sampler = WeightedRandomSampler(dataset.class_weights[class_idxs], num_samples)
    return DataLoader(dataset, batch_size, sampler=sampler)

def get_dataloaders(dataset_cls, root_dir=None, train_transform=None, test_transform=None, batch_size=64):
    """Create and return the train, test, and val data loaders.

    Parameters:
    - dataset_cls should be the dataset class to use for the data loader.
    - root_dir should be should be the path to the dataset root directory.
    - train_transform should be a callable to be applied on the train dataset.
    - test_transform should be a callable to be applied on the test and val datasets.
    - batch_size should be an integer for the batch size of the data loader.
    - shuffle should be a boolean for whether to shuffle the data in the data loader.
    """
    train_loader = make_dataloader(dataset_cls('train', root_dir, test_transform), batch_size)
    test_loader = make_dataloader(dataset_cls('test', root_dir, test_transform), batch_size)
    val_loader = make_dataloader(dataset_cls('val', root_dir, test_transform), batch_size)
    
    return train_loader, test_loader, val_loader
    
def train_model(model, train_loader, test_loader, device='cuda', epochs=90,
                optimizer='Adam', learning_rate=0.01, lr_scheduler=None,
                save_interval=0, model_path=None, history_path=None):
    """Train and test the model and return a training history.

    Parameters:
    - model should be able to accept images from the train and test loader.
    - train_loader should be able to provide images and labels to train the model.
    - test_loader should be able to provide images and labels to test the model.
    - device should be 'cuda' or 'cpu' for the location to mount the model and images.
    - epochs should be an integer for the number of epochs to train the model for.
    - optimizer should be a string for which optimizer from the torch.nn to use for training.
    - learning_rate should be a float for what learning rate to use for the optimizer.
    - lr_scheduler should be a function that takes the epoch and learning rate and returns a new learning rate.
    - save_interval should be an integer for how many epochs before saving the model and history.
    - model_path should be a path to store the model.
    - history_path should be a path to store the training history.
    """
    model.to(device)
    
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate)
    
    history = {'timestamp': [], 'epoch': []}
    
    start = datetime.now()
    print(f"[{start:%c}] Training started")
    
    for epoch in tqdm(range(1, epochs+1), desc='Progress'):
        if lr_scheduler:
            learning_rate = lr_scheduler(epoch, learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        results = run_model(model, train_loader, device, optimizer, training=True, progress=True, desc=f'Epoch {epoch}/{epochs}')
        
        test_results = run_model(model, test_loader)
        
        now = datetime.now()
        print(f"[{now:%c}] Time elapsed: {time_elapsed(start, now)} -",
              f"loss: {results['loss']:.3f} -",
              f"acc: {results['accuracy']:.3f} -",
              f"recall: {results['recall']:.3f} -",
              f"f1: {results['f1']:.3f} -",
              f"test_loss: {test_results['loss']:.3f} -",
              f"test_acc: {test_results['accuracy']:.3f} -",
              f"test_recall: {test_results['recall']:.3f} -",
              f"test_f1: {test_results['f1']:.3f}")

        history["timestamp"].append(now)
        history["epoch"].append(epoch)
        
        for k, v in results.items():
            if k not in history:
                history[k] = []
            history[k].append(v)
        
        for k, v in test_results.items():
            k = 'test_' + k
            if k not in history:
                history[k] = []
            history[k].append(v)
        
        if save_interval > 0 and epoch % save_interval == 0:
            save_model(model, model_path)
            save_history(history, history_path)
    
    now = datetime.now()
    print(f"[{now:%c}] Training complete - Time elapsed: {time_elapsed(start, now)}")
    
    return history

def run_model(model, data_loader, device='cuda', optimizer=None, training=False, progress=False, desc=None, show_results=False):
    """Run the model with data from the data loader and return the results.

    Parameters:
    - model should be able to accept images from the data loader.
    - data_loader should be able to provide images and labels to run the model.
    - device should be 'cuda' or 'cpu' for the location to mount the model and images.
    - optimizer should be able to be used to train the model parameters.
    - training should be a boolean for whether the model should be trained while running.
    - progress should be a boolean for whether a progress bar should be shown while running.
    - desc should be a string for the description to show on the progress bar while running.
    - show_results should be a boolean for whether to show the results.
    
    Note:
    If training is set to True, an optimizer should be provided for training.
    """
    if training:
        model.train()
        context = contextlib.suppress()
        if not optimizer:
            err_msg = 'training is set to True but optimizer was not provided. Please provide an optimizer for training.'
            raise ValueError(err_msg)
    else:
        model.eval()
        context = torch.no_grad()
    
    if progress:
        progress = tqdm(range(len(data_loader)), desc=desc)
    
    results = {'loss': 0}
    with context:
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            if training:
                optimizer.zero_grad()
            
            logits = model(images)
            output = F.log_softmax(logits, dim=1)
            
            weights = data_loader.dataset.class_weights
            current_loss = F.nll_loss(output, labels, weights.to(device))
            results['loss'] += current_loss.item()
            
            y_true = labels.data.cpu()
            y_pred = torch.exp(output).max(dim=1)[1].cpu()
            update_results(y_true, y_pred, results, weights[labels.long()])
            
            if training:
                current_loss.backward()
                optimizer.step()
            
            if progress:
                progress.update()
    
    n = len(data_loader)
    for metric in results:
        results[metric] /= n
    
    if show_results:
        plot_results(images, y_true, y_pred, data_loader.dataset.classes)
    
    return results

def update_results(y_true, y_pred, results=None, weights=None):
    """Evaluate the accuracy, precision, recall, and F1 scores and update and return the results."""
    if results is None:
        results = {}
    
    if 'accuracy' not in results:
        results['accuracy'] = 0
    results['accuracy'] += sklearn.metrics.accuracy_score(y_true, y_pred)
    
    average = 'weighted' if weights is not None else 'macro'
    for metric in ['precision', 'recall', 'f1']:
        if metric not in results:
            results[metric] = 0
        metric_fn = getattr(sklearn.metrics, metric+'_score')
        results[metric] += metric_fn(y_true, y_pred, average=average, sample_weight=weights, zero_division=0)
    
    return results

def plot_results(images, y_true, y_pred, classes, nrows=5, ncols=5):
    """Plot the predicted and true labels and the results of a batch of images."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    a = images.shape[0]
    sz = nrows * ncols
    if a <= sz:
        idxs = np.append(np.random.permutation(a), [None]*(sz-a))
    else:
        idxs = np.random.choice(a, sz, replace=False)
    axes = axes.flatten()
    for i in range(nrows*ncols):
        idx = idxs[i]
        if idx is not None:
            im = images[int(idx)].to('cpu').squeeze(dim=0)
            axes[i].imshow(im)
            axes[i].set_title(f'Actual: {classes[y_true[i]]}\nPredicted: {classes[y_pred[i]]}', size=6, y=0.97)
        axes[i].tick_params(which='both', left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()

def save_model(model, path='./models/temp.pt'):
    """Save a model at the specified path."""
    cp = {'model_args': model.args, 'state_dict': model.state_dict()}
    torch.save(cp, path)

def load_model(path):
    """Load and return a copy of the model stored at the specified path."""
    cp = torch.load(path)
    model = DenseNet(**cp['model_args'])
    model.load_state_dict(cp['state_dict'])
    return model

def save_history(history, path='./history/temp.pt'):
    """Save a training history to the specified path."""
    with open(path, 'wb') as file:
        pickle.dump(history, file)

def load_history(path):
    """Load a training history from the specified path."""
    with open(path, 'rb') as file:
        return pickle.load(file)

def plot_history(history, metric='accuracy'):
    """Plot out the loss and the specified metric data stored in the training history."""
    fig, ax = plt.subplots()
    
    l1, = ax.plot(history['epoch'], history[metric], color='blue', label='Training ' + metric)
    l2, = ax.plot(history['epoch'], history['test_' + metric], color='red', label='Test ' + metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.title())
    
    ax2 = ax.twinx()
    l3, = ax2.plot(history["epoch"], history["loss"], color="blue", linestyle='dotted', label="Training loss")
    l4, = ax2.plot(history["epoch"], history["test_loss"], color="red", linestyle='dotted', label="Test loss")
    ax2.set_ylabel("Loss")
    
    plt.legend(handles=[l1, l2, l3, l4], bbox_to_anchor=(1.10, 1), loc='upper left')
    plt.show()

def time_elapsed(start, end):
    """Return the time elapsed between start and end as a string in MM:SS format."""
    s = int((end - start).total_seconds())
    return f'{s//60}:{s%60:0>2d}'