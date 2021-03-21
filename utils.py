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
from torch.utils.data import DataLoader

def get_dataloaders(dataset, root_dir=None, train_transform=None, test_transform=None, batch_size=64, shuffle=True):
    """Create and return the train, test, and val data loaders.

    Parameters:
    - dataset should be the dataset class to use for the data loader.
    - root_dir should be should be the path to the dataset root directory.
    - train_transform should be a callable to be applied on the train dataset.
    - test_transform should be a callable to be applied on the test and val datasets.
    - batch_size should be an integer for the batch size of the data loader.
    - shuffle should be a boolean for whether to shuffle the data in the data loader.
    """
    train_loader = DataLoader(dataset('train', root_dir, train_transform), batch_size, shuffle)
    test_loader = DataLoader(dataset('test', root_dir, test_transform), batch_size, shuffle)
    val_loader = DataLoader(dataset('val', root_dir, test_transform), batch_size, shuffle)
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
              f"f1: {results['f1_score']:.3f} -",
              f"test_loss: {test_results['loss']:.3f} -",
              f"test_acc: {test_results['accuracy']:.3f} -",
              f"test_recall: {test_results['recall']:.3f} -",
              f"test_f1: {test_results['f1_score']:.3f}")

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
        data_loader = tqdm(data_loader, desc=desc)
    
    criterion = nn.NLLLoss()
    
    results = {'loss': 0}
    with context:
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            if training:
                optimizer.zero_grad()
            
            logits = model(images)
            output = F.log_softmax(logits, dim=1)
            
            current_loss = criterion(output, labels)
            results['loss'] += current_loss.item()
            
            y_true = labels.data
            y_pred = torch.exp(output).max(dim=1)[1]
            update_results(y_true, y_pred, results)
            
            if training:
                current_loss.backward()
                optimizer.step()
    
    n = len(data_loader)
    for metric in results:
        results[metric] /= n
    
    if show_results:
        plot_results(images, y_true, y_pred, data_loader.dataset.classes)
    
    return results

def update_results(y_true, y_pred, results=None):
    """Evaluate the class-wise averaged accuracy, precision, recall, F1-score, and specificity and update and return the results."""
    if results is None:
        results = {}
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    
    for metric in metrics:
        if metric not in results:
            results[metric] = 0
    
    # Get unique classes
    classes = torch.cat([y_true, y_pred], dim=0).unique().int().tolist()
    
    # Compute scores for each class
    scores = {cls:{'tp':0, 'fp':0, 'tn':0, 'fn':0} for cls in classes}
    for yp, y in zip(y_pred, y_true):
        for cls in classes:
            if cls == y:
                res = 'tp' if yp == y else 'fn'
            elif cls == yp:
                res = 'fp'
            else:
                res = 'tn'
            scores[cls][res] += 1
    
    # Compute metrics for each class
    epsilon = 1e-7
    for cls in classes:
        tp = scores[cls]['tp']
        fp = scores[cls]['fp']
        tn = scores[cls]['tn']
        fn = scores[cls]['fn']
        
        accuracy    = (tp + tn) / (tp + tn + fp + fn + epsilon)
        precision   = tp / (tp + fp + epsilon)
        recall      = tp / (tp + fn + epsilon)
        f1_score    = 2 * (precision * recall) / (precision + recall + epsilon)
        specificity = tn / (tn + fp + epsilon)
        
        scores[cls]['accuracy'] = accuracy
        scores[cls]['precision'] = precision
        scores[cls]['recall'] = recall
        scores[cls]['f1_score'] = f1_score
        scores[cls]['specificity'] = specificity
    
    # Compute class-wise averaged metrics
    n = len(classes)
    for metric in metrics:
        score = sum(scores[cls][metric] for cls in classes) / n
        results[metric] += score
    
    return results

def plot_results(images, y_true, y_pred, classes, nrows=5, ncols=5):
    """Plot the predicted and true labels and the results of a batch of images."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    a = images.shape[0]
    sz = nrows * ncols
    if a <= sz:
        idxs = np.append(np.arange(a), [None]*(sz-a))
    else:
        idxs = np.arange(sz)
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
    
    plt.legend(handles=[l1, l2, l3, l4])
    plt.show()

def time_elapsed(start, end):
    """Return the time elapsed between start and end as a string in MM:SS format."""
    s = int((end - start).total_seconds())
    return f'{s//60}:{s%60:0>2d}'