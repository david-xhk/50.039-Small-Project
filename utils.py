from densenet import DenseNet

from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import pickle
import pytz

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

import sklearn.metrics

def make_model(dataloader_args, model_args, training_args):
    """Make model function.
    
    Data loaders are first created using dataloader_args dict (See get_dataloaders function for arguments).
    DenseNet model is initialized with model_args dict (See DenseNet class for initialization arguments).
    Training on the model is carried out according to training_args dict (See train_model function for arguments).
    The learning curve is plotted after training is completed.
    Finally, the trained model is returned.
    """
    train_loader, test_loader, _ = get_dataloaders(**dataloader_args)

    model = DenseNet(**model_args)

    history = train_model(model, train_loader, test_loader, **training_args)
    
    plot_history(history, metric='accuracy')
    
    return model

def test_model(dataloader_args=None, training_args=None, model=None):
    """Test model function.
    
    If no model is provided, the model is loaded from the model path in training_args.
    The validation data loader is created using dataloader_args (See get_dataloaders function for arguments).
    The model is tested with the validation data loader and the results are displayed.
    If no model or model_path are provided, no testing happens.
    """
    if not isinstance(model, nn.Module) and training_args and 'model_path' in training_args:
        model = load_model(training_args['model_path'])
    
    if model is not None:
        _, _, val_loader = get_dataloaders(**dataloader_args)
        
        results = run_model(model, val_loader, show_results=True)
        
        print(f"Validation loss: {results['loss']:.3f} -",
              f"Validation accuracy: {results['accuracy']:.3f} -",
              f"Validation F1-score: {results['f1']:.3f} -",
              f"Validation recall: {results['recall']:.3f}")

def get_dataloaders(*, dataset_cls, root_dir=None, train_transform=None, test_transform=None, batch_size=64, num_samples=None):
    """Create and return the train, test, and val data loaders.

    Parameters:
    - dataset_cls should be the dataset class to use for the data loader.
    - root_dir should be should be the path to the dataset root directory.
    - train_transform should be a callable to be applied on the train dataset.
    - test_transform should be a callable to be applied on the test and val datasets.
    - batch_size should be an integer for the batch size of the data loader.
    - num_samples should be an integer for how many training samples to draw if data augmentation is employed.
    """
    train_dataset = dataset_cls('train', root_dir, train_transform)
    test_dataset = dataset_cls('test', root_dir, test_transform)
    val_dataset = dataset_cls('val', root_dir, test_transform)
    
    train_loader = make_dataloader(train_dataset, batch_size, sampler='weighted', replacement=True, num_samples=num_samples)
    test_loader = make_dataloader(test_dataset, batch_size)
    val_loader = make_dataloader(val_dataset, batch_size)
    
    return train_loader, test_loader, val_loader

def make_dataloader(dataset, batch_size, sampler=None, replacement=False, num_samples=None):
    """Make and return a data loader.
    
    If sampler is set to 'weighted', a weighted random sampler is used.
    Otherwise, if sampler is set to None, a random sampler is used.
    If num_samples is not specified, the default is set to len(dataset).
    """
    N = len(dataset)
    if sampler == 'weighted':
        class_idxs = torch.Tensor([dataset[i][1] for i in range(N)]).long()
        weights = dataset.class_weights[class_idxs]
        if num_samples is None or not replacement and num_samples > N:
            num_samples = N
        sampler = WeightedRandomSampler(weights, num_samples, replacement)
    
    elif sampler is None:
        if num_samples is None and replacement:
            num_samples = N
        elif not replacement:
            num_samples = None
        sampler = RandomSampler(dataset, replacement, num_samples)
    
    else:
        err_msg = "sampler should be either 'weighted' or None"
        raise ValueError(err_msg)
    
    return DataLoader(dataset, batch_size, sampler=sampler)

def train_model(model, train_loader, test_loader, *, device='cuda',
                optimizer='Adam', learning_rate=0.01, patience=20,
                model_path=None, history_path=None):
    """Train and test the model and return a training history.

    Parameters:
    - model should be able to accept images from the train and test loader.
    - train_loader should be able to provide images and labels to train the model.
    - test_loader should be able to provide images and labels to test the model.
    - device should be 'cuda' or 'cpu' for the location to mount the model and images.
    - optimizer should be a string for which optimizer from the torch.nn to use for training.
    - learning_rate should be a float for what learning rate to use for the optimizer.
    - patience should be an integer for how many epochs before activating early stopping.
    - model_path should be a path to store the model.
    - history_path should be a path to store the training history.
    """
    model.device = device
    model.to(device)
    
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate)
    
    history = {'timestamp': [], 'epoch': []}
    
    start = time_now()
    print(f"[{start:%c}] Training started")
    
    best_epoch = 1
    early_stopping = tqdm(total=patience, desc='Early Stopping', leave=True)
    
    epoch = 1
    while True:
        results = run_model(model, train_loader, optimizer, training=True, progress=True, desc=f'Epoch {epoch}')

        test_results = run_model(model, test_loader)

        now = time_now()
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

        if test_results['loss'] > history['test_loss'][best_epoch-1]:
            early_stopping.update(1)
            
            if epoch >= best_epoch + patience:
                print(f"Early stopping activated at epoch {epoch} -",
                      f"Best epoch: {best_epoch} -",
                      f"test_loss: {history['test_loss'][best_epoch-1]:.3f} -",
                      f"test_acc: {history['test_accuracy'][best_epoch-1]:.3f} -",
                      f"test_recall: {history['test_recall'][best_epoch-1]:.3f} -",
                      f"test_f1: {history['test_f1'][best_epoch-1]:.3f}")
                
                history['early_stopping'] = best_epoch
                break
        
        else:
            best_epoch = epoch
            early_stopping.reset()
            save_model(model, model_path)
        
        epoch += 1
    
    early_stopping.close()
    
    model.load_state_dict(torch.load(model_path)['state_dict'])
    
    now = time_now()
    print(f"[{now:%c}] Training complete - Time elapsed: {time_elapsed(start, now)}")
    
    save_history(history, history_path)
    
    return history

def run_model(model, data_loader, optimizer=None, training=False, progress=False, desc=None, show_results=False):
    """Run the model with data from the data loader and return the results.

    Parameters:
    - model should be able to accept images from the data loader.
    - data_loader should be able to provide images and labels to run the model.
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
    
    N = len(data_loader)
    
    if progress:
        progress_bar = tqdm(total=N, desc=desc)
    
    class_weights = data_loader.dataset.class_weights.to(model.device)
    
    results = {'loss': 0}
    with context:
        for images, labels in data_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            
            if training:
                optimizer.zero_grad()
            
            logits = model(images)
            output = F.log_softmax(logits, dim=1)
            
            loss = F.nll_loss(output, labels, class_weights)
            results['loss'] += loss.item()
            
            y_true = labels.data.cpu()
            y_pred = torch.exp(output).max(dim=1)[1].cpu()
            sample_weights = class_weights[labels.long()].cpu()
            update_results(y_true, y_pred, sample_weights, results)
            
            if show_results:
                plot_results(images, y_true, y_pred, data_loader.dataset.classes)
            
            if training:
                loss.backward()
                optimizer.step()
            
            if progress:
                progress_bar.update(1)
        
        if progress:
            progress_bar.close()
    
    for metric in results:
        results[metric] /= N
    
    return results

def update_results(y_true, y_pred, weights=None, results=None):
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
            im = images[int(idx)].cpu().squeeze(dim=0)
            axes[i].imshow(im)
            axes[i].set_title(f'Actual: {classes[y_true[i]]}\nPredicted: {classes[y_pred[i]]}', size=6, y=0.97)
        axes[i].tick_params(which='both', left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()

def save_model(model, path='./models/temp.pt'):
    """Save a model at the specified path."""
    cp = {'model_args': model.args, 'model_device': model.device, 'state_dict': model.state_dict()}
    torch.save(cp, path)

def load_model(path):
    """Load and return a copy of the model stored at the specified path."""
    cp = torch.load(path)
    model = DenseNet(**cp['model_args'])
    model.load_state_dict(cp['state_dict'])
    model.device = cp['model_device']
    model.to(model.device)
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
    l2, = ax.plot(history['epoch'], history['test_' + metric], color='orange', label='Test ' + metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.title())
    
    ax2 = ax.twinx()
    l3, = ax2.plot(history['epoch'], history['loss'], color='blue', linestyle='dotted', label='Training loss')
    l4, = ax2.plot(history['epoch'], history['test_loss'], color='orange', linestyle='dotted', label='Test loss')
    ax2.set_ylabel('Loss')
    
    l5 = plt.axvline(x=history['early_stopping'], color='red', linestyle='dashed', label='Early stopping')
    plt.legend(handles=[l1, l3, l2, l4, l5], bbox_to_anchor=(1.10, 1), loc='upper left')
    plt.show()

def time_now(tz='Singapore'):
    """Return the current time in the specified timezone."""
    return datetime.now(tz=pytz.timezone(tz))

def time_elapsed(start, end):
    """Return the time elapsed between start and end as a string in MM:SS format."""
    s = int((end - start).total_seconds())
    return f'{s//60}:{s%60:0>2d}'