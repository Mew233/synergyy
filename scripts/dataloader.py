"""
    Dataloader, customized K-fold trainer & evaluater
"""
from torch.utils.data import DataLoader
from itertools import chain
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# def dataloader(X_train_val_set, X_test_set, Y_train_val_set, Y_test_set):
def dataloader(*args,**kwargs):
    """
        should be starting from X, and Y
        X_{}_trainval, X_{}_test, Y_{}_trainval, Y_{}_test
    """
    temp_loader = {}
    for name, input in kwargs.items():
    # First, format
        input_name = name.split('_')
        if input_name[0].startswith('Y'):
            input = input.astype('float32')
            input = torch.from_numpy(input)
            input = input.unsqueeze(1)
            temp_loader[name] = input

        elif input_name[0].startswith('X'):
            input = input.astype('float32')
            input = torch.from_numpy(input)
            temp_loader[name] = input
    
    # Second, to tensordataset
    temp_loader_trainval,  temp_loader_test = [], []
    for key, val in temp_loader.items():
        load_name = key.split('_')
        # train_val_dataset
        if load_name[-1].endswith('trainval'):
            temp_loader_trainval.append(val)
        elif load_name[-1].endswith('test'):
            temp_loader_test.append(val)

    train_val_dataset = torch.utils.data.TensorDataset(*temp_loader_trainval)
    test_dataset = torch.utils.data.TensorDataset(*temp_loader_test)
    test_loader = DataLoader(test_dataset, batch_size=256,shuffle = True)
        
    #     Y_train_val_set = Y_train_val_set.astype('float32')
    #     Y_test_set = Y_test_set.astype('float32')
    #     X_train_val_set = X_train_val_set.astype('float32')
    #     X_test_set = X_test_set.astype('float32')

    #     X_train_val_set, X_test_set = torch.from_numpy(X_train_val_set), torch.from_numpy(X_test_set)
    #     Y_train_val_set, Y_test_set = torch.from_numpy(Y_train_val_set), torch.from_numpy(Y_test_set)
    #     Y_train_val_set = Y_train_val_set.unsqueeze(1)
    #     Y_test_set = Y_test_set.unsqueeze(1)
    
    # train_val_dataset = TensorDataset(X_train_val_set, Y_train_val_set)

    # test_dataset = TensorDataset(X_test_set, Y_test_set)
    # test_loader = DataLoader(test_dataset, batch_size=32,shuffle = True)

    return train_val_dataset, test_loader


def k_fold_trainer(dataset,model,args):
    """_summary_

    Args:
        dataset (_type_): _description_
        model (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Configuration options
    k_folds = 10
    num_epochs = args.epochs
    batch_size = args.batch_size

    loss_function = nn.BCELoss()
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, random_state=42, shuffle=True)
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        valloader = DataLoader(dataset,batch_size=batch_size, sampler=test_subsampler)

        # Init the neural network
        network = model
        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        for epoch in range(0, num_epochs):

            print(f'Starting epoch {epoch+1}')
            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
            
                inputs, targets = data[:-1], data[-1]
                # Zero the gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = network(inputs)

                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                # Print statistics
                current_loss += loss.item()
                if i % 2000 == 1999:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 2000))
                    current_loss = 0.0
            
            # Process is complete.
            # print('Training process has finished.')

            # print('Starting validation')

        # Evaluation for this fold
        with torch.no_grad():
            predictions, actuals = list(), list()
            for i, data in enumerate(valloader, 0):
                inputs, labels = data[:-1], data[-1]
                outputs = network(inputs)
                outputs = outputs.detach().numpy()

                # actual output
                actual = labels.numpy()
                actual = actual.reshape(len(actual), 1)
                # store the values in respective lists
                predictions.append(list(outputs))
                actuals.append(list(actual))

        actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
        predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]
        
        auc = roc_auc_score(y_true=actuals, y_score=predictions)

        # Print accuracy
        print(f'Accuracy for fold %d: %f' % (fold, auc))
        print('--------------------------------')
        results[fold] = auc
    
            # Saving the best model
        if results[fold] >= max(results.values()):
            save_path = os.path.join(ROOT_DIR, 'best_model_%s.pth' % args.model)
            torch.save(network.state_dict(), save_path)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum += value
    print(f'Average: {sum/len(results.items())}')

    return network


def evaluator(model,test_loader):
    """_summary_

    Args:
        model (_type_): _description_
        test_loader (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    predictions, actuals = list(), list()
    for i, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[:-1], data[-1]
        y_pred = model(inputs)
        y_pred = y_pred.detach().numpy()
        # pick the index of the highest values
        #res = np.argmax(y_pred, axis = 1) 

        # actual output
        actual = labels.numpy()
        actual = actual.reshape(len(actual), 1)
        # store the values in respective lists
        predictions.append(list(y_pred))
        actuals.append(list(actual))

    actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
    predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]

    return actuals, predictions