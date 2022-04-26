"""
    Dataloader, customized K-fold trainer & evaluater
"""
from unittest import TestLoader
from torch.utils.data import DataLoader
from itertools import chain
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import os
import torch_geometric.data
from torch_geometric.data import Batch
from itertools import cycle
from torch_geometric import data as DATA
import random
import pandas as pd
from torch.utils.data import Dataset,TensorDataset

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
            #input is tabular format
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
    test_loader = DataLoader(test_dataset, batch_size=256,shuffle = False)
        
    return train_val_dataset, test_loader


def dataloader_graph(*args,**kwargs):
    """
        should be starting from X, and Y
        X_{}_trainval, X_{}_test, Y_{}_trainval, Y_{}_test
    """
    temp_loader = {}
    for name, input in kwargs.items():
    # First, format
        temp_loader[name] = input

    # Second, combine trainval or test
    temp_loader_trainval,  temp_loader_test = [], []
    for key, val in temp_loader.items():
        load_name = key.split('_')
        # train_val_dataset
        if load_name[-1].endswith('trainval'):
            temp_loader_trainval.append(val)
        elif load_name[-1].endswith('test'):
            temp_loader_test.append(val)


    return temp_loader_trainval,temp_loader_test



def k_fold_trainer(dataset,model,args):

    # Configuration options
    k_folds = 5
    num_epochs = args.epochs
    batch_size = args.batch_size

    loss_function = nn.BCELoss()
    # For fold results
    results = {}

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



class MyDataset(TensorDataset):
    def __init__(self, trainval_df):
        super(MyDataset, self).__init__()
        self.df = trainval_df
        self.df.reset_index(drop=True, inplace=True)  # train_test_split之后，数据集的index混乱，需要reset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        #drug1, drug2, cell, target
        return (self.df.iloc[index,0], self.df.iloc[index,1], self.df.iloc[index,2], self.df.iloc[index,3])

def k_fold_trainer_graph(temp_loader_trainval,model,args):

    train_val_dataset_drug = temp_loader_trainval[0]
    train_val_dataset_drug2 = temp_loader_trainval[1]
    train_val_dataset_cell = temp_loader_trainval[2].tolist()
    train_val_dataset_target = temp_loader_trainval[3].tolist()

    # Configuration options
    k_folds = 5
    num_epochs = 10
    batch_size = 256

    loss_function = nn.BCELoss()
    # For fold results
    results = {}

    # Define the K-fold Cross Validator
    # skf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)

    # X,y = train_val_dataset_drug, []
    # for data_object in train_val_dataset_drug:
    #     labels = data_object.label
    #     y.append(labels)

    # skf.get_n_splits(X, y)
    kfold = KFold(n_splits=k_folds, random_state=42, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    # for fold, (train_ids, test_ids) in enumerate(skf.split(X,y)):
    
    trainval_df = [train_val_dataset_drug,train_val_dataset_drug2,train_val_dataset_cell,train_val_dataset_target]
    trainval_df = pd.DataFrame(trainval_df).T

    for fold, (train_ids, test_ids) in enumerate(kfold.split(trainval_df)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        #(Graph) For graph object needed to use torch_geometric.data.DataLoader
        if args.model == 'deepdds_wang':
            
            Dataset = MyDataset
            # self define dataset
            train_dataset = Dataset(trainval_df)
            
            trainloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_subsampler)
            valloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=test_subsampler)
        

        # Init the neural network
        network = model
        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        for epoch in range(0, num_epochs):

            print(f'Starting epoch {epoch+1}')
            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader):
                data1 = data[0]
                data2 = data[1]
                data_cell = data[2]
                data_target = data[3]
                x1, edge_index1, x2, edge_index2, cell, batch1, batch2 \
                    = data1.x, data1.edge_index, data2.x, data2.edge_index, data_cell, data1.batch, data2.batch

                targets = data_target.unsqueeze(1)
                                

                # Zero the gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = network(x1, edge_index1, x2, edge_index2, cell, batch1, batch2)

                loss = loss_function(outputs, targets.to(torch.float32))
                loss.backward()
                optimizer.step()
                # Print statistics
                current_loss += loss.item()
                if i % 100 == 99:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 100))
                    current_loss = 0.0
            
            # Process is complete.
            # print('Training process has finished.')

            # print('Starting validation')

        # Evaluation for this fold
        with torch.no_grad():
            predictions, actuals = list(), list()

            for i, data in enumerate(valloader):
                data1 = data[0]
                data2 = data[1]
                data_cell = data[2]
                data_target = data[3]
                x1, edge_index1, x2, edge_index2, cell, batch1, batch2 \
                    = data1.x, data1.edge_index, data2.x, data2.edge_index, data_cell, data1.batch, data2.batch

                targets = data_target

                # forward + backward + optimize
                outputs = network(x1, edge_index1, x2, edge_index2, cell, batch1, batch2)
                outputs = outputs.squeeze(1)
                outputs = outputs.detach().numpy()

                # actual output
                actual = targets.numpy()
                actual = actual.reshape(len(actual), 1)
                # store the values in respective lists
                predictions.append(list(outputs))
                actuals.append(list(actual))

        actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
        predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]
        try:
            auc = roc_auc_score(y_true=actuals, y_score=predictions)
        except ValueError:
            auc = 0

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

def evaluator_graph(model,temp_loader_test):
# For graph, the dataloader should be imported from torch geometric

    test_dataset_drug = temp_loader_test[0]
    test_dataset_drug2 = temp_loader_test[1]
    test_dataset_cell = temp_loader_test[2]
    test_dataset_target = temp_loader_test[3]

    test_df = [test_dataset_drug,test_dataset_drug2,test_dataset_cell,test_dataset_target]
    test_df = pd.DataFrame(test_df).T

    test_df = Dataset(test_df)
            
    test_loader = torch_geometric.data.DataLoader(test_df, batch_size=256,shuffle = False)

    predictions, actuals = list(), list()

    for i, data in enumerate(test_loader):
        
        data1 = data[0]
        data2 = data[1]
        data_cell = data[2]
        data_target = data[3]

        x1, edge_index1, x2, edge_index2, cell, batch1, batch2 \
            = data1.x, data1.edge_index, data2.x, data2.edge_index, data_cell, data1.batch, data2.batch


        y_pred = model(x1, edge_index1, x2, edge_index2, cell, batch1, batch2)
        y_pred = y_pred.detach().numpy()
        # pick the index of the highest values
        #res = np.argmax(y_pred, axis = 1) 

        # actual output
        actual = data_target.numpy()
        actual = actual.reshape(len(actual), 1)
        # store the values in respective lists
        predictions.append(list(y_pred))
        actuals.append(list(actual))

    actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
    predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]

    return actuals, predictions