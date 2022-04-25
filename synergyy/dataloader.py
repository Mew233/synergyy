"""
    Dataloader, customized K-fold trainer & evaluater
"""
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
from itertools import cycle

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
    # iterate to create the GCNDATA object

    def processDATA(temp_loader_trainval, drug = 'drug1'):
        train_val_dataset = []
        test_dataset = []
        
        ##combine graph, cell, lable into one torch.geometric.Data
        if drug =='drug1':
            X_deepdds_sm_drug1_trainval = temp_loader_trainval[0]
        elif drug =='drug2':
            X_deepdds_sm_drug1_trainval = temp_loader_trainval[1]
        X_cell_trainval = temp_loader_trainval[2]
        Y_trainval = temp_loader_trainval[3]

        for i in range(len(X_deepdds_sm_drug1_trainval)):
            labels = Y_trainval[i]
            new_cell = X_cell_trainval[i]
            X_deepdds_sm_drug1_trainval[i].cell = new_cell
            X_deepdds_sm_drug1_trainval[i].label = labels
            train_val_dataset.append(X_deepdds_sm_drug1_trainval[i])

        return train_val_dataset



    train_val_dataset_drug = processDATA(temp_loader_trainval, drug ='drug1')
    train_val_dataset_drug2 = processDATA(temp_loader_trainval, drug ='drug2')

    test_dataset_drug = processDATA(temp_loader_test, drug ='drug1')
    test_dataset_drug2 = processDATA(temp_loader_test, drug ='drug1')
    test_loader_drug = torch_geometric.data.DataLoader(test_dataset_drug, batch_size=256,shuffle = False)
    test_loader_drug2 = torch_geometric.data.DataLoader(test_dataset_drug2, batch_size=256,shuffle = False)
        
    return train_val_dataset_drug,train_val_dataset_drug2,test_loader_drug,test_loader_drug2


def k_fold_trainer(dataset,model,args):

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

def k_fold_trainer_graph(train_val_dataset_drug,train_val_dataset_drug2,model,args):

    # Configuration options
    k_folds = 10
    num_epochs = 1
    batch_size = 256

    loss_function = nn.BCELoss()
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)
    # Define the K-fold Cross Validator
    skf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)

    X,y = train_val_dataset_drug, []
    for data_object in train_val_dataset_drug:
        labels = data_object.label
        y.append(labels)

    skf.get_n_splits(X, y)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(skf.split(X,y)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        from torch_geometric.loader import HGTLoader
        #(Graph) For graph object needed to use torch_geometric.data.DataLoader
        if args.model == 'deepdds_wang':
            trainloader = torch_geometric.data.DataLoader(train_val_dataset_drug, batch_size=256, sampler=train_subsampler,shuffle = None)
            trainloader2 = torch_geometric.data.DataLoader(train_val_dataset_drug2, batch_size=256, sampler=train_subsampler,shuffle = None)

            valloader = torch_geometric.data.DataLoader(train_val_dataset_drug,batch_size=256, sampler=test_subsampler,shuffle = None)
            valloader2 = torch_geometric.data.DataLoader(train_val_dataset_drug2,batch_size=256, sampler=test_subsampler,shuffle = None)


        # Init the neural network
        network = model
        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        for epoch in range(0, num_epochs):

            print(f'Starting epoch {epoch+1}')
            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(zip(cycle(trainloader), trainloader2)):
                data1 = data[0]
                data2 = data[1]
                x1, edge_index1, x2, edge_index2, cell, batch1, batch2 \
                    = data1.x, data1.edge_index, data2.x, data2.edge_index, data1.cell, data1.batch, data2.batch

                targets = torch.FloatTensor(data1.label)
                # targets = targets.unsqueeze(1)

                # Zero the gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = network(x1, edge_index1, x2, edge_index2, cell, batch1, batch2)
                outputs = outputs.squeeze(1)

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

            for i, data in enumerate(zip(cycle(valloader), valloader2)):
                data1 = data[0]
                data2 = data[1]
                x1, edge_index1, x2, edge_index2, cell, batch1, batch2 \
                    = data1.x, data1.edge_index, data2.x, data2.edge_index, data1.cell, data1.batch, data2.batch

                targets = torch.FloatTensor(data1.label)

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

def evaluator_graph(model,test_loader_drug,test_loader_drug2):
# For graph, we should have batch set as 1

    predictions, actuals = list(), list()
    for i, data in enumerate(zip(cycle(test_loader_drug), test_loader_drug2)):
        
        data1 = data[0]
        data2 = data[1]
        x1, edge_index1, x2, edge_index2, cell, batch1, batch2 \
            = data1.x, data1.edge_index, data2.x, data2.edge_index, data1.cell, data1.batch, data2.batch


        y_pred = model(x1, edge_index1, x2, edge_index2, cell, batch1, batch2)
        y_pred = y_pred.detach().numpy()
        # pick the index of the highest values
        #res = np.argmax(y_pred, axis = 1) 

        # actual output
        actual = torch.FloatTensor(data1.label).numpy()
        actual = actual.reshape(len(actual), 1)
        # store the values in respective lists
        predictions.append(list(y_pred))
        actuals.append(list(actual))

    actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
    predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]

    return actuals, predictions