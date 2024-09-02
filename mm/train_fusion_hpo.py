import numpy as np
import pandas as pd
import copy
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, balanced_accuracy_score

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import *

################################ HELPER METHODS ################################
################################################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def data_split(info):
    pkl_id = np.asarray(info['pkl_id'])
    label = np.asarray((info['y_ratio']>0.5).astype(int))
    id_train, id_test, y_train, _ = train_test_split(pkl_id, label, test_size=0.20, stratify=label)
    id_train, id_val, _, _ = train_test_split(id_train, y_train, test_size=0.20, stratify=y_train)
    
    return id_train, id_val, id_test

def get_auc(labels, probs):
    preds = probs
    preds = [int(i > .5) for i in preds]
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = auc(fpr, tpr)
    bacc = balanced_accuracy_score(labels, preds)
    
    return auroc, bacc
    
def train_epoch(device, model, optim, loss_fn, loader):
    """ train one epoch """
    model.train()
    train_loss = 0.
    for i, (img, tab, y) in enumerate(loader):
        img, tab, y = img.to(device), tab.to(device), y.to(device)
        optim.zero_grad()
        out = model(img, tab)
        loss = loss_fn(out, y.view(-1,1))
        loss.backward()
        optim.step()
        train_loss += loss.item()
    train_loss /= i + 1

    return train_loss

def val_test(device, model, loss_fn, loader):
    """ validate the model """
    model.eval()
    val_loss = 0.
    all_y, all_prob = [], []
    with torch.no_grad():
        for i, (img, tab, y) in enumerate(loader):
            img, tab, y = img.to(device), tab.to(device), y.to(device)
            out = model(img, tab)
            loss = loss_fn(out, y.view(-1,1))
            val_loss += loss.item()
            prob = out
            all_prob = np.append(all_prob, prob.cpu().numpy())
            all_y = np.append(all_y, y.cpu().numpy())

    val_loss /= i + 1
    auroc, bacc = get_auc(all_y, all_prob)

    return val_loss, auroc, bacc


################################################################################
################################################################################


################################# ARCHITECTURE #################################
################################################################################

class ActivationWithReshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = nn.ReLU()(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(x.size(0), -1)
        return x

class img_net(nn.Module):
    def __init__(self, z=64, IMG_FTS=1):
        super().__init__()
        
        # architecture "guide": 
        # https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/baseline_models/chexpert/model.py
        model = xrv.models.DenseNet(weights="densenet121-res224-chex")
        n_ftrs = model.classifier.in_features

        self.dense_fts = nn.Sequential(
            model.features,
            ActivationWithReshape(),
            #nn.Linear(n_ftrs, z),
            nn.ReLU())
        
        # add fc layer to predict single output
        self.classifier = nn.Sequential(
            nn.Linear(z, IMG_FTS),
            nn.Sigmoid()
        )
        
    def forward(self, x, tab):
        x = self.dense_fts(x) # to freeze
        x = self.classifier(x)
        
        return x

class tab_net(nn.Module):
    def __init__(self, z=64, TAB_FTS=1):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(15, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(128, z),
            nn.BatchNorm1d(z),
            nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(z,TAB_FTS),
            nn.Sigmoid())
    
    def forward(self, img, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x
    
class target_net(nn.Module):
    def __init__(self, fts):
        super().__init__()
        w = 128
        w_extra = 128

        self.fusion = nn.Sequential(
            nn.Linear(fts,w_extra),
            nn.BatchNorm1d(w_extra),
            nn.ReLU(),
            nn.Linear(w_extra, w),
            nn.BatchNorm1d(w),
            nn.ReLU(),
            nn.Linear(w,1),
            nn.Sigmoid()
        )
            
        
    def forward(self, x_img, x_tab):
        x = torch.cat((x_img,x_tab), dim=1)
        x = self.fusion(x)

        return x
    

class fusion_net(nn.Module):
    def __init__(self, config, seq=False):
        super().__init__()
        
        z = 64
            
        """ IMG """
        z_img = config['img_ftrs']
        self.cnn = img_net()
        self.cnn.classifier = nn.Sequential(
            nn.Linear(z, z_img)
        )
        
        """ TAB """
        z_tab = config['tab_ftrs']
        self.mlp = tab_net()
        self.mlp.classifier = nn.Sequential(
            nn.Linear(z, z_tab)
        )
        
        self.fusion = target_net(z_img+z_tab)
        
    def forward(self, x_img, x_tab):
        x_img = self.cnn(x_img, x_tab)
        x_tab = self.mlp(x_img, x_tab)
        
        x = self.fusion(x_img, x_tab)
        
        return x

################################################################################
################################################################################

nas = {"img_ftrs": 1, "tab_ftrs": 1}

TARGET = 'Consolidation'
info = pd.read_csv('/export/scratch2/constellation-data/malafaia/physionet.org/files/haim-mm-mafi/consolidation_ids.csv')
print("____________________Running experiments on ", TARGET, " prediction____________________")

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
print("device: ", device)

# change to hpo grid
lrs = [1e-3, 1e-4, 1e-5]
wds = [0.001, 0.0001, 0]

test_results = pd.DataFrame(columns=['LR', 'WD', 'Fold', 'BCEloss', 'AUROC', 'BAcc', 'model'])

# iterate over lrs
for j in range(len(lrs)):
    LR = lrs[j]
    # iterate over wds
    for k in range(len(wds)):
        WD = wds[k]
        print("##### LR = ", LR, "; WD = ", WD, " #####")
        # iterate over each train-val test split
        for i in range(5): # 5-fold CV
            train, val, test = data_split(info)
            train_ids= info.loc[info['pkl_id'].isin(train)]
            val_ids = info.loc[info['pkl_id'].isin(val)]
            test_ids = info.loc[info['pkl_id'].isin(test)]
            
            train_dataset = HAIMDataset(train_ids)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_dataset = HAIMDataset(val_ids)
            val_loader = DataLoader(val_dataset)
            test_dataset = HAIMDataset(test_ids)
            test_loader = DataLoader(test_dataset)
            
            model = fusion_net(nas, True)
            model.to(device)
            # freeze ft engineering blocks
            tab_wts = torch.load('/export/scratch1/home/malafaia/HAIM/mm/saved_models/tab_' + str(i) + '.pth')
            mlp_wts = {k: v for k, v in tab_wts.items() if not k.startswith('classifier')}
            model.mlp.load_state_dict(mlp_wts, strict=False)
            for param in model.cnn.dense_fts.parameters():
                param.requires_grad = False
            for param in model.mlp.fc1.parameters():
                param.requires_grad = False
            for param in model.mlp.fc2.parameters():
                param.requires_grad = False
            
            loss_fn = nn.BCELoss()
            optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            
            patience = 10
            no_improvement = 0
            best_loss = np.inf
            best_model_wts = copy.deepcopy(model.state_dict())
            
            for epoch in range(100):
                # train
                train_loss = train_epoch(device, model, optim, loss_fn, train_loader)
                val_loss, _, _ = val_test(device, model, loss_fn, val_loader)
                print("#{}: train loss = {:.5f}; val loss = {:.5f}".format(epoch, train_loss, val_loss))
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    no_improvement = 0
                else: no_improvement += 1
                if no_improvement >= patience: break # early stopping
                
            model.load_state_dict(best_model_wts) # load best model
            
            test_loss, test_auc, test_acc = val_test(device, model, loss_fn, test_loader)
            print("*** RUN {} ***\n   *Test Loss={:.5f}\n   *AUC={:.5f}\n   *BAcc={:.5f}".format(i, test_loss, test_auc, test_acc))
            new_row = {'LR': LR, 'WD': WD ,'Fold': i, 'BCELoss': test_loss, 'AUROC': test_auc, 'BAcc': test_acc, 'model': model}
            test_results = test_results.append(new_row, ignore_index=True)

# test_results
# models
# save 5 models from best LR and WD
grouped_results = test_results.groupby(['LR','WD'])['BCELoss'].mean().reset_index()
best_params = grouped_results.loc[grouped_results['BCELoss'].idxmin()]
print("BEST PARAMETERS: ", best_params)
best_lr = best_params['LR']
best_wd = best_params['WD']

# get df with best hyperparameters
best_models_df = test_results[(test_results['LR']==best_lr) & (test_results['WD']==best_wd)]

save_dir = '/export/scratch1/home/malafaia/HAIM/mm/saved_models/fusion_'
# save each model
for idx, row in best_models_df.iterrows():
    print("model no. ", idx)
    print("* BCELoss: ", row['BCELoss'])
    print("* AUROC: ", row['AUROC'])
    print("* BAcc: ", row['BAcc'])
    print()
    model = row['model']
    fold = row['Fold']
    filename = save_dir + str(fold) + '.pth'
    torch.save(model.state_dict(), filename)      