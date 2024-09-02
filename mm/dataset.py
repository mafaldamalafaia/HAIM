import numpy as np
import pandas as pd
import pickle
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torchxrayvision as xrv

# MIMICIV PATIENT CLASS STRUCTURE
class Patient_MM(object):
    def __init__(self, admissions, demographics, core,\
        labevents, prescriptions,\
        procedureevents, chartevents,\
        cxr, imcxr):
        
        ## CORE
        self.admissions = admissions
        self.demographics = demographics
        self.core = core
        ## HOSP
        self.labevents = labevents
        self.prescriptions = prescriptions
        ## ICU
        self.procedureevents = procedureevents
        self.chartevents = chartevents
        ## CXR
        self.cxr = cxr
        self.imcxr = imcxr

class HAIMDataset(Dataset):
    def __init__(self, info, data_dir='/export/scratch2/constellation-data/malafaia/physionet.org/files/haim-mm-mafi/', target='Consolidation'):
        self.target = target
        self.img_dir = os.path.join(data_dir, 'pickle')
        self.tab = pd.read_csv(os.path.join(data_dir, 'tabular_mm.csv'))
        self.ids = info # pre-selects samples
        self.samples = self._create_samples_list()

    def _create_samples_list(self):
        samples = []
        for i in range(len(self.ids)):
            # load pkl file
            pkl_id = self.ids['pkl_id'].iloc[i]
            tab_fts = self.tab.loc[self.tab['pkl_id'] == pkl_id]
            filename = f"{pkl_id:08d}" + '.pkl'
            filepath = os.path.join(self.img_dir, filename)
            with open(filepath, 'rb') as input:  
                stay = pickle.load(input)
            
            # get valid imgs and labels
            df_meta = stay.cxr
            admittime = stay.admissions.admittime.values[0]
            dischtime = stay.admissions.dischtime.values[0]
            df_meta = df_meta.loc[(df_meta['charttime'] >= admittime) & (df_meta['charttime'] <= dischtime)]
            df_meta = df_meta.loc[df_meta[self.target].isin([0, 1])]
            
            valid_cxr = [stay.imcxr[i] for i in df_meta.index]
            labels = df_meta[self.target].values
            for img, label in zip(valid_cxr, labels):
                samples.append((img, tab_fts, label))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, tab, y = self.samples[idx]
        
        # img: normalise and convert to tensor
        img = xrv.datasets.normalize(img, 255)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        # tab: drop cols and convert to tensor
        tab = tab.drop(columns=['pkl_id']).values
        tab = torch.tensor(tab, dtype=torch.float32).squeeze(0)
        
        # y: convert to tensor
        y = torch.tensor(y, dtype=torch.float32)
        
        return img, tab, y
