# %%

###########################################################################################################
#                                  __    __       ___       __  .___  ___. 
#                                 |  |  |  |     /   \     |  | |   \/   | 
#                                 |  |__|  |    /  ^  \    |  | |  \  /  | 
#                                 |   __   |   /  /_\  \   |  | |  |\/|  | 
#                                 |  |  |  |  /  _____  \  |  | |  |  |  | 
#                                 |__|  |__| /__/     \__\ |__| |__|  |__| 
#
#                               HOLISTIC ARTIFICIAL INTELLIGENCE IN MEDICINE
#
###########################################################################################################
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                              IMPORTS                                            |

# System                                                                                           
import os
import sys

# Base
import cv2
import math
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm

from dask import dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()

# Core AI/ML
#import tensorflow as tf
import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
from torch.utils.data import Dataset, DataLoader

# Scipy
from scipy.stats import ks_2samp
from scipy.signal import find_peaks

# Scikit-learn
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

# Computer Vision
import cv2
import skimage, skimage.io
import torchxrayvision as xrv

# Warning handling
import warnings
warnings.filterwarnings("ignore")

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                Initializations & Data Loading                                   |
#                                                                                                 | 
"""
Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)
"""

# Define MIMIC IV Data Location
core_mimiciv_path = '/export/scratch2/constellation-data/malafaia/physionet.org/files/mimiciv/1.0/'

# Define MIMIC IV Image Data Location (usually external drive)
core_mimiciv_imgcxr_path = '/export/scratch2/constellation-data/malafaia/physionet.org/files/mimic-cxr-jpg/2.0.0.'

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                     HAIM-MIMICIV specific Patient Representation Function                       |
#                                                                                                 |

# MIMICIV PATIENT CLASS STRUCTURE
class Patient_ICU(object):
    def __init__(self, admissions, demographics, transfers, core,\
        diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents,\
        labevents, microbiologyevents, poe, poe_detail,\
        prescriptions, procedures_icd, services, procedureevents,\
        outputevents, inputevents, icustays, datetimeevents,\
        cxr, imcxr):#, chartevents, noteevents, dsnotes, ecgnotes, \
        #echonotes, radnotes):
        
        ## CORE
        self.admissions = admissions
        self.demographics = demographics
        self.transfers = transfers
        self.core = core
        ## HOSP
        self.diagnoses_icd = diagnoses_icd
        self.drgcodes = drgcodes
        self.emar = emar
        self.emar_detail = emar_detail
        self.hcpcsevents = hcpcsevents
        self.labevents = labevents
        self.microbiologyevents = microbiologyevents
        self.poe = poe
        self.poe_detail = poe_detail
        self.prescriptions = prescriptions
        self.procedures_icd = procedures_icd
        self.services = services
        ## ICU
        self.procedureevents = procedureevents
        self.outputevents = outputevents
        self.inputevents = inputevents
        self.icustays = icustays
        self.datetimeevents = datetimeevents
        #self.chartevents = chartevents
        ## CXR
        self.cxr = cxr
        self.imcxr = imcxr


# GET FULL MIMIC IV PATIENT RECORD USING DATABASE KEYS
def get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id):
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   key_hadm_id    -> hadm_id is unique to a patient hospital stay
    #   key_stay_id    -> stay_id is unique to a patient ward stay
    #   
    #   NOTES: Identifiers which specify the patient. More information about 
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> ICU patient stay structure

    #-> FILTER data
    ##-> CORE
    f_df_base_core = df_base_core[(df_base_core.subject_id == key_subject_id) & (df_base_core.hadm_id == key_hadm_id)]
    f_df_admissions = df_admissions[(df_admissions.subject_id == key_subject_id) & (df_admissions.hadm_id == key_hadm_id)]
    f_df_patients = df_patients[(df_patients.subject_id == key_subject_id)]
    f_df_transfers = df_transfers[(df_transfers.subject_id == key_subject_id) & (df_transfers.hadm_id == key_hadm_id)]
    ###-> Merge data into single patient structure
    f_df_core = f_df_base_core
    f_df_core = f_df_core.merge(f_df_admissions, how='left')
    f_df_core = f_df_core.merge(f_df_patients, how='left')
    f_df_core = f_df_core.merge(f_df_transfers, how='left')

    ##-> HOSP
    f_df_diagnoses_icd = df_diagnoses_icd[(df_diagnoses_icd.subject_id == key_subject_id)]
    f_df_drgcodes = df_drgcodes[(df_drgcodes.subject_id == key_subject_id) & (df_drgcodes.hadm_id == key_hadm_id)]
    f_df_emar = df_emar[(df_emar.subject_id == key_subject_id) & (df_emar.hadm_id == key_hadm_id)]
    f_df_emar_detail = df_emar_detail[(df_emar_detail.subject_id == key_subject_id)]
    f_df_hcpcsevents = df_hcpcsevents[(df_hcpcsevents.subject_id == key_subject_id) & (df_hcpcsevents.hadm_id == key_hadm_id)]
    f_df_labevents = df_labevents[(df_labevents.subject_id == key_subject_id) & (df_labevents.hadm_id == key_hadm_id)]
    f_df_microbiologyevents = df_microbiologyevents[(df_microbiologyevents.subject_id == key_subject_id) & (df_microbiologyevents.hadm_id == key_hadm_id)]
    f_df_poe = df_poe[(df_poe.subject_id == key_subject_id) & (df_poe.hadm_id == key_hadm_id)]
    f_df_poe_detail = df_poe_detail[(df_poe_detail.subject_id == key_subject_id)]
    f_df_prescriptions = df_prescriptions[(df_prescriptions.subject_id == key_subject_id) & (df_prescriptions.hadm_id == key_hadm_id)]
    f_df_procedures_icd = df_procedures_icd[(df_procedures_icd.subject_id == key_subject_id) & (df_procedures_icd.hadm_id == key_hadm_id)]
    f_df_services = df_services[(df_services.subject_id == key_subject_id) & (df_services.hadm_id == key_hadm_id)]
    ###-> Merge content from dictionaries
    f_df_diagnoses_icd = f_df_diagnoses_icd.merge(df_d_icd_diagnoses, how='left') 
    f_df_procedures_icd = f_df_procedures_icd.merge(df_d_icd_procedures, how='left')
    f_df_hcpcsevents = f_df_hcpcsevents.merge(df_d_hcpcs, how='left')
    f_df_labevents = f_df_labevents.merge(df_d_labitems, how='left')

    ##-> ICU
    f_df_procedureevents = df_procedureevents[(df_procedureevents.subject_id == key_subject_id) & (df_procedureevents.hadm_id == key_hadm_id) & (df_procedureevents.stay_id == key_stay_id)]
    f_df_outputevents = df_outputevents[(df_outputevents.subject_id == key_subject_id) & (df_outputevents.hadm_id == key_hadm_id) & (df_outputevents.stay_id == key_stay_id)]
    f_df_inputevents = df_inputevents[(df_inputevents.subject_id == key_subject_id) & (df_inputevents.hadm_id == key_hadm_id) & (df_inputevents.stay_id == key_stay_id)]
    f_df_icustays = df_icustays[(df_icustays.subject_id == key_subject_id) & (df_icustays.hadm_id == key_hadm_id) & (df_icustays.stay_id == key_stay_id)]
    f_df_datetimeevents = df_datetimeevents[(df_datetimeevents.subject_id == key_subject_id) & (df_datetimeevents.hadm_id == key_hadm_id) & (df_datetimeevents.stay_id == key_stay_id)]
    f_df_chartevents = df_chartevents[(df_chartevents.subject_id == key_subject_id) & (df_chartevents.hadm_id == key_hadm_id) & (df_chartevents.stay_id == key_stay_id)]
    ###-> Merge content from dictionaries
    f_df_procedureevents = f_df_procedureevents.merge(df_d_items, how='left')
    f_df_outputevents = f_df_outputevents.merge(df_d_items, how='left')
    f_df_inputevents = f_df_inputevents.merge(df_d_items, how='left')
    f_df_datetimeevents = f_df_datetimeevents.merge(df_d_items, how='left')
    f_df_chartevents = f_df_chartevents.merge(df_d_items, how='left')       

    ##-> CXR
    f_df_mimic_cxr_split = df_mimic_cxr_split[(df_mimic_cxr_split.subject_id == key_subject_id)]
    f_df_mimic_cxr_chexpert = df_mimic_cxr_chexpert[(df_mimic_cxr_chexpert.subject_id == key_subject_id)]
    f_df_mimic_cxr_metadata = df_mimic_cxr_metadata[(df_mimic_cxr_metadata.subject_id == key_subject_id)]
    f_df_mimic_cxr_negbio = df_mimic_cxr_negbio[(df_mimic_cxr_negbio.subject_id == key_subject_id)]
    ###-> Merge data into single patient structure
    f_df_cxr = f_df_mimic_cxr_split
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_chexpert, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_metadata, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_negbio, how='left')
    ###-> Get images of that timebound patient
    f_df_imcxr = []
    for img_idx, img_row in f_df_cxr.iterrows():
        img_folder = 'p' + str(img_row['subject_id'])[:2]
        img_id = 'p' + str(int(img_row['subject_id']))
        img_study = 's' + str(int(img_row['study_id']))
        img_name = str(img_row['dicom_id']) + '.jpg'
        img_path = core_mimiciv_imgcxr_path + 'files/' + img_folder + '/' + img_id + '/' + img_study + '/' + img_name
        #img_path = core_mimiciv_imgcxr_path + str(img_row['Img_Folder']) + '/' + str(img_row['Img_Filename'])
        img_cxr_shape = [224, 224]
        img_load = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_load is not None and img_load.size != 0:
            img_cxr = cv2.resize(img_load, (img_cxr_shape[0], img_cxr_shape[1]))
            f_df_imcxr.append(np.array(img_cxr))
        else: print("IMAGE IS EMPTY in patient ", img_path)
      
    ##-> NOTES
    f_df_noteevents = df_noteevents[(df_noteevents.subject_id == key_subject_id) & (df_noteevents.hadm_id == key_hadm_id)]
    f_df_dsnotes = df_dsnotes[(df_dsnotes.subject_id == key_subject_id) & (df_dsnotes.hadm_id == key_hadm_id) & (df_dsnotes.stay_id == key_stay_id)]
    f_df_ecgnotes = df_ecgnotes[(df_ecgnotes.subject_id == key_subject_id) & (df_ecgnotes.hadm_id == key_hadm_id) & (df_ecgnotes.stay_id == key_stay_id)]
    f_df_echonotes = df_echonotes[(df_echonotes.subject_id == key_subject_id) & (df_echonotes.hadm_id == key_hadm_id) & (df_echonotes.stay_id == key_stay_id)]
    f_df_radnotes = df_radnotes[(df_radnotes.subject_id == key_subject_id) & (df_radnotes.hadm_id == key_hadm_id) & (df_radnotes.stay_id == key_stay_id)]

    ###-> Merge data into single patient structure
    #--None


    # -> Create & Populate patient structure
    ## CORE
    admissions = f_df_admissions
    demographics = f_df_patients
    transfers = f_df_transfers
    core = f_df_core

    ## HOSP
    diagnoses_icd = f_df_diagnoses_icd
    drgcodes = f_df_diagnoses_icd
    emar = f_df_emar
    emar_detail = f_df_emar_detail
    hcpcsevents = f_df_hcpcsevents
    labevents = f_df_labevents
    microbiologyevents = f_df_microbiologyevents
    poe = f_df_poe
    poe_detail = f_df_poe_detail
    prescriptions = f_df_prescriptions
    procedures_icd = f_df_procedures_icd
    services = f_df_services

    ## ICU
    procedureevents = f_df_procedureevents
    outputevents = f_df_outputevents
    inputevents = f_df_inputevents
    icustays = f_df_icustays
    datetimeevents = f_df_datetimeevents
    chartevents = f_df_chartevents

    ## CXR
    cxr = f_df_cxr 
    imcxr = f_df_imcxr

    ## NOTES
    noteevents = f_df_noteevents
    dsnotes = f_df_dsnotes
    ecgnotes = f_df_ecgnotes
    echonotes = f_df_echonotes
    radnotes = f_df_radnotes


    # Create patient object and return
    Patient_ICUstay = Patient_ICU(admissions, demographics, transfers, core, \
                                  diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents, \
                                  labevents, microbiologyevents, poe, poe_detail, \
                                  prescriptions, procedures_icd, services, procedureevents, \
                                  outputevents, inputevents, icustays, datetimeevents, \
                                  cxr, imcxr)#, chartevents, noteevents, dsnotes, ecgnotes, \
                                  #echonotes, radnotes)

    return Patient_ICUstay


# DELTA TIME CALCULATOR FROM TWO TIMESTAMPS
def date_diff_hrs(t1, t0):
    # Inputs:
    #   t1 -> Final timestamp in a patient hospital stay
    #   t0 -> Initial timestamp in a patient hospital stay

    # Outputs:
    #   delta_t -> Patient stay structure bounded by allowed timestamps

    try:
        delta_t = (t1-t0).total_seconds()/3600 # Result in hrs
    except:
        delta_t = math.nan
    
    return delta_t


# GET TIMEBOUND MIMIC-IV PATIENT RECORD BY DATABASE KEYS AND TIMESTAMPS
def get_timebound_patient_icustay(Patient_ICUstay, start_hr = None, end_hr = None):
    # Inputs:
    #   Patient_ICUstay -> Patient ICU stay structure
    #   start_hr -> start_hr indicates the first valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #   end_hr -> end_hr indicates the last valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #
    #   NOTES: Identifiers which specify the patient. More information about 
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    
    # %% EXAMPLE OF USE
    ## Let's select a single patient
    '''
    key_subject_id = 10000032
    key_hadm_id = 29079034
    key_stay_id = 39553978
    start_hr = 0
    end_hr = 24
    patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    '''
    
    # Create a deep copy so that it is not the same object
    # Patient_ICUstay = copy.deepcopy(Patient_ICUstay)
    
    
    ## --> Process Event Structure Calculations
    admittime = Patient_ICUstay.core['admittime'].values[0]
    dischtime = Patient_ICUstay.core['dischtime'].values[0]
    Patient_ICUstay.emar['deltacharttime'] = Patient_ICUstay.emar.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.labevents['deltacharttime'] = Patient_ICUstay.labevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.microbiologyevents['deltacharttime'] = Patient_ICUstay.microbiologyevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.outputevents['deltacharttime'] = Patient_ICUstay.outputevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.datetimeevents['deltacharttime'] = Patient_ICUstay.datetimeevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    #Patient_ICUstay.chartevents['deltacharttime'] = Patient_ICUstay.chartevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    #Patient_ICUstay.noteevents['deltacharttime'] = Patient_ICUstay.noteevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    #Patient_ICUstay.dsnotes['deltacharttime'] = Patient_ICUstay.dsnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    #Patient_ICUstay.ecgnotes['deltacharttime'] = Patient_ICUstay.ecgnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    #Patient_ICUstay.echonotes['deltacharttime'] = Patient_ICUstay.echonotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    #Patient_ICUstay.radnotes['deltacharttime'] = Patient_ICUstay.radnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    
    # Re-calculate times of CXR database
    Patient_ICUstay.cxr['StudyDateForm'] = pd.to_datetime(Patient_ICUstay.cxr['StudyDate'], format='%Y%m%d')
    Patient_ICUstay.cxr['StudyTimeForm'] = Patient_ICUstay.cxr.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
    Patient_ICUstay.cxr['StudyTimeForm'] = pd.to_datetime(Patient_ICUstay.cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time
    Patient_ICUstay.cxr['charttime'] = Patient_ICUstay.cxr.apply(lambda r : dt.datetime.combine(r['StudyDateForm'],r['StudyTimeForm']),1)
    Patient_ICUstay.cxr['charttime'] = Patient_ICUstay.cxr['charttime'].dt.floor('Min')
    Patient_ICUstay.cxr['deltacharttime'] = Patient_ICUstay.cxr.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    
    ## --> Filter by allowable time stamps
    if not (start_hr == None):
        Patient_ICUstay.emar = Patient_ICUstay.emar[(Patient_ICUstay.emar.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.emar.deltacharttime)]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[(Patient_ICUstay.labevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.labevents.deltacharttime)]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[(Patient_ICUstay.microbiologyevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[(Patient_ICUstay.outputevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[(Patient_ICUstay.datetimeevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)]
        #Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[(Patient_ICUstay.chartevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)]
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)]
        Patient_ICUstay.imcxr = [Patient_ICUstay.imcxr[i] for i, x in enumerate((Patient_ICUstay.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)) if x]
        #Notes
        #Patient_ICUstay.noteevents = Patient_ICUstay.noteevents[(Patient_ICUstay.noteevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.noteevents.deltacharttime)]
        #Patient_ICUstay.dsnotes = Patient_ICUstay.dsnotes[(Patient_ICUstay.dsnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.dsnotes.deltacharttime)]
        #Patient_ICUstay.ecgnotes = Patient_ICUstay.ecgnotes[(Patient_ICUstay.ecgnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.ecgnotes.deltacharttime)]
        #Patient_ICUstay.echonotes = Patient_ICUstay.echonotes[(Patient_ICUstay.echonotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.echonotes.deltacharttime)]
        #Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]
        
        
    if not (end_hr == None):
        Patient_ICUstay.emar = Patient_ICUstay.emar[(Patient_ICUstay.emar.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.emar.deltacharttime)]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[(Patient_ICUstay.labevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.labevents.deltacharttime)]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[(Patient_ICUstay.microbiologyevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[(Patient_ICUstay.outputevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[(Patient_ICUstay.datetimeevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)]
        #Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[(Patient_ICUstay.chartevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)]
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)]
        Patient_ICUstay.imcxr = [Patient_ICUstay.imcxr[i] for i, x in enumerate((Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)) if x]
        #Notes
        #Patient_ICUstay.noteevents = Patient_ICUstay.noteevents[(Patient_ICUstay.noteevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.noteevents.deltacharttime)]
        #Patient_ICUstay.dsnotes = Patient_ICUstay.dsnotes[(Patient_ICUstay.dsnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.dsnotes.deltacharttime)]
        #Patient_ICUstay.ecgnotes = Patient_ICUstay.ecgnotes[(Patient_ICUstay.ecgnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.ecgnotes.deltacharttime)]
        #Patient_ICUstay.echonotes = Patient_ICUstay.echonotes[(Patient_ICUstay.echonotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.echonotes.deltacharttime)]
        #Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]
        
        # Filter CXR to match allowable patient stay
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.charttime <= dischtime)]
    
    return Patient_ICUstay


# LOAD MASTER DICTIONARY OF MIMIC IV EVENTS
def load_haim_event_dictionaries(core_mimiciv_path):
    # Inputs:
    #   df_d_items -> MIMIC chartevent items dictionary
    #   df_d_labitems -> MIMIC labevent items dictionary
    #   df_d_hcpcs -> MIMIC hcpcs items dictionary
    #
    # Outputs:
    #   df_patientevents_categorylabels_dict -> Dictionary with all possible event types

    # Generate dictionary for chartevents, labevents and HCPCS
    df_patientevents_categorylabels_dict = pd.DataFrame(columns = ['eventtype', 'category', 'label'])
  
    # Load dictionaries
    df_d_items = pd.read_csv(core_mimiciv_path + 'icu/d_items.csv.gz')
    df_d_labitems = pd.read_csv(core_mimiciv_path + 'hosp/d_labitems.csv')
    df_d_hcpcs = pd.read_csv(core_mimiciv_path + 'hosp/d_hcpcs.csv')

    # Get Chartevent items with labels & category
    df = df_d_items
    for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
        #print(category)
        category_list = df[df['category']==category]
        for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
            df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'chart', 'category': category, 'label': item}, ignore_index=True)
      
    # Get Lab items with labels & category
    df = df_d_labitems
    for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
        #print(category)
        category_list = df[df['category']==category]
        for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
            df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'lab', 'category': category, 'label': item}, ignore_index=True)
          
    # Get HCPCS items with labels & category
    df = df_d_hcpcs
    for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
        #print(category)
        category_list = df[df['category']==category]
        for item_idx, item in enumerate(sorted(category_list.long_description.astype(str).unique())):
            df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'hcpcs', 'category': category, 'label': item}, ignore_index=True)
            

    return df_patientevents_categorylabels_dict



#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                            Data filtering by condition and outcome                              |
#                                                                                                 | 
"""
Resources to identify tables and variables of interest can be found in the MIMIC-IV official API 
(https://mimic-iv.mit.edu/docs/)
"""

# QUERY IN ALL SINGLE PATIENT ICU STAY RECORD FOR KEYWORD MATCHING
def is_haim_patient_keyword_match(patient, keywords, verbose = 0):
    # Inputs:
    #   patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   keywords -> List of string keywords to attempt to match in an "OR" basis
    #   verbose -> Flag to print found keyword outputs (0,1,2)
    #
    # Outputs:
    #   is_key -> Boolean flag indicating if any of the input Keywords are present
    #   keyword_mask -> Array indicating which of the input Keywords are present (0-Absent, 1-Present)
  
    # Retrieve list of all the contents of patient datastructures
    patient_dfs_list = [## CORE
                        patient.core,
                        ## HOSP
                        patient.diagnoses_icd,
                        patient.drgcodes,
                        patient.emar,
                        patient.emar_detail,
                        patient.hcpcsevents,
                        patient.labevents,
                        patient.microbiologyevents,
                        patient.poe,
                        patient.poe_detail,
                        patient.prescriptions,
                        patient.procedures_icd,
                        patient.services,
                        ## ICU
                        patient.procedureevents,
                        patient.outputevents,
                        patient.inputevents,
                        patient.icustays,
                        patient.datetimeevents,
                        #patient.chartevents,
                        ## CXR
                        patient.cxr,
                        patient.imcxr,
                        ## NOTES
                        #patient.noteevents,
                        #patient.dsnotes,
                        #patient.ecgnotes,
                        #patient.echonotes,
                        #patient.radnotes
                        ]

    patient_dfs_dict = ['core', 'diagnoses_icd', 'drgcodes', 'emar', 'emar_detail', 'hcpcsevents', 'labevents', 'microbiologyevents', 'poe',
                        'poe_detail', 'prescriptions', 'procedures_icd', 'services', 'procedureevents', 'outputevents', 'inputevents', 'icustays',
                        'datetimeevents', 'cxr', 'imcxr']#, 'chartevents', 'noteevents', 'dsnotes', 'ecgnotes', 'echonotes', 'radnotes']
  
    # Initialize query mask
    keyword_mask = np.zeros([len(patient_dfs_list), len(keywords)])
    for idx_df, patient_df in enumerate(patient_dfs_list):
        for idx_keyword, keyword in enumerate(keywords):
            try:
                patient_df_text = patient_df.astype(str)
                is_df_key = patient_df_text.sum(axis=1).str.contains(keyword, case=False).any()

                if is_df_key:
                    keyword_mask[idx_df, idx_keyword]=1
                    if (verbose >= 2):
                        print('')
                        print('Keyword: ' + '"' + keyword + ' " ' +  '(Found in "' + patient_dfs_dict[idx_df] + '" table )')
                        print(patient_df_text)
                else:
                    keyword_mask[idx_df, idx_keyword]=0
              
            except:
                is_df_key = False
                keyword_mask[idx_df, idx_keyword]=0

    # Create final keyword mask
    if keyword_mask.any():
        is_key = True
    else:
        is_key = False
    
    return is_key, keyword_mask


# QUERY IN ALL SINGLE PATIENT ICU STAY RECORD FOR INCLUSION CRITERIA MATCHING
def is_haim_patient_inclusion_criteria_match(patient, inclusion_criteria, verbose = 0):
    # Inputs:
    #   patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   inclusion_criteria -> Inclusion criteria in groups of keywords. 
    #                         Keywords in groups are follow and "OR" logic,
    #                         while an "AND" logic is stablished among groups
    #   verbose -> Flag to print found keyword outputs (0,1,2)
    #
    # Outputs:
    #   is_included -> Boolean flag if inclusion criteria is found in patient
    #   inclusion_criteria_mask -> Binary mask of inclusion criteria found in patient
  
    # Clean out process bar before starting
    inclusion_criteria_mask = np.zeros(len(inclusion_criteria))
    for idx_keywords, keywords in enumerate(inclusion_criteria):
        is_included_flag, _ = is_haim_patient_keyword_match(patient, keywords, verbose)
        inclusion_criteria_mask[idx_keywords] = is_included_flag
    
    if inclusion_criteria_mask.all():
        is_included = True
    else:
        is_included = False

    # Print if patient has to be included
    if (verbose >=2):
        print('')
        print('Inclusion Criteria: ' + str(inclusion_criteria))
        print('Inclusion Vector: ' + str(inclusion_criteria_mask) + ' , To include: ' + str(is_included))
    
    return is_included, inclusion_criteria_mask



# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
def search_key_mimiciv_patients(df_haim_ids, core_mimiciv_path, inclusion_criteria, verbose = 0):
    # Inputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   core_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #
    # Outputs:
    #   nfiles -> Number of single patient HAIM files produced

    # Clean out process bar before starting
    sys.stdout.flush()

    # List of key patients
    key_haim_patient_ids = []

    # Extract information for patient
    nfiles = len(df_haim_ids)
    with tqdm(total = nfiles) as pbar:
        # Update process bar
        nbase= 0
        pbar.update(nbase)
        #Iterate through all patients
        for haim_patient_idx in range(nbase, nfiles):
            #Load precomputed patient file
            filename = f"{haim_patient_idx:08d}" + '.pkl'
            patient = load_patient_object(core_mimiciv_path + 'pickle/' + filename)
            #Check if patient fits keywords
            is_key, _ = is_haim_patient_inclusion_criteria_match(patient, keywords, verbose)
            if is_key:
                key_haim_patient_ids.append(haim_patient_idx)

            # Update process bar
            pbar.update(1)

    return key_haim_patient_ids



# GET MIMIC IV PATIENT LIST FILTERED BY DESIRED CONDITION
def get_haim_ids_only_by_condition(condition_tokens, core_mimiciv_path):
    # Inputs:
    #   condition_tokens     -> string identifier of the condition you want to isolate (condition_tokens= ['heart failure','chronic'])
    #   outcome_tokens       -> string identifier of the outcome you want to isolate
    #   core_mimiciv_path    -> path to folder where the base MIMIC IV dataset files are located
  
    # Outputs:
    #   condition_outcome_df -> Dataframe including patients IDs with desired Condition, indicating the Outcome.
  
    # Load necessary ICD diagnostic lists and general patient information
    d_icd_diagnoses = pd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv')
    d_icd_diagnoses['long_title'] = d_icd_diagnoses['long_title'].str.lower()
    diagnoses_icd['icd_code'] = diagnoses_icd['icd_code'].str.replace(' ', '')
  
    admissions = pd.read_csv(core_mimiciv_path + 'core/admissions.csv')
    patients = pd.read_csv(core_mimiciv_path + 'core/patients.csv')
    admissions = pd.merge(admissions, patients, on = 'subject_id')
  
    #list of unique hadm id with conditions specified
    condition_list = []
    condition_list = d_icd_diagnoses[d_icd_diagnoses['long_title'].str.contains(condition_keywords[0])]
    for i in condition_keywords[1:]:
        condition_list = condition_list[condition_list['long_title'].str.contains('chronic')]
      
    icd_list = condition_list['icd_code'].unique().tolist() 
    hid_list_chf = diagnoses_icd[diagnoses_icd['icd_code'].isin(icd_list) & 
                  (diagnoses_icd['seq_num']<=3)]['hadm_id'].unique().tolist()
  
    pkl_id = pd.read_csv(core_mimiciv_path + 'pickle/haim_mimiciv_key_ids.csv')
    id_hf = pkl_id[pkl_id['hadm_id'].isin(hid_list_chf)].drop_duplicates(subset='hadm_id')
  
    # delete all pkl files with only less than 1 day recorded
    pkl_list_adm = admissions[admissions['hadm_id'].isin(id_hf['hadm_id'])]
    pkl_list_adm['dischtime'] = pd.to_datetime(pkl_list_adm['dischtime'])
    pkl_list_adm['admittime'] = pd.to_datetime(pkl_list_adm['admittime'])
    pkl_list_adm['deltatime'] = (pkl_list_adm['dischtime'] - pkl_list_adm['admittime']).astype('timedelta64[D]').values
    pkl_no_zero = pkl_list_adm[pkl_list_adm['deltatime'] != 0]['hadm_id']
    no_zero_id = pkl_id[pkl_id['hadm_id'].isin(pkl_no_zero)].drop_duplicates(subset='hadm_id')
  
    haim_ids_list = no_zero_id['haim_id_pickle'].values
  
    return haim_ids_list



#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                             Core embeddings for MIMIC-IV Deep Fusion                        |
#   

# LOAD CORE INFO OF MIMIC IV PATIENTS
def load_core_mimic_haim_info(core_mimiciv_path, df_haim_ids):
    # Inputs:
    #   core_mimiciv_path -> Base path of mimiciv
    #   df_haim_ids -> Table of HAIM ids and corresponding keys
    #
    # Outputs:
    #   df_haim_ids_core_info -> Updated dataframe with integer representations of core data

    # %% EXAMPLE OF USE
    # df_haim_ids_core_info = load_core_mimic_haim_info(core_mimiciv_path)

    # Load core table
    df_mimiciv_core = pd.read_csv(core_mimiciv_path + 'core/core.csv')

    # Generate integer representations of categorical variables in core
    core_var_select_list = ['gender', 'ethnicity', 'marital_status', 'language','insurance']
    core_var_select_int_list = ['gender_int', 'ethnicity_int', 'marital_status_int', 'language_int','insurance_int']
    df_mimiciv_core[core_var_select_list] = df_mimiciv_core[core_var_select_list].astype('category')
    df_mimiciv_core[core_var_select_int_list] = df_mimiciv_core[core_var_select_list].apply(lambda x: x.cat.codes)

    # Combine HAIM IDs with core data
    df_haim_ids_core_info = pd.merge(df_haim_ids, df_mimiciv_core, on=["subject_id", "hadm_id"])

    return df_haim_ids_core_info


# GET DEMOGRAPHICS EMBEDDINGS OF MIMIC IV PATIENT
def get_demographic_embeddings(dt_patient, verbose=0):
    # Inputs:
    #   dt_patient -> Timebound mimic patient structure
    #   verbose -> Flag to print found keyword outputs (0,1,2)
    #
    # Outputs:
    #   base_embeddings -> Core base embeddings for the selected patient

    # %% EXAMPLE OF USE
    # base_embeddings = get_demographic_embeddings(dt_patient, df_haim_ids_core_info, verbose=2)

    # Retrieve dt_patient and get embeddings 
    print(dt_patient.core.columns)
    demo_embeddings =  dt_patient.core.loc[0, ['anchor_age', 'gender', 'ethnicity', 'marital_status', 'language', 'insurance']]
    #demo_embeddings =  dt_patient.core.loc[0, ['anchor_age', 'gender_int', 'ethnicity_int', 'marital_status_int', 'language_int', 'insurance_int']]

    if verbose >= 1:
        print(demo_embeddings)

    demo_embeddings = demo_embeddings.values

    return demo_embeddings


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                           Vision CXR embeddings for MIMIC-IV Deep Fusion                       |
#   

'''
## -> VISION REPRESENTATION OF MIMIC-IV EHR USING CNNs
A library for chest X-ray datasets and models. Including pre-trained models.
Motivation: While there are many publications focusing on the prediction of radiological and clinical findings from chest X-ray images much of this work is inaccessible to other researchers.

In the case of researchers addressing clinical questions it is a waste of time for them to train models from scratch. To address this, TorchXRayVision provides pre-trained models which are trained on large cohorts of data and enables 1) rapid analysis of large datasets 2) feature reuse for few-shot learning.
In the case of researchers developing algorithms it is important to robustly evaluate models using multiple external datasets. Metadata associated with each dataset can vary greatly which makes it difficult to apply methods to multiple datasets. TorchXRayVision provides access to many datasets in a uniform way so that they can be swapped out with a single line of code. These datasets can also be merged and filtered to construct specific distributional shifts for studying generalization. https://github.com/mlmed/torchxrayvision
'''

def get_single_chest_xray_embeddings(img):
    # Inputs:
    #   img -> Image array
    #
    # Outputs:
    #   densefeature_embeddings ->  CXR dense feature embeddings for image
    #   prediction_embeddings ->  CXR embeddings of predictions for image
    
    
    # %% EXAMPLE OF USE
    # densefeature_embeddings, prediction_embeddings = get_single_chest_xray_embeddings(img)
    
    # Clean out process bar before starting
    sys.stdout.flush()
    
    # Select if you want to use CUDA support for GPU (optional as it is usually pretty fast even in CPUT)
    cuda = False
    
    # Select model with a String that determines the model to use for Chest Xrays according to https://github.com/mlmed/torchxrayvision
    #model_weights_name = "densenet121-res224-all" # Every output trained for all models
    #model_weights_name = "densenet121-res224-rsna" # RSNA Pneumonia Challenge
    #model_weights_name = "densenet121-res224-nih" # NIH chest X-ray8
    #model_weights_name = "densenet121-res224-pc") # PadChest (University of Alicante)
    model_weights_name = "densenet121-res224-chex" # CheXpert (Stanford)
    #model_weights_name = "densenet121-res224-mimic_nb" # MIMIC-CXR (MIT)
    #model_weights_name = "densenet121-res224-mimic_ch" # MIMIC-CXR (MIT)
    #model_weights_name = "resnet50-res512-all" # Resnet only for 512x512 inputs
    # NOTE: The all model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset.
    
    # Extract chest x-ray image embeddings and preddictions
    densefeature_embeddings = []
    prediction_embeddings = []
    
    #img = skimage.io.imread(img_path) # If importing from path use this
    img = xrv.datasets.normalize(img, 255)

    # For each image check if they are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("Error: Dimension lower than 2 for image!")
    
    # Add color channel for prediction
    #Resize using OpenCV
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)   
    img = img[None, :, :]

    #Or resize using core resizer (thows error sometime)
    #transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
    #img = transform(img)
    model = xrv.models.DenseNet(weights = model_weights_name)
    # model = xrv.models.ResNet(weights="resnet50-res512-all") # ResNet is also available

    output = {}
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        if cuda:
            img = img.cuda()
            model = model.cuda()
          
        # Extract dense features
        feats = model.features(img)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        densefeatures = feats.cpu().detach().numpy().reshape(-1)
        densefeature_embeddings = densefeatures

        # Extract predicted probabilities of considered 18 classes:
        # Get by calling "xrv.datasets.default_pathologies" or "dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))"
        # ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',Fibrosis',
        #  'Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule',Mass','Hernia',
        #  'Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']
        preds = model(img).cpu()
        predictions = preds[0].detach().numpy()
        prediction_embeddings = predictions  

    # Return embeddings
    return densefeature_embeddings, prediction_embeddings

def get_chest_xray_embeddings(dt_patient, verbose=0):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   verbose -> Level of printed output of function
    #
    # Outputs:
    #   aggregated_densefeature_embeddings -> CXR aggregated dense feature embeddings for all images in timebound patient
    #   densefeature_embeddings ->  List of CXR dense feature embeddings for all images
    #   aggregated_prediction_embeddings -> CXR aggregated embeddings of predictions for all images in timebound patient
    #   prediction_embeddings ->  List of CXR embeddings of predictions for all images
    #   imgs_weights ->  Array of weights for embedding aggregation

    # %% EXAMPLE OF USE
    # aggregated_densefeature_embeddings, densefeature_embeddings, aggregated_prediction_embeddings, prediction_embeddings, imgs_weights = get_chest_xray_embeddings(dt_patient, verbose=2)

    # Clean out process bar before starting
    sys.stdout.flush()

    # Select if you want to use CUDA support for GPU (optional as it is usually pretty fast even in CPUT)
    cuda = True

    # Select model with a String that determines the model to use for Chest Xrays according to https://github.com/mlmed/torchxrayvision
    #   model_weights_name = "densenet121-res224-all" # Every output trained for all models
    #   model_weights_name = "densenet121-res224-rsna" # RSNA Pneumonia Challenge
    #model_weights_name = "densenet121-res224-nih" # NIH chest X-ray8
    #model_weights_name = "densenet121-res224-pc") # PadChest (University of Alicante)
    model_weights_name = "densenet121-res224-chex" # CheXpert (Stanford)
    #   model_weights_name = "densenet121-res224-mimic_nb" # MIMIC-CXR (MIT)
    #model_weights_name = "densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)
    #model_weights_name = "resnet50-res512-all" # Resnet only for 512x512 inputs
    # NOTE: The all model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset.


    # Extract chest x-ray images from timebound patient and iterate through them
    imgs = dt_patient.imcxr
    densefeature_embeddings = []
    prediction_embeddings = []

    # Iterate
    nImgs = len(imgs)
    with tqdm(total = nImgs) as pbar:
        for idx, img in enumerate(imgs):
            img = xrv.datasets.normalize(img, 255)
          
            # For each image check if they are 2D arrays
            if len(img.shape) > 2:
                img = img[:, :, 0]
            if len(img.shape) < 2:
                print("Error: Dimension lower than 2 for image!")

            # Add color channel for prediction
            #Resize using OpenCV
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)   
            img = img[None, :, :]
    
            model = xrv.models.DenseNet(weights = model_weights_name)
            
            output = {}
            with torch.no_grad():
                img = torch.from_numpy(img).unsqueeze(0)
                if cuda:
                    img = img.cuda()
                    model = model.cuda()
              
                # Extract dense features
                feats = model.features(img)
                feats = F.relu(feats, inplace=True)
                feats = F.adaptive_avg_pool2d(feats, (1, 1))
                densefeatures = feats.cpu().detach().numpy().reshape(-1)
                densefeature_embeddings.append(densefeatures) # append to list of dense features for all images
                
                # Extract predicted probabilities of considered 18 classes:
                # Get by calling "xrv.datasets.default_pathologies" or "dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))"
                # ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',Fibrosis',
                #  'Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule',Mass','Hernia',
                #  'Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']
                preds = model(img).cpu()
                predictions = preds[0].detach().numpy()
                prediction_embeddings.append(predictions) # append to list of predictions for all images
            
                if verbose >=1:
                    # Update process bar
                    pbar.update(1)
        
        
    # Get image weights by hours passed from current time to image
    orig_imgs_weights = np.asarray(dt_patient.cxr.deltacharttime.values)
    adj_imgs_weights = orig_imgs_weights - orig_imgs_weights.min()
    imgs_weights = (adj_imgs_weights) / (adj_imgs_weights).max()
  
    # Aggregate with weighted average of ebedding vector across temporal dimension
    try:
        aggregated_densefeature_embeddings = np.average(densefeature_embeddings, axis=0, weights=imgs_weights)
        if np.isnan(np.sum(aggregated_densefeature_embeddings)):
            aggregated_densefeature_embeddings = np.zeros_like(densefeature_embeddings[0])
    except:
        aggregated_densefeature_embeddings = np.zeros_like(densefeature_embeddings[0])
      
    try:
        aggregated_prediction_embeddings = np.average(prediction_embeddings, axis=0, weights=imgs_weights)
        if np.isnan(np.sum(aggregated_prediction_embeddings)):
            aggregated_prediction_embeddings = np.zeros_like(prediction_embeddings[0])
    except:
        aggregated_prediction_embeddings = np.zeros_like(prediction_embeddings[0])
      
      
    if verbose >=2:
        x = orig_imgs_weights
        y = prediction_embeddings
        plt.xlabel("Time [hrs]")
        plt.ylabel("Disease probability [0-1]")
        plt.title("A test graph")
        for i in range(len(y[0])):
            plt.plot(x,[pt[i] for pt in y],'o', label = xrv.datasets.default_pathologies[i])
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.show()

    # Return embeddings
    return aggregated_densefeature_embeddings, densefeature_embeddings, aggregated_prediction_embeddings, prediction_embeddings, imgs_weights


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                  Preprocessing MIMIC-IV Dataset                                 |
#

# SAVE SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV 
def save_patient_object(obj, filepath):
    # Inputs:
    #   obj -> Timebound ICU patient stay object
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   VOID -> Object is saved in filename path
    # Overwrites any existing file.
    with open(filepath, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# LOAD SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV
def load_patient_object(filepath):
    # Inputs:
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   obj -> Loaded timebound ICU patient stay object

    # Overwrites any existing file.
    with open(filepath, 'rb') as input:  
        return pickle.load(input)

# GET ALL DEMOGRAPHCS DATA OF A TIMEBOUND PATIENT RECORD
def get_demographics(dt_patient):
    dem_info = dt_patient.demographics[['gender', 'anchor_age', 'anchor_year']] 
    dem_info['gender'] = (dem_info['gender'] == 'M').astype(int)
    return dem_info.values[0]





