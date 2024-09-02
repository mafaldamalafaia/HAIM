###########################################################################################################
#                      Create HAIM-MIMIC-MM & pickle files from MIMIC-IV & MIMIC-CXR-JPG
###########################################################################################################
# 
# Licensed under the Apache License, Version 2.0**
# You may not use this file except in compliance with the License. You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is 
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
# implied. See the License for the specific language governing permissions and limitations under the License.

#-> Authors: 
#      Luis R Soenksen (<soenksen@mit.edu>),
#      Yu Ma (<midsumer@mit.edu>),
#      Cynthia Zeng (<czeng12@mit.edu>),
#      Leonard David Jean Boussioux (<leobix@mit.edu>),
#      Kimberly M Villalobos Carballo (<kimvc@mit.edu>),
#      Liangyuan Na (<lyna@mit.edu>),
#      Holly Mika Wiberg (<hwiberg@mit.edu>),
#      Michael Lingzhi Li (<mlli@mit.edu>),
#      Ignacio Fuentes (<ifuentes@mit.edu>),
#      Dimitris J Bertsimas (<dbertsim@mit.edu>),
# -> Last Update: Dec 30th, 2021
# # Code for Patient parsing in pickle files from HAIM-MIMIC-MM creation ( MIMIC-IV + MIMIC-CXR-JPG)
# # This files describes how we generate the patient-admission-stay level pickle files. 

#HAIM
from MIMIC_IV_HAIM_API import *

# Display optiona
from IPython.display import Image # IPython display
pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('float_format', '{:f}'.format)
pd.options.mode.chained_assignment = None  # default='warn'
#get_ipython().run_line_magic('matplotlib', 'inline')


# ### -> Initializations & Data Loading
# Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)

# DATASET Location
data_path = '/export/scratch2/constellation-data/malafaia/physionet.org/files/'
# Define MIMIC IV Data Location
core_mimiciv_path = '/export/scratch2/constellation-data/malafaia/physionet.org/files/mimiciv/1.0/'
# Define MIMIC IV Image Data Location (usually external drive)
core_mimiciv_imgcxr_path = '/export/scratch2/constellation-data/malafaia/physionet.org/files/mimic-cxr-jpg/2.0.0/'

# load all tables in memory
df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_drgcodes, df_emar, df_emar_detail, df_hcpcsevents, df_labevents, df_microbiologyevents, df_poe, df_poe_detail, df_prescriptions, df_procedures_icd, df_services, df_d_icd_diagnoses, df_d_icd_procedures, df_d_hcpcs, df_d_labitems, df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_datetimeevents, df_chartevents, df_d_items, df_mimic_cxr_split, df_mimic_cxr_chexpert, df_mimic_cxr_metadata, df_mimic_cxr_negbio = load_mimiciv(data_path)

# -> MASTER DICTIONARY of health items
# Generate dictionary for chartevents, labevents and HCPCS
df_patientevents_categorylabels_dict = pd.DataFrame(columns = ['eventtype', 'category', 'label'])

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
		
		
## CORE
print('- CORE > df_admissions')
print('--------------------------------')
print(df_admissions.dtypes)
print('\n\n')

print('- CORE > df_patients')
print('--------------------------------')
print(df_patients.dtypes)
print('\n\n')

print('- CORE > df_transfers')
print('--------------------------------')
print(df_transfers.dtypes)
print('\n\n')


## HOSP
print('- HOSP > df_d_labitems')
print('--------------------------------')
print(df_d_labitems.dtypes)
print('\n\n')

print('- HOSP > df_d_icd_procedures')
print('--------------------------------')
print(df_d_icd_procedures.dtypes)
print('\n\n')

print('- HOSP > df_d_icd_diagnoses')
print('--------------------------------')
print(df_d_icd_diagnoses.dtypes)
print('\n\n')

print('- HOSP > df_d_hcpcs')
print('--------------------------------')
print(df_d_hcpcs.dtypes)
print('\n\n')

print('- HOSP > df_diagnoses_icd')
print('--------------------------------')
print(df_diagnoses_icd.dtypes)
print('\n\n')

print('- HOSP > df_drgcodes')
print('--------------------------------')
print(df_drgcodes.dtypes)
print('\n\n')

print('- HOSP > df_emar')
print('--------------------------------')
print(df_emar.dtypes)
print('\n\n')

print('- HOSP > df_emar_detail')
print('--------------------------------')
print(df_emar_detail.dtypes)
print('\n\n')

print('- HOSP > df_hcpcsevents')
print('--------------------------------')
print(df_hcpcsevents.dtypes)
print('\n\n')

print('- HOSP > df_labevents')
print('--------------------------------')
print(df_labevents.dtypes)
print('\n\n')

print('- HOSP > df_microbiologyevents')
print('--------------------------------')
print(df_microbiologyevents.dtypes)
print('\n\n')

print('- HOSP > df_poe')
print('--------------------------------')
print(df_poe.dtypes)
print('\n\n')

print('- HOSP > df_poe_detail')
print('--------------------------------')
print(df_poe_detail.dtypes)
print('\n\n')

print('- HOSP > df_prescriptions')
print('--------------------------------')
print(df_prescriptions.dtypes)
print('\n\n')

print('- HOSP > df_procedures_icd')
print('--------------------------------')
print(df_procedures_icd.dtypes)
print('\n\n')

print('- HOSP > df_services')
print('--------------------------------')
print(df_services.dtypes)
print('\n\n')


## ICU
print('- ICU > df_procedureevents')
print('--------------------------------')
print(df_procedureevents.dtypes)
print('\n\n')

print('- ICU > df_outputevents')
print('--------------------------------')
print(df_outputevents.dtypes)
print('\n\n')

print('- ICU > df_inputevents')
print('--------------------------------')
print(df_inputevents.dtypes)
print('\n\n')

print('- ICU > df_icustays')
print('--------------------------------')
print(df_icustays.dtypes)
print('\n\n')

print('- ICU > df_datetimeevents')
print('--------------------------------')
print(df_datetimeevents.dtypes)
print('\n\n')

print('- ICU > df_d_items')
print('--------------------------------')
print(df_d_items.dtypes)
print('\n\n')

print('- ICU > df_chartevents')
print('--------------------------------')
print(df_chartevents.dtypes)
print('\n\n')


## CXR
print('- CXR > df_mimic_cxr_split')
print('--------------------------------')
print(df_mimic_cxr_split.dtypes)
print('\n\n')

print('- CXR > df_mimic_cxr_chexpert')
print('--------------------------------')
print(df_mimic_cxr_chexpert.dtypes)
print('\n\n')

print('- CXR > df_mimic_cxr_metadata')
print('--------------------------------')
print(df_mimic_cxr_metadata.dtypes)
print('\n\n')

print('- CXR > df_mimic_cxr_negbio')
print('--------------------------------')
print(df_mimic_cxr_negbio.dtypes)
print('\n\n')


## NOTES
print('- NOTES > df_noteevents')
print('--------------------------------')
#print(df_noteevents.dtypes)
print('\n\n')

#print('- NOTES > df_icunotes')
#print('--------------------------------')
#print(df_dsnotes.dtypes)
#print('\n\n')

#print('- NOTES > df_ecgnotes')
#print('--------------------------------')
#print(df_ecgnotes.dtypes)
#print('\n\n')

#print('- NOTES > df_echonotes')
#print('--------------------------------')
#print(df_echonotes.dtypes)
#print('\n\n')

#print('- NOTES > df_radnotes')
#print('--------------------------------')
#print(df_radnotes.dtypes)
#print('\n\n')


# ## -> GET LIST OF ALL UNIQUE ID COMBINATIONS IN MIMIC-IV (subject_id, hadm_id, stay_id)

# Get Unique Subject/HospAdmission/Stay Combinations
df_ids = pd.concat([pd.DataFrame(), df_procedureevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_outputevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_inputevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_icustays[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_datetimeevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_chartevents[['subject_id','hadm_id','stay_id']]], sort=True).drop_duplicates()

# Get Unique Subjects with Chest Xrays
df_cxr_ids = pd.concat([pd.DataFrame(), df_mimic_cxr_chexpert[['subject_id']]], sort=True).drop_duplicates()

# Get Unique Subject/HospAdmission/Stay Combinations with Chest Xrays
df_haim_ids = df_ids[df_ids['subject_id'].isin(df_cxr_ids['subject_id'].unique())] 

# Save Unique Subject/HospAdmission/Stay Combinations with Chest Xrays    
df_haim_ids.to_csv(core_mimiciv_path + 'haim_mimiciv_key_ids.csv', index=False)

print('Unique Subjects: ' + str(len(df_patients['subject_id'].unique())))
print('Unique Subjects/HospAdmissions/Stays Combinations: ' + str(len(df_ids)))
print('Unique Subjects with Chest Xrays Available: ' + str(len(df_cxr_ids)))

# Save Unique Subject/HospAdmission/Stay Combinations with Chest Xrays    
df_haim_ids = pd.read_csv(core_mimiciv_path + 'haim_mimiciv_key_ids.csv')
print('Unique HAIM Records Available: ' + str(len(df_haim_ids)))

# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
#nfiles = generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path)


# my method
# Extract information for patient
# replace generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path)
nfiles = len(df_haim_ids)
with tqdm(total = nfiles) as pbar: # progress bar
    # Iterate through all patients
    for haim_patient_idx in range(nfiles):
        # Let's select each single patient and extract patient object
        start_hr = None # Select timestamps
        end_hr = None   # Select timestamps
        
        # import os
        filename = f"{haim_patient_idx:08d}" + '.pkl'
        filepath = core_mimiciv_path + 'pickle/' + filename
        if not os.path.exists(filepath):
            # replace key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient = extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr)
            # Save
            # Extract information for patient
            key_subject_id = df_haim_ids.iloc[haim_patient_idx].subject_id
            key_hadm_id = df_haim_ids.iloc[haim_patient_idx].hadm_id
            key_stay_id = df_haim_ids.iloc[haim_patient_idx].stay_id
            start_hr = start_hr # Select timestamps
            end_hr = end_hr   # Select timestamps
            
            # replace patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id)
            #-> FILTER data
            ##-> CORE
            df_base_core = df_admissions.merge(df_patients, how='left').merge(df_transfers, how='left')
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
            #f_df_chartevents = f_df_chartevents.merge(df_d_items, how='left')       
            
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
            #f_df_noteevents = df_noteevents[(df_noteevents.subject_id == key_subject_id) & (df_noteevents.hadm_id == key_hadm_id)]
            #f_df_dsnotes = df_dsnotes[(df_dsnotes.subject_id == key_subject_id) & (df_dsnotes.hadm_id == key_hadm_id) & (df_dsnotes.stay_id == key_stay_id)]
            #f_df_ecgnotes = df_ecgnotes[(df_ecgnotes.subject_id == key_subject_id) & (df_ecgnotes.hadm_id == key_hadm_id) & (df_ecgnotes.stay_id == key_stay_id)]
            #f_df_echonotes = df_echonotes[(df_echonotes.subject_id == key_subject_id) & (df_echonotes.hadm_id == key_hadm_id) & (df_echonotes.stay_id == key_stay_id)]
            #f_df_radnotes = df_radnotes[(df_radnotes.subject_id == key_subject_id) & (df_radnotes.hadm_id == key_hadm_id) & (df_radnotes.stay_id == key_stay_id)]
             
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
            #noteevents = f_df_noteevents
            #dsnotes = f_df_dsnotes
            #ecgnotes = f_df_ecgnotes
            #echonotes = f_df_echonotes
            #radnotes = f_df_radnotes
             
            # Create patient object and return
            patient = Patient_ICU(admissions, demographics, transfers, core, \
										diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents, \
										labevents, microbiologyevents, poe, poe_detail, \
										prescriptions, procedures_icd, services, procedureevents, \
										outputevents, inputevents, icustays, datetimeevents, \
										cxr, imcxr)#, chartevents, noteevents, dsnotes, ecgnotes, \
										#echonotes, radnotes)
            dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
            save_patient_object(dt_patient, core_mimiciv_path + 'pickle/' + filename)
        # Update process bar
        pbar.update(1)
