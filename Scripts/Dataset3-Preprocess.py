# %%
# ! IMPORTS
# =============================================================================
from CommonFuncs import (pickle_import, pickle_export, show_all, make_montage,
                         modify_folder, generate_templates, N200_detection,
                         ajdust_topo_plot, reg_analysis, select_files,
                         data_struct, tree)
import itertools
# import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import pathlib
# import pickle
# from PIL import Image
from pprint import pprint
from pymatreader import read_mat
import re
import seaborn as sns
import statsmodels.api as sm
import warnings


# * ########################## GLOBAL PARAMETERS ##############################

mne.set_log_level(verbose='WARNING')
warnings.filterwarnings('ignore',category=DeprecationWarning)

plt.rcParams['figure.dpi'] = 100
# plt.rcParams['figure.figsize']  = [19.2, 14.4]
plt.rcParams['figure.figsize']  = [9.6, 7.2]

# matplotlib.pyplot.rcdefaults()

# * ########################## GLOBAL VARIABLES ###############################
# /!\ # 1250 = stim onset time /!\ 

px = 1 / plt.rcParams['figure.dpi']  # pixel size in inches
figsize = (1280 * px, 1280 * px)
# units = {'time': 'ms', 'eeg':10**-6}

sfreq = 1000  # sampling frequency in Hz

template_rad = 600

n200_window = [115, 275]  # 125-275 ms after stimulus onset

# Selecting the first 'nb_comps' svd components  (here: first 10 svd components)
nb_comps = 10
nb_chans = 5

templates_path = 'C:/Users/chris/OneDrive - UvA/Pinier - '\
                    'Internship/Data/templates/N200template.mat'

headmodel_path = 'C:/Users/chris/OneDrive - UvA/Pinier - '\
                    'Internship/Data/headmodels/eginn128hm.mat'

# %% 
# ########################## DIRECTORIES & FILES ###############################

base = pathlib.Path('C:/Users/chris/Videos/EEG_data/proj3')

if base != (wd := pathlib.Path.cwd()):
    print(f'- Current working directory: {wd}')
    print(f'- Changing it to: {base}')
    os.chdir(base)

dir_data = base/'1-Data/EEG data and scripts/conv'

dir_ICA = base/'2-Fitted_ICA'
dir_figs = base/'3-Figures'
dir_pickle = base/'Pickle_files'
# dir_subj = dir_data/'subjects'
# dir_subj = pathlib.Path('D:/pdmfinal/Rearranged')
# subjects = {path.name:path for path in dir_subj.iterdir() if path.is_dir()}

# %%
# ############################## TEMPLATES  ####################################
wave_template, template_img_arr = generate_templates(
                                                    template_path = templates_path,
                                                    headmodel_path = headmodel_path,
                                                    save_path = base,
                                                    figsize = figsize
                                                    )

df_topo_tpl = pd.DataFrame(template_img_arr)


data_dict = dict()

#  ################# proj1 data_dict:             
# ['events', 'event_counts', 'event_valids', 'event_invalids', 'events_cleaned', 
# 'bipolar_ref', 'epochs', 'rejections', 'cleaned_epochs', 'evoked', 'duration', 
# 'bad_ch', 'subject_info', 'ICA_fitted', 'accuracy', 'resp_times'])
# 
# ################# proj2 data_dict:
# dict_keys(['file', 'task', 'ICA', 'chan_pos', 'resp_times', 'condition', 'n200_data'])

# FIG FOR PRESENTATION
# plt.ioff()
# sns.lineplot(data=wave_template)
# plt.axis('off')
# plt.savefig("test.png", dpi=300, transparent=True)




# ########################### BEHAVIORAL DATA ##################################
SAT3_fpath = dir_data/'SAT3.csv'
df_behav = pd.read_csv(SAT3_fpath)

subj_info_mat = r"C:\Users\chris\Videos\EEG_data\proj3\1-Data\EEG data and scripts\conv\subj_info.mat"
subj_info = read_mat(subj_info_mat)
subj_info['subj_info'][0] #.keys()
# pd.read_clipboard()
# subj_list = df_behav['subj'].unique()

df_accuracy = df_behav.pivot_table(index='subj', values='acc', aggfunc=sum) / 200
df_accuracy.to_dict()
sns.barplot(x=df_accuracy.index, y=df_accuracy['acc'], data=df_accuracy, color='#64B5CD')
plt.ylabel('Accuracy')
plt.xlabel('Subjects')
plt.savefig(base/'accuracy.png', dpi=300)


df_resp_percentile = df_behav.pivot_table(index='subj', values='RT', 
                    aggfunc=lambda x: np.nanpercentile(x, 10))
                    
df_resp_percentile.index = [f'subj{str(i).zfill(2)}' for i in 
                            df_resp_percentile.index]


# ############################ EEG FILE ########################################
eeg_files = [f for f in dir_data.iterdir() if re.search('MD3-\d{4}.vhdr', f.name)]
exp_data_files = [f for f in dir_data.iterdir() if re.search('data.+.mat', f.name)]


file_0 = read_mat(exp_data_files[0])['data']

# file_0.keys()
# file_0['hdr'].keys()


ch_names = file_0['label'].copy()

ch_types = file_0['hdr']['chantype']
# nb_chans = len(ch_names)

sfreq = file_0['fsample']
timepoints = file_0['time']

# unit_conv = 1E-7
# trials = [t * unit_conv for t in file_0['trial']]

# info = mne.create_info(ch_names, sfreq, ch_types='eeg', verbose=None)


# Create a 10-20 montage
mont1020 = mne.channels.make_standard_montage('standard_1020')
for ch in [('Fp1', 'FP1'), ('Fp2', 'FP2')]:
    mont1020.ch_names[mont1020.ch_names.index(ch[0])] = ch[1] 

# Choose what channels you want to keep 
# Make sure that these channels exist e.g. T1 does not exist in the standard 10-20 EEG system!
kept_channels = [ch for ch in ch_names if ch in mont1020.ch_names]

ind = [i for (i, channel) in enumerate(mont1020.ch_names) if channel in kept_channels]
montage = mont1020.copy()

# Keep only the desired channels
montage.ch_names = [mont1020.ch_names[x] for x in ind]
kept_channel_info = [mont1020.dig[x + 3] for x in ind]

# Keep the first three rows as they are the fiducial points information
montage.dig = mont1020.dig[0:3] + kept_channel_info


eog_names = ['EOGh', 'EOGv']



def apply_ICA(f):
    subj = re.search('\d{4}', f.name)[0].lstrip('0').zfill(2)
    
    subj_raw = mne.io.read_raw_brainvision(f, eog=eog_names, misc='auto', 
                                            scale=1.0, preload=True, 
                                            verbose=None)
    
    # for ch in [('FP1', 'Fp1'), ('FP2', 'Fp2')]:
    #     subj_raw.ch_names[subj_raw.ch_names.index(ch[0])] = ch[1] 

    subj_raw.set_montage(montage)
    
    
    print('Setting Average Reference')
    subj_raw.set_eeg_reference(ref_channels='average')
    
    print('Filtering')
    subj_raw.filter(l_freq=.1, h_freq=100)
    
    all_raws.append(subj_raw)
    
    print('Running the ICA')
    ICA = mne.preprocessing.ICA(n_components=None, noise_cov=None,
                                random_state=97, method='fastica', max_iter='auto')
    
    ICA.fit(subj_raw)
    ICA.save(dir_ICA/f'subj{subj}_fitted-ica.fif')
    
    eog_inds, eog_scores = ICA.find_bads_eog(subj_raw, ch_name=eog_names)

    ICA.exclude = eog_inds

    all_ICA.append(ICA)
    subj_raw = ICA.apply(subj_raw)
    
    subj_raw.save(base/f'subj{subj}_raw.fif')
    
    subj_events = mne.events_from_annotations(subj_raw, event_id='auto')[0]
    all_events.append(subj_events)
    
    print('Epoching')
    subj_epochs = mne.Epochs(subj_raw, subj_events, tmin=-0.2, tmax=0.3, 
                            baseline=(None, 0), detrend=1)
    all_epochs.append(subj_epochs)
    
    print('Creating ERP')
    subj_evoked = subj_epochs.average()
    all_evoked.append(subj_evoked)

    # subj_data  = {
    #         'subj_raw':subj_raw, 
    #         'ICA':ICA, 
    #         'eog_inds':eog_inds, 
    #         'eog_scores':eog_scores, 
    #         'subj_events':subj_events,
    #         'subj_epochs':subj_epochs,
    #         'subj_evoked':subj_evoked
    #         }
    # pickle_export(subj_data, dir_pickle, f_name=f'{subj}_ICA_dict')



def analysis(f):
    
    subj = re.search('subj\d{2}', f.name)[0]
    
    data_dict[subj] = dict(
                            file = None,
                            ICA = {
                                    'eog_inds':None, 
                                    'eog_scores':None, 
                                    'ica_file':None
                                    },
                            chan_pos = None,
                            resp_times = None,
                            N200_data = None,
                            subj_info = None,
                            evoked = None,
                            events = None,
                            event_ids= None)

    
    
    raw = mne.io.Raw(f, preload=True)

    print('Bandpass Filter 1-10 Hz ...')
    raw.filter(l_freq=1, h_freq=10)
    
    events, event_ids = mne.events_from_annotations(raw, event_id='auto')

    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.3, baseline=(None, 0), 
                        detrend=1, reject = dict(eeg=100e-6))

    evoked = epochs.average()

    evoked.plot(spatial_colors=True, show=False)

    data_dict[subj]['events'] = events
    data_dict[subj]['event_ids'] = event_ids
    
    data_dict[subj]['N200_data'] = N200_detection(subj, evoked, dir_figs,
                                                  dir_pickle, n200_window,
                                                  nb_chans, nb_comps,
                                                  wave_template, df_topo_tpl)
    
    # pickle_export(data_dict, base, f_name=subj)



preprocessed_raws = [f for f in (base/'Prepocessed').iterdir() if f.is_file()]

for f in preprocessed_raws:
    analysis(f)

data_dict = dict(*pickle_import('Proj3-DataDict.pickle'))

# data_dict['subj01']['subj_info']
data_dict['subj01']['N200_data'].keys()

## PLOTING ALL SELECTED COMP TOGETHER ##
plt.ioff()
plt.figure(figsize=figsize)
for subj in data_dict:
    comp = data_dict[subj]['N200_data']['picked_comp']
    fig_data = data_dict[subj]['N200_data']['u_comps'][comp, :]
    timepoints = np.linspace(115, 276, 80)
    plt.plot(timepoints, fig_data)
plt.savefig(base/'selected_comps-combined.png')



latencies = pd.Series({k: data_dict[k]['N200_data']['u_peak']['latency'] 
                        for k in data_dict}, name='latency')



df_reg_data = df_resp_percentile
df_reg_data = df_reg_data.merge(latencies.to_frame(), left_index=True, right_index=True)
df_reg_data = df_reg_data.reindex(columns=['latency', 'RT'])
df_reg_data.columns = ['latency', 'resp. time (10th %ile)']

# Export as excel file
df_reg_data.to_excel(base/'reg_data.xlsx')

# x = np.array([i[0] for i in reg_data.values()])
# y = np.array([i[1] for i in reg_data.values()])


# df_reg_data = pd.read_clipboard()

# x = df_reg_data['latency'].to_numpy()
# y = df_reg_data['resp. time (10th %ile)'].to_numpy()

 
# reg = reg_analysis(x, y, save_dir=base, show=True)


# ############### LIN REG FORM STATSMODELS

# df_reg_data


# sns.regplot(x='latency', y='resp. time (10th %ile)', data=df_reg_data)
# plt.show(block=False)
# plt.savefig(base/'sns_regplot.png', dpi=300)

# X = sm.add_constant(df_reg_data['latency']) # adding a constant
# Y = df_reg_data['resp. time (10th %ile)']


# model = sm.OLS(Y, X).fit()
# predictions = model.predict(X)

# model_summary = model.summary()
# dir(model_summary)
# model_summary.as_text()

# with open(base/'LinReg-Summary.txt', 'w') as f:
#     f.write(model_summary.as_text())

# print(model_summary)

# plt.figure(figsize=figsize)
# plt.scatter(df_reg_data['latency'], Y)
# plt.plot(df_reg_data['latency'], model.predict(), 'orange')
# plt.show(block=False)



# # ######## Export correlation results with templates as excel file ############

# df_correlation = pd.DataFrame()
# df_correlation_selected = pd.DataFrame()

# for subj in data_dict:
#     df = data_dict[subj]['N200_data']['df_corr']
#     df_correlation = pd.concat([df_correlation, df])

#     selected = df.sort_values('prod', ascending=False).iloc[0:1]
#     df_correlation_selected = pd.concat([df_correlation_selected, selected])


# df_correlation_selected
# df_correlation_selected.to_excel(base/'Selected_SVD_Components.xlsx')



# ## PLOTING ALL SELECTED COMP TOGETHER ##
# plt.ion()
# plt.figure(figsize=figsize)
# for subj in data_dict:
#     comp = data_dict[subj][ses]['N200_data']['picked_comp']
#     fig_data = data_dict[subj][ses]['N200_data']['u_comps'][comp, :]
#     plt.plot(fig_data)
# plt.savefig(base/'selected_comps-combined.png')

