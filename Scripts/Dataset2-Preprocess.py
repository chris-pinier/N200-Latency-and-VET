# %%
# ! IMPORTS
# =============================================================================
from CommonFuncs import (pickle_import, pickle_export, show_all, make_montage,
                         modify_folder, generate_templates, N200_detection,
                         ajdust_topo_plot, reg_analysis, select_files,
                         data_struct, tree)
import itertools
import matplotlib
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
# import scipy.io
from sklearn.linear_model import LinearRegression
import warnings


# * ########################### GLOBAL PARAMETERS ##############################

mne.set_config('MNE_USE_CUDA', 'true')
mne.set_config('MNE_BROWSER_USE_OPENGL', 'true')
mne.get_config()
mne.set_log_level(verbose='WARNING')
# mne.viz.get_browser_backend()
# mne.viz.get_3d_backend()

warnings.filterwarnings('ignore', category=DeprecationWarning)

# plt.rcParams.update(matplotlib.rcParamsDefault)
# plt.rcParams['figure.figsize']  = [19.2, 14.4]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = [9.6, 7.2]

# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['savefig.facecolor'] = 'white'
# plt.rcParams['axes.facecolor'] =  'white'


# * ######################## DIRECTORIES & FILES ###############################

base = pathlib.Path('C:/Users/chris/Videos/EEG_data/proj2')

if base != (wd := pathlib.Path.cwd()):
    print(f'- Current working directory: {wd}')
    print(f'- Changing it to: {base}')
    os.chdir(base)


# IMPORT DIRECTORIES
dir_data = pathlib.Path("D:/pdmfinal/pdmfinal_task1 - Copy")

# EXPORT DIRECTORIES
dir_figs = base/'1-Figures'
dir_figs_montage = dir_figs/'montage'

dir_pickle = base/'2-Pickle_Files'
dir_ICA = base/'3-Fitted_ICA'

for directory in [dir_figs, dir_figs_montage, dir_pickle, dir_ICA]:
    directory.mkdir(exist_ok=True)


# * ######################## GLOBAL VARIABLES ##################################
# !         ########### /!\ # 1250 = stim onset time /!\ ###########

# Pixel size in inches
px = 1 / plt.rcParams['figure.dpi']

figsize = (1280 * px, 1280 * px)

# Sampling frequency in Hz
sfreq = 1000  

template_rad = 600

# Setting the N200 time window: 125-275 ms post stimulu-onset
n200_window = [125, 275] 

# Selecting the first 'nb_comps' svd components  (here: first 10 svd components)
nb_comps = 10
nb_chans = 20


# * ######################## N200 TEMPLATES ####################################
# ! /!\ CHANGE TEMPLATE DIRECTORY

templates_path = str("C:/Users/chris/Videos/N200-NDT Analysis/"\
                            "Template_Files/N200template.mat")

headmodel_path = str("C:/Users/chris/Videos/N200-NDT Analysis/"\
                            "Template_Files/eginn128hm.mat")

wave_template, topo_tpl_arr = generate_templates(template_path = templates_path,
                                                headmodel_path = headmodel_path,  
                                                save_path = base)
                                                
template_img_arr = topo_tpl_arr

df_topo_tpl = pd.DataFrame(topo_tpl_arr)


# * ########################### FUNCTIONS ######################################

def load_data_1():
    for subj in subjects.keys():
        # Get all sessions from each subject
        subj_sessions = {sess.name: sess for sess in subjects[subj].iterdir()}

        # Get EEG and Behavior files from each session
        eeg_files = {sess.name: [f for f in sess.iterdir() if str(
            f).endswith(f_select)][0] for sess in subj_sessions.values()}
        behav_files = {sess.name: [f for f in (
            sess/'behav').iterdir()][0] for sess in subj_sessions.values()}

        # Add each session's files for every session and every subject
        data_dict[subj] = dict()
        for sess in subj_sessions:
            eeg_f = eeg_files[sess]
            behav_f = behav_files[sess]
            task = re.search('task.', str(eeg_f))[0]
            data_dict[subj][sess] = {'eeg': eeg_f,
                                     'behav': behav_f, 'task': task}


def load_data_2(dir_data):
    
    subjects = sorted(set(
                [re.search('s\d{3}', str(f))[0] for f in dir_data.iterdir()])
                )
                
    files = [f for f in dir_data.iterdir() if re.search('_final', str(f))]

    subj_dict = dict()

    for subj in subjects:
        subj_dict[subj] = {
                            'ses1': {'file': None, 'task': None, 'ICA': None},
                            'ses2': {'file': None, 'task': None, 'ICA': None},
                            'subj_info': {}}

        for f in files:
            if subj in str(f):
                ses = re.search('ses\d', str(f))[0]
                task = re.search('task\d', str(f))[0]

                subj_dict[subj][ses]['file'] = f
                subj_dict[subj][ses]['task'] = task

                # files.remove(f)

    return subj_dict


# * ####################### REST OF THE CODE ###################################

data_dict = load_data_2(dir_data)

# pprint(data_dict, depth=3)

subj_list = list(data_dict.keys())

task = 'task1'

subj_ses_list = []

for subj in subj_list:
    for ses in data_dict[subj].keys():
        # if 'ses' in ses and data_dict[subj][ses]['file'] is not None:
        if data_dict[subj][ses].get('task') == task:
            subj_ses_list.append((subj, ses,))
            # subj_ses_list.append((subj, ses, task))


def subj_analysis(subj_ses):
    
    subj, ses = subj_ses
    # print('\t' * 2, f'{" ".join([i for i in subj_ses])}:', sep='')
    session = data_dict[subj][ses]
    subj_ses = f'{subj}_{ses}'
    print(f'{subj_ses}:')

    # Load the .mat file containing the experiment's data
    print('     loading .mat file...')
    mat_file_dict = read_mat(str(session['file']))

    # load the eeg data from the .mat file
    ''' shape = Time (ms) x Channels (μV) x Trials
        slice[:, :-1, :]  -> omit the stimulus channel
        convert into μV     -> 1E-6 '''
        
    eeg_data = mat_file_dict['data'][:, :-1, :] * 1E-6
    # data_dict[subj][ses]['eeg_data'] = eeg_data


    # Electrodes' position
    headmodel = mat_file_dict['hm']

    # [:-1] -> Omit the stimulus channel;  * 0.1 ->
    chan_pos = headmodel['Electrode']['CoordOnSphere'][:-1] * 0.1
    chan_pos = dict(zip([str(i + 1).zfill(2) for i in range(len(chan_pos))],
                        chan_pos))

    # Addtional information

    # button_press = mat_file_dict['button']
    # artifacts = mat_file_dict['artifact']
    # sfreq = mat_file_dict['sr']

    experiment_info = mat_file_dict['expinfo']

    # resp time in ms
    resp_times = experiment_info['rt'].tolist()

    condition = experiment_info['condition']
    unique_cond = sorted(set(condition))

    condition_dict = {}
    for unique in unique_cond:
        indices = np.where(condition == unique)[0]
        indices = (indices[0], indices[-1])
        condition_dict[unique] = indices

    # Add subject's info to data_dict
    subj_info = dict(
                    gender = experiment_info['gender'],
                    age = experiment_info['age'],
                    hand = experiment_info['hand'],
                    vision = experiment_info['vision'],
                    capsize = experiment_info['capsize'])

    data_dict[subj][ses]['chan_pos'] = chan_pos
    data_dict[subj][ses]['resp_times'] = resp_times
    data_dict[subj][ses]['condition'] = condition
    data_dict[subj]['subj_info'] = subj_info
    
    # Load the .mat file containing behavioral data
    # behav_mat = read_mat(str(session['behav']))
    
    # ######################################################################
    # from tabulate import tabulate
    # exp_info_keys = sorted([k for k in experiment_info.keys() if not k.startswith('__')])
    # behave_mat_keys = sorted([k for k in behav_mat.keys() if not k.startswith('__')])
    # print(len(exp_info_keys) == len(behave_mat_keys))
    # print(tabulate(zip(exp_info_keys, behave_mat_keys), headers=['exp_info', 'behave']))
    # experiment_info['condition']
    ########################################################################
    
    
    stim_onset = 1250  # ms
    stim_onset_sec = stim_onset / 1000  # stim. onset in seconds
    
    nb_trials = eeg_data.shape[2]
    
    trial_length = eeg_data.shape[0]  # ms
    
    # individual trial's legth in seconds
    trial_length_sec = trial_length / 1000
    
    # total duration of all trials combined
    total_length = trial_length * nb_trials
    
    # setting up the EEG montage
    nasion = chan_pos['17']
    lpa = chan_pos['48']
    rpa = chan_pos['113']
    
    montage = mne.channels.make_dig_montage(ch_pos=chan_pos, nasion=nasion,
                                            lpa=lpa, rpa=rpa)
    
    montage.plot(show_names=False, show=False)
    plt.savefig(dir_figs_montage/f'{subj_ses}_montage.png', dpi=300)
    
    
    # Combine all the epochs together
    
    info = mne.create_info(list(chan_pos.keys()), sfreq, ch_types='eeg',
                           verbose=None)
    
    epochs_list = []
    
    print('applying filter (1-10 Hz) and baseline(-0.1, 0) to all trials')
    
    for trial_idx in range(nb_trials):
        # transpose the array to get chan. x times
        trial = eeg_data[:, :, trial_idx].T
    
        raw = mne.io.RawArray(trial, info, first_samp=0, copy='auto',
                              verbose=False)
    
        raw.filter(1, 10)
    
        events = mne.make_fixed_length_events(
                                            raw, id=1, start=stim_onset_sec,
                                            stop=None, first_samp=True,
                                            overlap=0.0,
                                            duration=trial_length_sec
                                              - stim_onset_sec)
    
        # epoch = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5,
        #                    baseline=None, verbose=False)
    
        epoch = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5,
                           baseline=(-0.1, 0), verbose=False)
    
        epochs_list.append(epoch)
    
    epochs = mne.concatenate_epochs(epochs_list).set_montage(montage)
    
    # epochs.filter(1, 10)
    
    # epochs.apply_baseline(baseline=(-0.1, 0))
    
    evoked = epochs.average()
    
    
    ''' Create dictionnary of epochs for each condition '''
    # epochs_dict = {}
    # for k in condition.keys():
    #     epochs = data[:, :, slice(*condition[k])]
    #     epochs = mne.EpochsArray(epochs.T, info)
    #     epochs.set_montage(montage)
    #     epochs_dict[k] = epochs
    # # epochs.plot(n_epochs=2, n_channels=20)
    # epochs_dict
    
    
    n200_data = N200_detection(subj_ses, evoked, dir_figs, dir_pickle,
                                n200_window, nb_chans, nb_comps, wave_template,
                                df_topo_tpl, figsize=figsize)
    
    print(f'{subj_ses} DONE', '\n\n')
    data_dict[subj][ses]['n200_data'] = n200_data


indices = [
            slice(0, 10),
            slice(10, 20),
            slice(20, 30),
            slice(30, 40),
            slice(40, 49)
            ]


# for subj_ses in subj_ses_list[indices[0]]:
for subj_ses in subj_ses_list[:]:
    subj_analysis(subj_ses[:2])


data_dict = dict(*pickle_import('DATA_DICT-TASK1-FINAL.pickle'))



df = pd.DataFrame()
for i in data_dict:
    subj = pd.DataFrame(data_dict[i]['subj_info'], index=[i])
    df = pd.concat([df, subj])
# 
# 
# df.to_excel(base/'participants.xlsx')


############# AVERAGING COMPONENTS FOR EVOKED GIFs
combined_u = pd.DataFrame()
combined_v = pd.DataFrame()

for subj in subj_ses_list[:]:
    subj, ses = subj[:2]
    for ses in data_dict[subj].keys():
        if data_dict[subj][ses].get('task') == task:
            comp = data_dict[subj][ses]['n200_data']['picked_comp']
            v = data_dict[subj][ses]['n200_data']['v_comps'][comp]
            u = data_dict[subj][ses]['n200_data']['u_comps'][comp]
            combined_u[subj] = u
            combined_v[subj] = v
            
combined_u['mean'] = combined_u.mean(axis=1)
combined_v['mean'] = combined_v.mean(axis=1)

# combined_u.index = # timepoints 
# combined_v.index = # timepoints

sns.lineplot(x=combined_u['mean'].index, y=combined_u['mean'].values)
plt.show(block=False)
plt.close('all')

#


#################



## PLOTING ALL SELECTED COMP TOGETHER ##
plt.figure(figsize=figsize)
for subj in data_dict:
    for ses in data_dict[subj].keys():
        if data_dict[subj][ses].get('task') == task:
            comp = data_dict[subj][ses]['n200_data']['picked_comp']
            fig_data = data_dict[subj][ses]['n200_data']['u_comps'][comp, :]
            timepoints = np.linspace(125, 276, 151)
            plt.plot(timepoints, fig_data)
plt.savefig(base/'selected_comps-combined.png')

# pprint(data_dict)

# ##############################################################################
# ##############################################################################
# ##############################################################################



# # ########################### REGRESSION ANALYSIS ##############################
# 
# n200_dict = {re.search('s\d{3}_ses\d', f.name)[0]:f for f in dir_pickle.iterdir()}
# n200_dict = {k:dict(*pickle_import(v)) for k,v in n200_dict.items()}
# 
# for k,v in n200_dict.items():
#     data_dict[k[:4]][k[5:]]['n200_data'] = v
# 
# 
# 
# 
# reg_data = {}
# df_correlation = pd.DataFrame()
# df_correlation_selected = pd.DataFrame()
# 
# # task = 'task1'
# 
# for subj in subj_ses_list[:]:
#     subj, ses = subj[:2]
#     for ses in data_dict[subj].keys():
#         if data_dict[subj][ses].get('task') == task:
# 
#             resp_time = np.nanpercentile(
#                 data_dict[subj][ses]['resp_times'], 10)
# 
#             # resp_time = round(mean_resp_time, 3)
# 
#             latency = data_dict[subj][ses]['n200_data']['u_peak']['latency']
# 
#             reg_data[subj] = (latency, resp_time)
# 
#             df = data_dict[subj][ses]['n200_data']['df_corr']
#             df_correlation = pd.concat([df_correlation, df])
# 
#             selected = df.sort_values('prod', ascending=False).iloc[0:1]
#             df_correlation_selected = pd.concat(
#                 [df_correlation_selected, selected])
# 
# df_reg_data = pd.DataFrame(reg_data).T
# df_reg_data.columns = ['latency', 'resp. time (10th %ile)']
# df_reg_data.to_excel(base/'reg_data.xlsx')
# 
# df_correlation_selected.to_excel(base/'Selected_SVD_Components.xlsx')
# 
# 
# # plotting resp time
# plt.figure(figsize=figsize)
# plt.bar(df_reg_data.index, df_reg_data['resp. time (10th %ile)'])
# plt.title('NDTs (10th Percentile Resp. Time)', fontsize=24)
# plt.ylabel('NDTs (ms)', fontsize=18)
# plt.xlabel('Participants', fontsize=18)
# plt.xticks(rotation=90)
# plt.savefig(base/'NDTs_dist.png')
# # # pd.set_option('display.max_rows', None)
# # 
# # pprint(reg_data)
# # 
# # 
# # 
# # 
# # 
# # 
# x = np.array([i[0] for i in reg_data.values()])
# y = np.array([i[1] for i in reg_data.values()])
# 
# 
# reg = reg_analysis(x, y, save_dir=base, show=True)
# 
# reg.predict(np.array([[120]]))
# 
# 
# pickle_export(data_dict, base, 'DATA_DICT-TASK1-FINAL')
# data_dict[subj]['ses1']
# 
# 
# 
# 
# import statsmodels.api as sm
# import seaborn as sns
# 
# df_reg_data
# 
# sns.regplot(x='latency', y='resp. time (10th %ile)', data=df_reg_data)
# plt.savefig(base/'sns_regplot.png', dpi=300)
# 
# X = sm.add_constant(df_reg_data['latency']) # adding a constant
# Y = df_reg_data['resp. time (10th %ile)']
# 
# 
# model = sm.OLS(Y, X).fit()
# predictions = model.predict(X)
# 
# model_summary = model.summary()
# 
# with open(base/'LinReg-Summary.txt', 'w') as f:
#     f.write(model_summary.as_text())
# 
# 
# mean_age = np.mean([data_dict[i]['subj_info']['age'] for i in data_dict])
# max_age = np.max([data_dict[i]['subj_info']['age'] for i in data_dict])
# min_age = np.min([data_dict[i]['subj_info']['age'] for i in data_dict])
# gender_count = np.unique([data_dict[i]['subj_info']['gender'] for i in data_dict], return_counts=True)
# 
# # ################################################################################
# 

# %%
