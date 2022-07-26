# %%
# ! IMPORTS
# * ===========================================================================
from CommonFuncs import (pickle_import, pickle_export, show_all, make_montage, 
                            modify_folder, generate_templates,N200_detection,
                            ajdust_topo_plot, reg_analysis, select_files, 
                            data_struct, tree)
import itertools
# from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
import pandas as pd
import pathlib
import pickle
from PIL import Image
from pprint import pprint
import psutil
from pymatreader import read_mat
import re
import scipy.io
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import warnings


# * ########################## GLOBAL PARAMETERS ##############################

mne.set_config('MNE_USE_CUDA', 'true')
mne.set_config('MNE_BROWSER_USE_OPENGL', 'true')
mne.get_config()
mne.set_log_level(verbose='WARNING')
warnings.filterwarnings('ignore', category=DeprecationWarning)
# mne.viz.get_browser_backend()
# mne.viz.get_3d_backend()

plt.rcParams['figure.dpi'] = 100
# plt.rcParams['figure.figsize']  = [19.2, 14.4]
plt.rcParams['figure.figsize'] = [9.6, 7.2]
plt.rcParams.update(matplotlib.rcParamsDefault)

# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['savefig.facecolor'] = 'white'
# plt.rcParams['axes.facecolor'] =  'white'

#  * ######################### DATA INFORMATION ################################

event_ids = {
    'Match Resp.': 1,
    'Non-Match Resp.': 2,
    'Neutral Correct': 20,
    'Neutral Wrong': 25,
    'B': 30,
    'C': 31,
    'F': 32,
    'H': 33,
    'J': 34,
    'K': 35,
    'A': 36,
    'Rest': 37,
    'Prepare': 38,
    'Sound Wrong': 40,
    'Sound Penalty': 45,
    'Sound Correct': 60,
    'Sound Bonus': 65,
}

ch_names = ['1A1', '1A2', '1A3', '1A4', '1A5', '1A6', '1A7', '1A8', '1A9',
            '1A10', '1A11', '1A12', '1A13', '1A14', '1A15', '1A16', '1A17',
            '1A18', '1A19', '1A20', '1A21', '1A22', '1A23', '1A24', '1A25',
            '1A26', '1A27', '1A28', '1A29', '1A30', '1A31', '1A32', '1B1',
            '1B2', '1B3', '1B4', '1B5', '1B6', '1B7', '1B8', '1B9', '1B10',
            '1B11', '1B12', '1B13', '1B14', '1B15', '1B16', '1B17', '1B18',
            '1B19', '1B20', '1B21', '1B22', '1B23', '1B24', '1B25', '1B26',
            '1B27', '1B28', '1B29', '1B30', '1B31', '1B32', '1C1', '1C2',
            '1C3', '1C4', '1C5', '1C6', '1C7', '1C8', '1C9', '1C10', '1C11',
            '1C12', '1C13', '1C14', '1C15', '1C16', '1C17', '1C18', '1C19',
            '1C20', '1C21', '1C22', '1C23', '1C24', '1C25', '1C26', '1C27',
            '1C28', '1C29', '1C30', '1C31', '1C32', '1D1', '1D2', '1D3', '1D4',
            '1D5', '1D6', '1D7', '1D8', '1D9', '1D10', '1D11', '1D12', '1D13',
            '1D14', '1D15', '1D16', '1D17', '1D18', '1D19', '1D20', '1D21',
            '1D22', '1D23', '1D24', '1D25', '1D26', '1D27', '1D28', '1D29',
            '1D30', '1D31', '1D32', '1E1', '1E2', '1E3', '1E4', '1E5', '1E6',
            '1E7', '1E8', '1E9', '1E10', '1E11', '1E12', '1E13', '1E14', '1E15',
            '1E16', '1E17', '1E18', '1E19', '1E20', '1E21', '1E22', '1E23',
            '1E24', '1E25', '1E26', '1E27', '1E28', '1E29', '1E30', '1E31',
            '1E32', '1F1', '1F2', '1F3', '1F4', '1F5', '1F6', '1F7', '1F8',
            '1F9', '1F10', '1F11', '1F12', '1F13', '1F14', '1F15', '1F16',
            '1F17', '1F18', '1F19', '1F20', '1F21', '1F22', '1F23', '1F24',
            '1F25', '1F26', '1F27', '1F28', '1F29', '1F30', '1F31', '1F32',
            '1G1', '1G2', '1G3', '1G4', '1G5', '1G6', '1G7', '1G8', '1G9',
            '1G10', '1G11', '1G12', '1G13', '1G14', '1G15', '1G16', '1G17',
            '1G18', '1G19', '1G20', '1G21', '1G22', '1G23', '1G24', '1G25',
            '1G26', '1G27', '1G28', '1G29', '1G30', '1G31', '1G32', '1H1',
            '1H2', '1H3', '1H4', '1H5', '1H6', '1H7', '1H8', '1H9', '1H10',
            '1H11', '1H12', '1H13', '1H14', '1H15', '1H16', '1H17', '1H18',
            '1H19', '1H20', '1H21', '1H22', '1H23', '1H24', '1EX1', '1EX2',
            '1EX3', '1EX4', '1EX5', '1EX6', '1EX7', '1EX8', 'Status']

bad_channels = {
    'sess_01': ['1D27'],
    'sess_02': ['1B16', '1B19'],
    'sess_03': ['1C24', '1D8', '1F11', '1F5', '1H19'],
    'sess_04': ['1D25'],
    'sess_05': [],
    'sess_06': ['1A12', '1D8', '1E21', '1G7'],
    'sess_07': ['1B19'],
    'sess_08': ['1A16', '1B1', '1B19', '1B8', '1C15', '1D29', '1G7', '1H10'],
    'sess_09': ['1F15'],
    'sess_10': ['1F2', '1F8', '1H5'],
    'sess_11': ['1B19'],
    'sess_12': ['1A10', '1A11', '1A18', '1A19', '1A2', '1A22', '1A25', '1A26', '1A3', '1A30', '1E16', '1G20'],
    'sess_13': ['1B19', '1C9', '1F2', '1G32'],
    'sess_14': ['1A26', '1A30', '1A31', '1D20', '1G1'],
    'sess_15': ['1E32', '1F8'],
    'sess_16': ['1B19', '1D20', '1D21'],
    'sess_17': ['1A17', '1A21', '1A24', '1A30', '1A31', '1B19', '1C2', '1E8'],
    'sess_18': ['1F2'],
    'sess_19': ['1D23', '1H6'],
    'sess_20': [],
    'sess_21': ['1A19', '1A30', '1C6', '1D21'],
    'sess_22': ['1B12', '1D25'],
    'sess_23': ['1A32', '1B12', '1B19', '1F14', '1F4'],
    'sess_24': ['1D14'],
    'sess_25': ['1C7', '1F4', '1G24', '1G32'],
    }

# info_xl = base/'experiment_info.xlsx'
# with pd.ExcelWriter(info_xl) as writer:
#     pd.Series(ch_names).to_excel(writer,  sheet_name='Sheet1')
#     pd.Series(event_ids).to_excel(writer, sheet_name='Sheet2')
#     pd.Series(bad_channels).to_excel(writer, sheet_name = 'Sheet3')


# * ######################## DIRECTORIES & FILES ###############################

base = pathlib.Path('C:/Users/chris/Videos/EEG_data/proj1/')
# base = pathlib.Path('Proj1-Analysis')

if base != (wd := pathlib.Path.cwd()):
    print(f'- Current working directory: {wd}')
    print(f'- Changing it to: {base}')
    os.chdir(base)

# IMPORT DIRECTORIES
dir_data = base/'OLD/1-Data'  # /!\ CHANGE FOLDER (REMOVE "OLD")
dir_sessions = dir_data/'Sessions'


# EXPORT DIRECTORIES
dir_figs = base/'1-Figures'
dir_figs_montage = dir_figs/'montage'

dir_pickle = base/'2-Pickle_Files'
dir_ICA = base/'3-Fitted_ICA'

for directory in [dir_figs, dir_figs_montage, dir_pickle, dir_ICA]:
    directory.mkdir(exist_ok=True)

# * ######################## GLOBAL VARIABLES ##################################

px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

figsize = (1280 * px, 1280 * px)

template_rad = 496

# setting the N200 time window: 125-275 ms post stimulu-onset
n200_window = [125, 275]

# Set the number of SVD components to be selected from the U and V matrices
nb_comps = 10

# Set the number of channels for the N200 plots:
#   selected by their amount of "contributions" to the selected component in 
#   the U matrix
nb_chans = 20


# * ######################## N200 TEMPLATES ####################################
# ! /!\ CHANGE TEMPLATE DIRECTORY

templates_path = 'C:/Users/chris/OneDrive - UvA/Pinier - '\
    'Internship/Data/templates/N200template.mat'

headmodel_path = 'C:/Users/chris/OneDrive - UvA/Pinier - '\
    'Internship/Data/templates/eginn128hm.mat'

wave_template, topo_tpl_arr = generate_templates(template_path = templates_path,
                                                headmodel_path = headmodel_path,  
                                                save_path = base)

template_img_arr = topo_tpl_arr

df_topo_tpl = pd.DataFrame(topo_tpl_arr)

# # Checking pixel values in grayscale
# val_counts = np.unique(template_img_arr, return_counts=True)
# pprint( {px_val:count for px_val, count in zip(val_counts[0], val_counts[1])} )


# * ############################ FUNCTIONS #####################################

def load_data(sessions_dir):
    
    sess_folders = [f.path for f in os.scandir(sessions_dir) if f.is_dir()]
    
    sessions = [[os.path.join(session, f) for f in os.listdir(session)]
                for session in sess_folders]
    
    sessions_bdf = [[f for f in session if f.endswith('.bdf')]
                    for session in sessions]
    
    sessions_elp = [[f for f in session if f.endswith('.elp')]
                    for session in sessions]
    
    bdf_files = {}
    for idx_sess, session in enumerate(sessions_bdf):
        for idx_file, file in enumerate(session):
            sess_name = f'sess_{str(idx_sess+1).zfill(2)}-{idx_file+1}'
            bdf_files[sess_name] = file
    
    
    elp_files = {}
    for idx_sess, session in enumerate(sessions_elp):
        if len(session) == 1:
            for idx_file, file in enumerate(session):
                sess_name = f'sess_{str(idx_sess+1).zfill(2)}'
                elp_files[sess_name] = file
        else:
            print(f'more than 1 montage file have been found in {session}')
            elp_files[sess_name] = []
            for idx_file, file in enumerate(session):
                sess_name = f'sess_{str(idx_sess+1).zfill(2)}-{idx_file+1}'
                elp_files[sess_name].append(file)
    
    return bdf_files, elp_files


def prepro(sess_names):
    
    data = dict()
    
    df_events = pd.DataFrame()
    df_selected_ev = pd.DataFrame()
    
    cleaned_events = dict()
    resp_times_dict = dict()
    accuracy_dict = dict()
    interstim_times_dict = dict()
    rejections = dict()
    u_matrices = dict()
    
    
    for sess_name in sess_names:
        
        data[sess_name] = {
                            'events': None,
                            'event_counts': None,
                            'event_valids': None,
                            'event_invalids': None,
                            'events_cleaned': None,

                            'bipolar_ref': None,

                            'epochs': None,
                            'rejections': {'channels': None, 'epochs': None,
                                            'ICA_excluded': None},
                            # 'bad_epochs': None,
                            'cleaned_epochs': None,

                            'evoked': None,

                            'duration': None,
                            'bad_ch': None,
                            'subject_info': {'subj_id': None, 'sex': None, 
                                                'age': None, 'hand': None},
                            'resp_times': None,
                            }

        # * ### PREPROCESSING ###
        print(f'ANALYZING {sess_name}\n'\
                '  >  preprocessing (averaging, filtering, ICA)')

        bdf_f = bdf_files[sess_name]
        elp_f = elp_files[sess_name[:-2]]
        

        raw = mne.io.read_raw_bdf(bdf_f, preload=True, verbose=True)
        samp_freq = raw.info['sfreq']
        
        
        std_montage = False

        if std_montage:
            montage = mne.channels.make_standard_montage('biosemi256')
            montage.rename_channels(
                    {old: new for old, new in zip(montage.ch_names,
                                                    ch_names[:-1])})
        else:
            montage = mne.channels.read_dig_polhemus_isotrak(elp_f,
                                                             ch_names = ch_names
                                                             + ['unknown'])

        # * MONTAGE *
        print('\n' + 'SETTING MONTAGE:' + '\n')
        raw.set_montage(montage)

        raw.info['bads'].extend(bad_channels[sess_name[:-2]])

        # data[sess_name]['raw'] = raw.copy()
        data[sess_name]['duration'] = raw.times[-1]

        # * Plot 2D Montage *
        raw.plot_sensors(kind='topomap', ch_type=None, title=sess_name[:-2],
                        show_names=False, ch_groups='position', block=False,
                        show=False)
        
        f_name = sess_name + '_2D-montage.png'
        plt.savefig(dir_figs_montage/f_name, dpi=300)


        # * Plot 3D Montage *
        # plt.figure(figsize=figsize)
        # raw.plot_sensors(kind='3d', ch_type=None, title=sess_name[:-2],
        #                 show_names=raw.info['bads'], ch_groups='position',
        #                 block=False, show=False)

        # f_name = sess_name + '_3D-montage.png'
        # plt.savefig(dir_figs_montage/f_name, figsize=figsize)



        events = mne.find_events(raw, stim_channel='Status', shortest_event=1,
                                    initial_event=True, verbose=True)

        data[sess_name]['events'] = events

        # Average Reference
        print('\n' + 'AVERAGING:' + '\n')
        raw.set_eeg_reference(ref_channels='average')

        # Bipolar Reference
        print('\n' + 'SETTING BIPOLAR REFERENCE:' + '\n')
        anode, cathode = '1EX4', '1EX5'
        data[sess_name]['bipolar_ref'] = [anode, cathode]
        
        # Plot electrodes used for bipolar_ref in 3D
        raw.plot_sensors(kind='3d', ch_type=None, title=sess_name[:-2],
                        show_names=[anode, cathode], ch_groups='position',
                        block=False, show=False)

        f_name = sess_name + '_Sim-EOG.png'
        plt.savefig(dir_figs_montage/f_name, dpi=300)

        
        mne.set_bipolar_reference(raw, anode, cathode, ch_name='sim_EOG',
                                    ch_info=None, drop_refs=True, copy=False,
                                    verbose=False)
                                    
        raw.set_channel_types({'sim_EOG':'eog'}, verbose=False)
        

        # Bandpass Filter: 0.1 - 100 Hz
        print('\n' + 'BANDPASS FILTER 0.1-100 Hz ...' + '\n')
        raw.filter(l_freq=.1, h_freq=100)
        print()

        # prepro_fname = f'{sess_name}-preprocessed_eeg.fif'
        # raw.save(prepro_raw_folder/prepro_fname)

        # ICA
        print('\n' + 'FITTING ICA:' + '\n')
        
        # REPLACE n_components with 0.999999
        ica = mne.preprocessing.ICA(n_components=0.8, noise_cov=None,
                                    random_state=97, method='fastica', max_iter='auto')

        ica.fit(raw)
        eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name='sim_EOG')
        print('\t'+ f"Rejected EOG components' indices: {eog_inds}")
        
        ica.exclude = eog_inds

        f_name= sess_name + '_fitted-ica.fif'
        ica.save(dir_ICA/f_name)

        raw = ica.apply(raw)

        # data[sess_name]['ICA_fitted'] = ica
        data[sess_name]['rejections']['ICA_excluded'] = eog_inds
        
        # data[sess_name]['ICA_applied'] = raw.copy()


        # barplot of ICA component "EOG match" scores
        # plt.figure()
        # ica.plot_scores(eog_scores, show=False)
        # ica.plot_sources(raw, show_scrollbars=False)
        # ica.plot_components(picks=[i for i in range(10)])
        # plt.show(block=False)
        # plt.close('all')

        # plot diagnostics
        # ica.plot_properties(raw, picks=eog_inds)

        # plot ICs applied to raw data, with EOG matches highlighted
        # ica.plot_sources(raw, show_scrollbars=False)

        # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        # ica.plot_sources(eog_evoked)

        # Bandpass Filter: 1 - 10 Hz
        print('BANDPASS FILTER 1-10 Hz ...')
        raw.filter(l_freq=1, h_freq=10)
        
        print('\n\n')
        

        # ############################ EVENT CLEANING ##############################

        print('\n  >  Cleaning Events')

        responses = ['Match Resp.', 'Non-Match Resp.']
        stimuli = ['A','B', 'C', 'F', 'H', 'J', 'K']
        correct_feedback = ['Sound Correct', 'Sound Bonus', 'Neutral Correct']
        event_ids_reversed = {v:k for k,v in event_ids.items()}

        # events = data[sess_name]['events']
        df = pd.DataFrame(events, columns=['sample_nb', 'stim_ch_val', 'event_id'])

        # convert sample nb into event onset time in seconds
        event_time = df['sample_nb'] / samp_freq

        # calculate time difference between every events (event_time - previous event_time)
        time_diff = event_time.diff()

        df.insert(1, 'event_time', event_time)
        df.insert(2, 'time_diff', time_diff)

        df['session'] = [sess_name] * len(events)

        # replace event ID number by event name
        df['event_id'].replace(to_replace=event_ids_reversed, inplace=True)

        # add all events of session to df_events
        df_events = pd.concat([df_events, df], axis=0)

        # Add event information in the 'data' dictionnary

        # event_id_counts = df['event_id'].value_counts()
        # valid_ids = event_id_counts[event_id_counts.index.isin(event_ids)]
        # invalid_ids = event_id_counts[~event_id_counts.index.isin(event_ids)]

        data[sess_name]['event_counts'] = (event_id_counts := 
                                           df['event_id'].value_counts())

        data[sess_name]['event_valids'] = event_id_counts[
                                            event_id_counts.index.isin(event_ids)]

        data[sess_name]['event_invalids'] = event_id_counts[
                                            ~event_id_counts.index.isin(event_ids)]


        # Only select stimuli presetend right before participant's responses (button presses)
        # (e.g. not the first 2 stimuli presented at start of each trial)
        df_stim = df[df['event_id'].shift(-1).isin(responses)]

        # Make sure that only stimuli are selected (e.g. not auditory feedback or double button press)
        df_stim = df_stim[df_stim['event_id'].isin(stimuli)]

        # Only select participant's responses made after a stimulus' presentation
        df_resp = df.iloc[df_stim.index + 1]

        # Selecting indices of correct feedback events
        df_correct_fb = df[(df['event_id'].shift(+1).isin(responses)) & (df['event_id'].isin(correct_feedback))]
        correct_fb_inds = df_correct_fb.index.to_list()
        # df_correct_fb = df[df['event_id'].shift(+1).isin(responses)] # Select feedback events
        # correct_fb_inds = df[df['event_id'].isin(correct_feedback)].index.to_list()

        nb_correct = len(correct_fb_inds)
        nb_resp = len(df_resp)

        accuracy = round(nb_correct / nb_resp, 4)
        data[sess_name]['accuracy'] = accuracy
        accuracy_dict[sess_name] = accuracy

        # Calculate response times
        resp_times = df_resp['time_diff']
        resp_time_stats = resp_times.describe()
        resp_times_dict[sess_name] = {
                                        'all': resp_times,
                                        'df': df_resp,
                                        'stats': resp_time_stats
                                        }
        data[sess_name]['resp_times'] = resp_times

        # Calculate mininmum time between stimuli for each session
        interstim_times = df_stim.copy()
        interstim_times['time_diff'] = interstim_times['event_time'].diff()
        interstim_times_stats = interstim_times['time_diff'].describe()
        interstim_times_dict[sess_name] = {
                                            'all': interstim_times['time_diff'],
                                            'df': interstim_times,
                                            'stats': interstim_times_stats
                                            }

        df_combined = pd.concat([df_stim, df_resp], axis=0).sort_index()
        df_combined.loc[df_combined['event_id'].isin(stimuli), 'time_diff'] = None
        df_combined.rename(columns={'time_diff': 'resp_time'}, inplace=True)

        df_selected_ev = pd.concat([df_selected_ev, df_combined], axis=0)


        check = list(df_combined.index)
        if len(check) %2 != 0: print('length of stimuli-responses DF not even')

        check = [ ( check[check.index(i) + 1] - i ) == 1 for i in check[::2] ]
        
        if not all(check): 
            print('/!\ not all stimulus-response pairs have a consecutive index')


        event_array = df_stim.loc[:,['sample_nb', 'stim_ch_val', 'event_id']]
        event_array['event_id'].replace(to_replace=event_ids, inplace=True)
        event_array = event_array.to_numpy(dtype='int')

        cleaned_events[sess_name] = event_array
        data[sess_name]['events_cleaned'] = event_array

        print(sess_name + ': events processed')

        # ############################ EPOCHING ##############################
        print('\n' + 'EPOCHING:' + '\n')

        # Adding EXs electrodes to the bad / rejected channels
        rej_ch = raw.info['bads'].copy() + ['1EX1','1EX2', '1EX3','1EX6', '1EX7', 
                                            '1EX8','bipolar_ref', 'Status']
                                            
        picked_chs = [ch for ch in raw.ch_names if ch not in rej_ch]

        epochs = mne.Epochs(raw, data[sess_name]['events_cleaned'],
                            event_id = event_ids, tmin = -0.2, tmax = 0.3,
                            baseline = (-0.1, 0), detrend = 1,
                            on_missing = 'warn', preload = True,
                            picks = picked_chs)


        data[sess_name]['epochs'] = epochs.copy()

        ### SELECTING EPOCHS FOR REJECTION

        df_epochs = epochs.to_data_frame()

        epoch_inds = [idx for idx in df_epochs['epoch'].unique()]
        epochs_info = {}

        for idx in epoch_inds:
            # Create a DF per epoch
            df = df_epochs[df_epochs['epoch'] == idx]
                
            # Select only columns with channel info 
            df = df.iloc[:, 3:]  
            
            # Total nb of channels
            nb_of_channels = df.shape[1]  
            
            # nb of times each channel has values higher than +-100 µV
            ch_counts = df[(df <= -100) | (df >= 100)].count()
            
            # Select only those that display such values at least once  
            ch_counts = ch_counts[ch_counts > 0].to_dict()  

            bad_ch_ratio = len(ch_counts) / nb_of_channels

            epochs_info[idx] = []
            epochs_info[idx].append(ch_counts)
            epochs_info[idx].append(bad_ch_ratio)

        rejections[sess_name] = {'channels': None, 'epochs': None} # REMOVE
        reject_crit = [
                        20,  # Reject entire channel if, for over 20 trials, amplitudes +- 100 microV
                        0.3,  # Reject entire trial if ~30% electrodes have some amplitudes over +- 100 microV
                        ]


        # Reject entire channel if, for over 20 trials, amplitudes +- 100 microV
        rej_ch_list = [list(info[0].keys()) for info in epochs_info.values()] # list of bad channels per epoch
        rej_ch_list = list(itertools.chain(*rej_ch_list))  # Flatten the list
        rej_ch_list = pd.Series(rej_ch_list).value_counts()  # Nb of trials where channels display values more than +-100 µV
        rej_ch_list = rej_ch_list[rej_ch_list > reject_crit[0]].index.to_list()


        # Reject entire trial if ~30% electrodes have some amplitudes over +- 100 microV
        rej_epoch_list = [idx for idx, info in epochs_info.items() if info[1] > reject_crit[1]]

        # rejections[sess_name]['channels'] = rej_ch_list # REMOVE
        # rejections[sess_name]['epochs'] = rej_epoch_list # REMOVE

        data[sess_name]['rejections']['channels'] = rej_ch_list
        data[sess_name]['rejections']['epochs'] = rej_epoch_list


        raw.info['bads'].extend(rej_ch_list)
        data[sess_name]['bad_ch'] = raw.info['bads']


        # ### FINAL EPOCHING
        rej_ch += rej_ch_list
        picked_chs = [ch for ch in raw.ch_names if ch not in rej_ch]

        epochs = mne.Epochs(raw, cleaned_events[sess_name], event_id=event_ids,
                            tmin= -0.150, tmax= 0.400, baseline=(None, 0),
                            detrend=1, on_missing='warn', preload=True,
                            picks=picked_chs)

        # epochs.drop_channels(rej_ch_list)
        epochs.drop(rej_epoch_list)
        data[sess_name]['cleaned_epochs'] = epochs

        evoked = epochs.average()
        data[sess_name]['evoked'] = evoked

        pickle_export(data=data[sess_name], path=dir_pickle, 
                    f_name=f'{sess_name}-DataDict')

        plt.close('all')
        print('\n\n')
        
        return data


# * ################# PIPELINE PROCESS, ONE FILE AT A TIME ######################

bdf_files, elp_files = load_data(dir_sessions)


# Removing the first 5 participants from analysis (sess 01-1 to 04-2)
sess_names = list(bdf_files.keys())[8:] 

data_dict = prepro(sess_names)

for sess_name in sess_names[:2]:
    print(f'Extracting N200 from {sess_name}...')
    evoked = data_dict[sess_name]['evoked']
    data[sess_name]['N200_data'] = N200_detection(sess_name, evoked, dir_figs,
                                                n200_window, nb_chans,
                                                nb_comps, wave_template, 
                                                df_topo_tpl, figsize=figsize,
                                                dir_pickle=dir_pickle)
    print('\n\n')

list(data['sess_05-1'].keys())

data['sess_05-1']['events']
data['sess_05-1']['event_counts']
data['sess_05-1']['event_valids']
data['sess_05-1']['event_invalids']
data['sess_05-1']['events_cleaned']

data['sess_05-1']['bipolar_ref']

data['sess_05-1']['epochs']
data['sess_05-1']['rejections']
data['sess_05-1']['cleaned_epochs']
data['sess_05-1']['evoked']





################################################################################
################################################################################
################################################################################

# 
# n200_data_dict = [dict(*pickle_import(f)) for f in dir_pickle.iterdir() if 
#                     'N200' in f.name]
# 
# n200_data_dict = {k:v for k,v in zip(sess_names, n200_data_dict)}
# 
# for sess_name in sess_names:
#     data[sess_name]['N200_data'] = n200_data_dict[sess_name]
# 
# 
# 
# ## PLOTING ALL SELECTED COMP TOGETHER ##
# plt.figure(figsize=figsize)
# 
# for subj in data_dict:
#     comp = data_dict[subj]['N200_data']['picked_comp']
#     fig_data = data_dict[subj]['N200_data']['u_comps'][comp, :]
#     timepoints = np.linspace(125, 276, 39)
#     plt.plot(timepoints, fig_data)
# 
# # plt.savefig(base/'selected_comps-combined.png')
# plt.show()
# 
# 
# plt.close('all')
# 
# 
# 
# reg_data = {}
# df_correlation = pd.DataFrame()
# df_correlation_selected = pd.DataFrame()
# 
# for subj in subj_list:
#     resp_time = np.nanpercentile(data[subj]['resp_times'], 10) * 1000
# 
#     # resp_time = round(mean_resp_time, 3)
# 
#     latency = data[subj]['N200_data']['u_peak']['latency']
# 
#     reg_data[subj] = (latency, resp_time)
# 
#     df = data[subj]['N200_data']['df_corr']
#     df_correlation = pd.concat([df_correlation, df])
# 
#     selected = df.sort_values('prod', ascending=False).iloc[0:1]
#     df_correlation_selected = pd.concat(
#     [df_correlation_selected, selected])
# 
# df_reg_data = pd.DataFrame(reg_data).T
# df_reg_data.columns = ['latency', 'resp. time (10th %ile)']
# df_reg_data.to_excel(base/'reg_data.xlsx')
# 
# df_correlation_selected.to_excel(base/'Selected_SVD_Components.xlsx')
# 
# reg_data
# df_correlation_selected
# 
# 
# x = np.array([i[0] for i in reg_data.values()])
# y = np.array([i[1] for i in reg_data.values()])
# 
# reg = reg_analysis(x, y, base, show=True)
# 
# 
# 
# 
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
# 
# with open(base/'LinReg-Summary.txt', 'w') as f:
#     f.write(model_summary.as_text())
# 
# print(model_summary)
# 
# plt.figure(figsize=figsize)
# plt.scatter(df_reg_data['latency'], Y)
# plt.plot(df_reg_data['latency'], model.predict(), 'orange')
# plt.show(block=False)
# 
# 
# 
# ################################################################################
# ################################################################################
# ################################################################################
# 
# ############# PICKLE EXPORT #################
# to_pickle = [data, df_events, df_selected_ev, cleaned_events, resp_times_dict,
#                 accuracy_dict, interstim_times_dict, rejections, u_matrices]
# 
# export_path = pathlib.Path('C:/Users/chris/Videos/EEG_data/test')
# pickle_export(data, f_name='data-sess_12(1)-12(2)-13', path=export_path)
# #############################################
# 
# 
# 
# 
# df_evoked = evoked.to_data_frame().T
# times = df_evoked.loc['time']
# df_evoked.drop(index='time', inplace=True)
# df_evoked.columns = times
# ch_names = df_evoked.index.to_list()
# 
# 
# df_evoked_arr = pd.DataFrame(evoked_arr) * 10**6
# df_evoked_arr.index = ch_names
# df_evoked_arr.columns = times
# 
# # df_evoked
# # df_evoked_arr
# 
# df_u = pd.DataFrame(u_matrices[sess_name], index=ch_names, columns=times)
# df_u.min().sort_values()
# 
# # ##############################################################################
# # ############## PLOT ELECTRODES THAT LIKELY RECORDED N200 #####################
# 
# # lowest values on interval 0.9 - 2.1 sec post stimulus
# 
# df_n200 = df_evoked.loc[:, 90:210].min(axis=1).sort_values()
# threshold = -5
# n200_chs = df_n200[df_n200 <= -5].index.to_list()
# 
# plt.plot(df_evoked.loc[n200_chs, :].T)
# plt.show(block=False)
# 
# # ############## PLOT ELECTRODES THAT LIKELY RECORDED N200 #####################
# # ##############################################################################
# 
# 
# 
# 
# 
# 
# 
# with matplotlib.rc_context({'backend': 'agg', 'figure.autolayout': True, 'figure.dpi': 250}):
#     # evoked.plot(spatial_colors=True, window_title=sess_name)
#     for i in range(10):
#         plt.figure()
#         plt.plot(u.T[i])
#         plt.show(block=False)
# 
#     print('DONE')
# 

# %%
