from N200_extraction.py import (pickle_import, pickle_export, generate_templates,
                          N200_detection)
import itertools
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import pathlib
from pprint import pprint
from pymatreader import read_mat
import re
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import time
import warnings
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor


logging.basicConfig(level=logging.INFO)

from my_funcs1 import fc # ! TO BE REMOVED


# * ############################ GLOBAL PARAMETERS ############################

# plt.isinteractive()
# plt.ioff()
# %matplotlib qt


# mne.set_config('MNE_USE_CUDA', 'true')
# mne.set_config('MNE_BROWSER_USE_OPENGL', 'true')
mne.get_config()
mne.set_log_level(verbose='WARNING')
warnings.filterwarnings('ignore', category=DeprecationWarning)
# mne.viz.get_browser_backend()
# mne.viz.get_3d_backend()

plt.rcParams.update(matplotlib.rcParamsDefault)
# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['figure.figsize'] = [9.6, 7.2]


# * ############################ GLOBAL VARIABLES #############################

px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

figsize = (1280 * px, 1280 * px)

# * Head radius of the topographic template
template_rad = 496

# * setting the N200 time window: 125-275 ms post stimulu-onset
n200_window = [125, 275]

# * Set the number of SVD components to be selected from the U and V matrices
nb_comps = 10

# * Set the number of channels for the N200 plots:
''' Selected by their "amount of contribution" to the selected component in
    the U matrix '''
nb_chans = 20



# * ########################### DIRECTORIES & FILES ###########################
proj_n = 1
base = pathlib.Path(os.getcwd())/f'Proj{proj_n}'

# * IMPORT DIRECTORIES
dir_data = base/f'{proj_n}.1-Data'
dir_sessions = dir_data/'Sessions'

# * EXPORT DIRECTORIES
dir_figs = base/f'{proj_n}.2-Figures'
dir_figs_montage = dir_figs/'montage'

dir_pickle = base/f'{proj_n}.3-Pickle_Files'
dir_ICA = base/f'{proj_n}.4-Fitted_ICAs' # ! Remove if using already fitted ica files

for directory in [dir_figs, dir_figs_montage, dir_pickle, dir_ICA]:
    directory.mkdir(exist_ok=True)


# * ############################ DATA INFORMATION #############################

info_file = pd.read_excel(base/"Proj1-info.xlsx", sheet_name=None)

subj_info_df, event_ids, ch_names = [sheet for sheet in info_file.values()]

subj_info_df.set_index('subj', drop=True, inplace=True)

subj_info_df['bad_ch'] = subj_info_df['bad_ch'].replace(
    [np.nan], [None]).str.split(', ')


subj_info_dict = subj_info_df.to_dict(orient='index')

# * Experiment's event IDs (used for the ERP analysis)
event_ids = {event: val for event, val in zip(
    event_ids.event, event_ids.value)}

# * Channels' / electrodes' names
ch_names = ch_names.iloc[:, 0].to_list()


# * ############################# N200 TEMPLATES ##############################

templates_path = base.parent/'Template_Files/N200template.mat'

headmodel_path = base.parent/'Template_Files/eginn128hm.mat'

wave_template, topo_tpl_arr = generate_templates(templates_path=templates_path,
                                                 headmodel_path=headmodel_path,
                                                 save_path=base) 
                                                #! Consider adding a "blank" parameter -> see adjust_topo_template()

template_img_arr = topo_tpl_arr # ! Remove this, rename appropriately before

# * Array of the template topomap in DataFrame format
df_topo_tpl = pd.DataFrame(topo_tpl_arr)


# * ################################ FUNCTIONS ################################

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


def subj_analysis(sess_name, ica_f=None):
    steps = {'Analyzing': f'PREPROCESSING {sess_name} (averaging, filtering, ICA)',
            'Montage': 'SETTING UP HEAD MONTAGE',
            'Averaging': 'AVERAGING',
            'Bipolar Ref': 'SETTING BIPOLAR REFERENCE',
            'Bandpass1': 'BANDPASS FILTER 0.1-100 Hz',
            'Bandpass2': 'BANDPASS FILTER 1-10 Hz',
            'ICA1_1': 'LOADING ICA FILE',
            'ICA1_2': 'FITTING ICA',
            'ICA2': 'APPLYING ICA',
            'EOG_ids': "Rejected EOG components' indices: %s",
            'cleaning_events': 'CLEANING EVENTS',
            'epoching': 'EPOCHING',
            }
    
    subj_info = subj_info_dict[int(sess_name[5:7])]
    
   
    # ! =========================== PREPROCESSING =============================
    print(steps.get('Analyzing', ''))

    # * .bdf file containig EEG data and events * 
    bdf_f = bdf_files[sess_name]
    
    # * .elp file containing head montage information *
    elp_f = elp_files[sess_name[:-2]]
          

    raw = mne.io.read_raw_bdf(bdf_f, preload=True, verbose=False)
    
    samp_freq = raw.info['sfreq']
    
    subj_info = subj_info_dict[int(sess_name[5:7])]
    
    
    if subj_info['bad_ch']:
        raw.info['bads'].extend(subj_info['bad_ch'])


    def get_montage(std_montage = False):
        
        print(steps.get('Montage'))
        
        if std_montage:
            montage = mne.channels.make_standard_montage('biosemi256')
            montage.rename_channels(
                {old: new for old, new in zip(montage.ch_names,
                                            ch_names[:-1])})
        else:
            montage = mne.channels.read_dig_polhemus_isotrak(
                elp_f, ch_names=ch_names + ['unknown'])

        # * Link the head montage to the EEG data *
        raw.set_montage(montage)

        # raw.info['bads'].extend(bad_channels[sess_name[:-2]])

        # * Plot 2D Montage *
        raw.plot_sensors(kind='topomap', ch_type=None, title=sess_name[:-2],
                        show_names=False, ch_groups='position', block=False,
                        show=False)

        f_name = sess_name + '_2D-montage.png'
        plt.savefig(dir_figs_montage/f_name, dpi=300)
        plt.close()

        # * Plot 3D Montage *
        plt.figure(figsize=figsize)
        raw.plot_sensors(kind='3d', ch_type=None, title=sess_name[:-2],
                        show_names=raw.info['bads'], ch_groups='position',
                        block=False, show=False)

        f_name = sess_name + '_3D-montage.png'
        plt.savefig(dir_figs_montage/f_name, figsize=figsize)
        plt.close()

        return montage
    
    # * Set up head montage and save the figures 
    montage = get_montage()
    
    
    # * Detect events in raw EEG data 
    events = mne.find_events(raw, stim_channel='Status', shortest_event=1,
                             initial_event=True, verbose=False)  
   

    # * Average Reference *
    print(steps.get('Averaging'))
    raw.set_eeg_reference(ref_channels='average')

    # * Bipolar Reference for EOG detection* 
    print(steps.get('Bipolar Ref', ''))
    
    anode, cathode = '1EX4', '1EX5'
    
    # * Plot electrodes used for bipolar_ref in 3D *
    raw.plot_sensors(kind='3d', ch_type=None, title=sess_name[:-2],
                     show_names=[anode, cathode], ch_groups='position',
                     block=False, show=False)

    f_name = sess_name + '_Sim-EOG.png'
    plt.savefig(dir_figs_montage/f_name, dpi=300)
    plt.close()

    # * Set up a simulated EOG channel *
    mne.set_bipolar_reference(raw, anode, cathode, ch_name='sim_EOG',
                              ch_info=None, drop_refs=True, copy=False,
                              verbose=False)

    raw.set_channel_types({'sim_EOG': 'eog'}, verbose=False)

    # * Bandpass Filter: 0.1 - 100 Hz *
    print(steps.get('Bandpass1'))
    raw.filter(l_freq=.1, h_freq=100)


    # * EOG artifact rejection using ICA *
    
    if ica_f:
        print(steps.get('ICA1_1'))
        ica = mne.preprocessing.read_ica(ica_f)
        
    else: 
        print(steps.get('ICA1_2'))
        ica = mne.preprocessing.ICA(n_components=0.999999, noise_cov=None,
                                    random_state=97, method='fastica', max_iter='auto')
        ica.fit(raw) 
        
        f_name = sess_name + '_fitted-ica.fif'
        ica.save(dir_ICA/f_name)
    
    eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name='sim_EOG')
    
    print(steps.get('EOG_ids')%eog_inds)
    
    ica.exclude = eog_inds

    print(steps.get('ICA2'))
    raw = ica.apply(raw)
    
    
    #! COMMENTED OUT 21/08/2022
    
    # data[sess_name]['rejections']['ICA_excluded'] = eog_inds

    # data[sess_name]['ICA_applied'] = raw.copy()

    # * barplot of ICA component "EOG match" scores
    # plt.figure()
    # ica.plot_scores(eog_scores, show=False)
    # ica.plot_sources(raw, show_scrollbars=False)
    # ica.plot_components(picks=[i for i in range(10)])
    # plt.show(block=False)
    # plt.close('all')

    # * plot diagnostics
    # ica.plot_properties(raw, picks=eog_inds)

    # * plot ICs applied to raw data, with EOG matches highlighted
    # ica.plot_sources(raw, show_scrollbars=False)

    # * plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    # ica.plot_sources(eog_evoked)
    
    #! COMMENTED OUT 21/08/2022

    # * Bandpass Filter: 1 - 10 Hz *
    print(steps.get('Bandpass2'))

    raw.filter(l_freq=1, h_freq=10)

    # ! =========================== EVENT CLEANING ============================

    print(steps.get('cleaning_events'))

    responses = ['Match Resp.', 'Non-Match Resp.']
    stimuli = ['A', 'B', 'C', 'F', 'H', 'J', 'K']

    correct_feedback = ['Sound Correct', 'Sound Bonus', 'Neutral Correct']
    incorrect_feedback = ['Sound Wrong', 'Neutral Wrong', 'Sound Penalty']
    feedbacks_combined = incorrect_feedback + correct_feedback

    event_ids_reversed = {v: k for k, v in event_ids.items()}

    df_events = pd.DataFrame(events, columns=['sample_nb', 'stim_ch_val', 'event_id'])

    # * convert sample nb into event onset time in seconds *
    event_time = df_events['sample_nb'] / samp_freq


    # df_events.insert(0, 'session', [sess_name] * len(events))
    df_events.insert(3, 'event_time', event_time)

    # * replace event ID number by event name *
    df_events['event_id'].replace(to_replace=event_ids_reversed, inplace=True)
    

    def get_trial_level(df):
        # * Get trial level data
        
        trial_col = pd.Series(
            [np.nan for _ in range(df.shape[0])], dtype=float)

        
        ''' Below:
            Only select stimuli presetend right before participant's responses 
            (button presses; e.g. not the first 2 stimuli presented at start 
            of each trial).
            
            Make sure that only stimuli are selected (e.g. not auditory 
            feedback or double button press)
        ''' 
        
        def cond1(x): return df.iloc[x - 1]['event_id'] in (stimuli)

        def cond2(x): return df.iloc[x]['event_id'] in (responses)

        def cond3(x): return df.iloc[x + 1]['event_id'] in (feedbacks_combined)

        trial_nb = 0
        
        for idx, row in df['event_id'].iteritems():
            
            if cond1(idx) and cond2(idx) and cond3(idx):
                
                trial_nb += 1
                
                trial_col.iloc[idx - 1] = trial_nb
                trial_col.iloc[idx] = trial_nb
                trial_col.iloc[idx + 1] = trial_nb

        df['trial'] = trial_col

        df_trials = df.copy().dropna()
        df_trials.insert(0, 'original_idx', df_trials.index)
        df_trials.reset_index(drop=True, inplace=True)

        return df_trials


    def get_RTs(df_trials):

        RT_col = pd.Series([np.nan for _ in range(df_trials.shape[0])])

        for trial in df_trials['trial'].unique():
            for idx, row in df_trials.query(f'trial == {trial}').iterrows():
                if row['event_id'] in responses:
                    RT = df_trials.iloc[idx]['event_time'] \
                        - df_trials.iloc[idx - 1]['event_time']

                    RT_col.iloc[idx] = RT

        return RT_col


    df_trials = get_trial_level(df_events)
    df_trials['RT'] = get_RTs(df_trials)

    
    # * Remove unwanted events and keep trial level data only *
    ''' 1 trial = stim -> resp -> feedback  
        Unwanted events = Prepare, Rest, etc. '''
    
    # * Select stimulus onset for each trial * 
    events_cleaned = df_trials.query(f'event_id in {stimuli}').copy()

    # * Rename event IDs to their original number-coded format *
    events_cleaned['event_id'].replace(to_replace=event_ids, inplace=True)
    
    # * Convert to MNE events format * 
    events_cleaned = events_cleaned[['sample_nb', 'stim_ch_val',
                                     'event_id']].to_numpy()
    
    
    # ! ============================= EPOCHING ================================
    
    print(steps.get('epoching'))

    # * Remove "EX"s chans, simulated EOG & stim chan from analysis*
    removed = raw.info['bads'].copy() + ['1EX1', '1EX2', '1EX3', '1EX6', '1EX7',
                                        '1EX8', 'sim_EOG', 'Status']
    
    picked_chs = [ch for ch in raw.ch_names if ch not in removed]
    
    epochs = mne.Epochs(raw, events_cleaned, event_id=event_ids,
                        tmin=-0.2, tmax=0.5, baseline=(-0.2, 0),
                        detrend=1, on_missing='warn', preload=True,
                        picks=picked_chs)
        
    
    # * Selecting epochs to reject * 
    df_epochs = epochs.to_data_frame()

    epoch_inds = [idx for idx in df_epochs['epoch'].unique()]
    epochs_info = {}

    for idx in epoch_inds:
        # * Create a DF per epoch *
        df_ep = df_epochs[df_epochs['epoch'] == idx]

        # * Select only columns with channel info *
        df_ep = df_ep.iloc[:, 3:]

        # * Total nb of channels *
        nb_of_channels = df_ep.shape[1]

        # * nb of times each channel has values higher than +-100 µV *
        ch_counts = df_ep[(df_ep <= -100) | (df_ep >= 100)].count()

        # * Select only those that display such values at least once *
        ch_counts = ch_counts[ch_counts > 0].to_dict()

        bad_ch_ratio = len(ch_counts) / nb_of_channels

        epochs_info[idx] = []
        epochs_info[idx].append(ch_counts)
        epochs_info[idx].append(bad_ch_ratio)


    # * Reject trial if ~30% electrodes have some amplitudes over +- 100 μV *
    # * Reject channel if, for over 20 trials, amplitudes +- 100 μV *
    reject_crit = [20, 0.3]   

    # * list of bad channels per epoch
    rej_ch_list = [list(info[0].keys()) for info in epochs_info.values()]

    # * Flatten the list
    rej_ch_list = list(itertools.chain(*rej_ch_list))

    # * Nb of trials where channels display values more than +-100 µV
    rej_ch_list = pd.Series(rej_ch_list).value_counts()

    rej_ch_list = rej_ch_list[rej_ch_list > reject_crit[0]].index.to_list()

    # * Reject entire trial if ~30% electrodes have some amplitudes over +- 100 microV
    rej_epoch_list = [idx for idx, info in epochs_info.items() 
                      if info[1] > reject_crit[1]]

    raw.info['bads'].extend(rej_ch_list)

    
    print(f'{len(raw.info["bads"])} electrodes rejected')
    print(f'{len(rej_epoch_list)} epochs rejected')

    # * Dropping bad epochs and channels from the MNE epochs instance *
    epochs.drop_channels(rej_ch_list)
    epochs.drop(rej_epoch_list)
    
    # * Getting the Event-Related Potential (ERP) 
    evoked = epochs.average()
    
    # * Dropping bad epochs from df_trial *
    inds_to_drop = df_trials.query(
                            f'trial.isin({[e + 1 for e in rej_epoch_list]})'
                            ).index
    
    df_trials.drop(index=inds_to_drop, inplace=True)
    

    # * Accuracy list -> Correct resp. = 1; Incorrect resp. = 0 *
    accuracy = df_trials\
        .query(f'event_id.isin({feedbacks_combined})')['event_id']\
        .replace(incorrect_feedback, 0)\
        .replace(correct_feedback, 1)\
        .to_numpy()
    
    # * Ratio of correct responses *  
    accuracy_ratio = np.count_nonzero(accuracy == 1) / len(accuracy)
    accuracy_ratio = round(accuracy_ratio, 2)
        
    # * Reaction Times (RT)
    RT = df_trials['RT'].dropna().to_numpy()

    subj_data = dict(
        gender = subj_info['gender'],
        age = subj_info['age'],
        hand = subj_info['hand'],
        accuracy = accuracy,
        accuracy_ratio = accuracy_ratio,
        RT = RT,
        chan_pos = montage.get_positions(),
        evoked = evoked,
        events = df_events['event_id'].value_counts().to_dict(),
        rejections = dict(channels = rej_ch_list, 
                          epochs = rej_epoch_list,
                          EOG_comps = eog_inds),
        bad_ch = raw.info['bads'],
        bipolar_ref = dict(anode = anode, 
                           cathode = cathode),
        # trials = dict(total = len(df_events.query(f'event_id.isin([feedbacks_combined])')))
        #  raw_events = events
    )

    
    pickle_export(data=subj_data, path=dir_pickle,
                  f_name=f'{sess_name}-DataDict')
    
    # * N200_detection
    N200_detection(sess_name, evoked, dir_figs, dir_pickle, n200_window,
                   nb_chans, nb_comps, wave_template, df_topo_tpl, 
                   figsize=figsize)
    
    plt.close('all')
    print('\n\n')
    # return data
    
def data_to_df(subj_data, subj):

    temp_dict = subj_data.copy()

    for key in ['chan_pos', 'accuracy_ratio', 'evoked', 'events', 'rejections',
                'bad_ch', 'bipolar_ref']:
        del temp_dict[key]

    to_array = ['gender', 'age', 'hand']

    n = len(temp_dict['RT'])

    subj_df = pd.DataFrame({k: np.array([v for _ in range(n)])
                            if k in to_array else v for k, v in
                            temp_dict.items()})

    subj_df['subj'] = [subj for _ in range(n)]

    return subj_df


def export_results(exp_dir=base):
    data_dicts = {f.name[:9]: dict(*pickle_import(f))
                  for f in dir_pickle.iterdir() if 'N200' not in f.name}

    n200_dicts = {f.name[:9]: dict(*pickle_import(f))
                  for f in dir_pickle.iterdir() if 'N200' in f.name}

    final_df1 = pd.DataFrame()
    final_df2 = pd.DataFrame()
    final_dict = dict()

    combined = list(zip(sess_names, data_dicts.values(),  n200_dicts.values()))

    data_dicts['sess_05-1']['RT']
    for subj, data, n200_data in combined:
        final_dict[subj] = dict()

        rt_quant10 = np.nanquantile(data['RT'], 0.1)
        rt_mean = np.nanmean(data['RT'])
        rt_min = np.nanmin(data['RT'])
        n200_latency = n200_data['u_peak']['latency']
        accuracy_ratio = data['accuracy_ratio']

        final_dict[subj] = {'rt_quant10': rt_quant10,
                            'rt_mean': rt_mean,
                            'rt_min': rt_min,
                            'n200_latency': n200_latency,
                            'accuracy_ratio': accuracy_ratio
                            }

        df1 = pd.DataFrame(final_dict[subj], index=[subj])
        final_df1 = pd.concat([final_df1, df1])

        df2 = pd.DataFrame({'subj': [subj for _ in range(len(data['RT']))],
                            'RT': data['RT'],
                            'accuracy': data['accuracy'],
                            })
        final_df2 = pd.concat([final_df2, df2])

        final_df1.to_excel(base/'Proj1-Final_DF1.xlsx')
        final_df2.to_excel(base/'Proj1-Final_DF2.xlsx')
        pickle_export(final_dict, base, 'Proj1-Final_Dict')

        return final_df1, final_df2, final_dict


def multiprocessing_prepro(nb_cpus=None):

    if nb_cpus == None:
        nb_cpus = ProcessPoolExecutor()._max_workers - 2
    # else:
    #     nb_cpus = ProcessPoolExecutor()._max_workers - nb_cpus

    with ProcessPoolExecutor(nb_cpus) as executor:

        # res = executor.map(args1, args2, args3, args4, args5)
        res = executor.map(prepro, sess_names)

        for result in res:
            print(result)


# * ################################## MAIN ###################################

bdf_files, elp_files = load_data(dir_sessions)
ica_files = {re.search('sess_\d{2}-\d', f.name)[0]: f for f in dir_ICA.iterdir()}


# ! Removing the first 5 participants from analysis (sess 01-1 to 04-2) 
# ! Indices 0 to 8 
sess_names = list(bdf_files.keys())[8:]


for sess in sess_names:
    subj_analysis(sess, ica_f=ica_files[sess])
    



# if __name__ == '__main__':
#     start = time.perf_counter()

#     multiprocessing_prepro()

#     finish = time.perf_counter()

#     print(f'\n\nFinished in {round(finish-start, 2)} second(s)')
