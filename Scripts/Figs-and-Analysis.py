# %%
# ! IMPORTS
# =============================================================================
import pandas as pd
import seaborn as sns
import pathlib
import os
import matplotlib
import matplotlib.pyplot as plt
from CommonFuncs import (pickle_import, pickle_export, modify_folder,
                         reg_analysis, select_files)
import numpy as np
import statsmodels.api as sm

# %% 
# ! DEFINING FUNCTIONS
# =============================================================================
# * GLOBAL PARAMETERS

# plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

base = pathlib.Path('C:/Users/chris/Videos/EEG_data/FINAL')

if base != (wd := pathlib.Path.cwd()):
    print(f'- Current working directory: {wd}')
    print(f'- Changing it to: {base}')
    os.chdir(base)
    
N200_window = [125, 275]


# =============================================================================
# * NDT GRAPHS

def NDT_Latency_plots():
    f_path = base/'JASP_analysis/RegData.csv'

    d1 = pd.read_csv(f_path)

    # NDT HISTOGRAM
    p1 = sns.displot(data=d1, x="NDT", hue="dataset", kde=True, palette='tab10')
    p1.set(xlabel='NDT (ms; $10^{th}$ RT Percentiles)')
    plt.savefig(base/'NDT-hist.png', dpi=300)
    plt.close('all')

    # NDT DISTRIBUTIONS
    p2 = sns.displot(data=d1, x="NDT", hue="dataset", kind='kde', palette='tab10')
    p2.set(xlabel='NDT (ms; $10^{th}$ RT Percentiles)')
    plt.savefig(base/'NDT-dist.png', dpi=300)
    plt.close('all')

    # N200 LATENCY HISTOGRAM
    p3 = sns.displot(data=d1, x="latency", hue="dataset", kde=True, palette='tab10')
    p3.set(xlabel='Latency (ms)')
    plt.savefig(base/'N200_Latency-hist.png', dpi=300)
    plt.close('all')

    # N200 LATENCY DISTRIBUTIONS
    p4 = sns.displot(data=d1, x="latency", hue="dataset", kind='kde', palette='tab10')
    p4.set(xlabel='Latency (ms)')
    plt.savefig(base/'N200_Latency-dist.png', dpi=300)
    plt.close('all')


    # COMBINED HISTOGRAMS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25.6, 12.8), constrained_layout=True)
    sns.histplot(ax=ax1, data=d1, x="NDT", hue="dataset", kde=True, palette='tab10')
    sns.histplot(data=d1, x="latency", hue="dataset", kde=True, palette='tab10')


    ax1.set_title('NDT ($10^{th}$ RT Percentiles)', fontsize=22)
    ax2.set_title('N200 Peak Latency', fontsize=22)
    fig.supxlabel('Time (ms)', fontsize=18)

    ax1.set(xlabel=None)
    ax1.set_ylabel('Count', fontsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)

    ax2.set(xlabel=None)
    ax2.set(ylabel=None)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
        

    plt.setp(ax1.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax1.get_legend().get_title(), fontsize='16') # for legend title

    plt.setp(ax2.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax2.get_legend().get_title(), fontsize='16') # for legend title

    plt.savefig(base/'Latency-NDT-Combined.png')
    plt.close('all')


# =============================================================================
# * N200 GRAPHS

def N200_plots():
    # path1 = pathlib.Path(r'C:\Users\chris\Videos\EEG_data\proj1\OLD\pickle_files')
    path1 = pathlib.Path(r'C:\Users\chris\Videos\EEG_data\proj1\2-Pickle_Files')
    path2 = pathlib.Path(r'C:\Users\chris\Videos\EEG_data\proj2\test_pickle')
    path3 = pathlib.Path(r'C:\Users\chris\Videos\EEG_data\proj3\Pickle_files')

    dir1 = [f for f in path1.iterdir() if 'N200' in f.name]
    dir2 = [f for f in path2.iterdir() if 'N200' in f.name]
    dir3 = [f for f in path3.iterdir() if 'N200' in f.name]


    N200_dict1 = dict()
    N200_dict2 = dict()
    N200_dict3 = dict()

    N200_dicts_list = [N200_dict1, N200_dict2, N200_dict3]

    N200_data_mapped = list(zip([dir1, dir2, dir3], N200_dicts_list))

    for directory, N200_dict in N200_data_mapped:
        for f in directory:
            subj = f.name[:f.name.find('-N200')]
            print(subj)
            N200_dict[subj] = dict(*pickle_import(f))

    # N200_dict_vals = [[idx, N200_dict.values()] for idx, N200_dict in enumerate(N200_dicts_list)]

    N200_dict_keys = {}
    for idx, N200_dict in enumerate(N200_dicts_list):
        N200_dict_keys[idx + 1] = list(N200_dict[list(N200_dict.keys())[0]].keys())

    N200_dict_keys = pd.DataFrame.from_dict(N200_dict_keys, orient='index').T
    N200_dict_keys


    sfreqs = [256, 1000, 500]
    n_points = [int((sfreq / 1000) * np.diff(N200_window)[0]+1) for sfreq in sfreqs]
    # timepoints = [np.linspace(*N200_window, n) for n in n_points]

    timepoints = [
                    np.linspace(125, 275, round((256/1000) * 151)),
                    np.linspace(125, 275, round((1000/1000) * 151)),
                    np.linspace(115, 275, round((500/1000) * 161)),
                ]

    combined_v_comps = dict()
    combined_u_comps = dict()
    combined_picked_comps = dict()

    # combined_waveforms = None
    average_waveforms = dict()


    for idx_dict, N200_dict in enumerate(N200_dicts_list):

        dataset = f'N200_dict{idx_dict + 1}'
        
        combined_v_comps[dataset] = dict()
        combined_u_comps[dataset] = dict()
        combined_picked_comps[dataset] = dict()
        
        waveforms = pd.DataFrame()
        
        for idx_subj, subj in enumerate(N200_dict.keys()):
            # print(N200_dict[subj])
            picked_comp = N200_dict[subj]['picked_comp']
            combined_picked_comps[dataset][subj] = picked_comp
            
            v_comps = N200_dict[subj]['v_comps']
            combined_v_comps[dataset][subj] = v_comps
            
            u_comps = N200_dict[subj]['u_comps']
            
            combined_u_comps[dataset][subj] = u_comps
            
            N200_waveform = pd.DataFrame(u_comps[picked_comp], 
                                        columns=[idx_subj + 1], 
                                        index=timepoints[idx_dict])

            waveforms = pd.concat([waveforms, N200_waveform], axis=1)

        
        average_waveform = waveforms.mean(axis=1)
        
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25.6, 12.8), sharex=True,
        sharey=True, constrained_layout=True)
        
        fig.suptitle(f'Dataset {idx_dict + 1}', fontsize=28)
        ax1.set_title('N200 Waveforms', fontsize=22)
        ax2.set_title('Average N200 Waveform', fontsize=22)
        ax1.set_ylabel('Amplitude (μV)', fontsize=18)
        fig.supxlabel('Time (ms)', fontsize=18)
        
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        
        ax2.tick_params(axis='x', labelsize=16)
        ax2.tick_params(axis='y', labelsize=16)
            
        for col in waveforms.columns:
            sns.lineplot(ax=ax1, data=waveforms[col])
            
        sns.lineplot(ax=ax2, data=average_waveform)
        plt.savefig(base/f'average_waveform-{dataset}.png', dpi=300)
        
        plt.close('all')
        
        average_waveforms[dataset] = average_waveform
                
                
# =============================================================================
# * REG DATA TABLE

def reg_data_table():
    summary_stats = dict()
    summary_stats['combined'] = d1[['latency','NDT']].describe()

    for i in [1,2,3]:
        table = d1.query(f'dataset == {i}')[['latency','NDT']].describe()
        summary_stats[f'D{i}'] = table

    with pd.ExcelWriter(base/"summary_stats.xlsx") as writer:
        for name, table in summary_stats.items():
            table.to_excel(writer, sheet_name=name)

    # summary_stats['D1']


# =============================================================================
# * LINEAR REGRESSION PLOTS

# reg_data = d1.copy()
def one_plot_all():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=2, figsize=(25.6, 12.8), 
                                            constrained_layout=True)
                                            

    ax1[0].set_title('Dataset 1', fontsize=22)
    ax1[1].set_title('Dataset 2', fontsize=22)

    ax2[0].set_title('Dataset 3', fontsize=22)
    ax2[1].set_title('Dataset 2 & 3', fontsize=22)

    ax3[0].set_title('All Combined', fontsize=22)

    # Dataset 1
    query = reg_data.query(f'dataset == {1}')
    sns.regplot(ax=ax1[0], x='latency', y='NDT', data=query)

    # Dataset 2
    query = reg_data.query(f'dataset == {2}')
    sns.regplot(ax=ax1[1], x='latency', y='NDT', data=query)

    # Dataset 2 & 3 
    query = reg_data.query(f'dataset == {2} | dataset == {3}')
    sns.regplot(ax=ax2[0], x='latency', y='NDT', data=query)

    # Dataset 3
    query = reg_data.query(f'dataset == {3}')
    sns.regplot(ax=ax2[1], x='latency', y='NDT', data=query)

    # All datasets combined
    sns.regplot(ax=ax3[0], x='latency', y='NDT', data=reg_data)


    # ax1.set_ylabel('amplitude (μV)', fontsize=18)
    fig.supxlabel('Time (ms)', fontsize=18)


    plt.savefig(base/'combined.png')
    plt.close('all')


def separate_combined_plots(size1=(15.6, 25.6), size2=(15.6, 20.6)):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=size1, 
                                            constrained_layout=True)
                                            

    # Dataset 1
    query = reg_data.query(f'dataset == {1}')
    reg1 = sns.regplot(ax=ax1, x='latency', y='NDT', data=query)

    # Dataset 2
    query = reg_data.query(f'dataset == {2}')
    reg2 = sns.regplot(ax=ax2, x='latency', y='NDT', data=query)


    # Dataset 3
    query = reg_data.query(f'dataset == {3}')
    reg3 = sns.regplot(ax=ax3, x='latency', y='NDT', data=query)
    
    # reg1.set(xlim=(100, 280))
    # reg2.set(xlim=(100, 280))
    # reg3.set(xlim=(100, 280))
    
    ax1.set_title('Dataset 1', fontsize=22)
    ax2.set_title('Dataset 2', fontsize=22)
    ax3.set_title('Dataset 3', fontsize=22)

    ax1.set(xlabel=None)
    ax2.set(xlabel=None)
    ax3.set(xlabel=None)
    
    ax1.set(ylabel=None)
    ax2.set(ylabel=None)
    ax3.set(ylabel=None)
        
    ax1.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='x', labelsize=14)
    
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    
    fig.supylabel('NDT (ms; $10^{th}$ RT Percentiles)', fontsize=20)
    fig.supxlabel('Latency (ms)', fontsize=20)
    
    plt.savefig(base/'reg_plots-DS-1_2_3.png')
    plt.close('all')


    #######################################################

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=size2, 
                                            constrained_layout=True)
                                            
    # Dataset 2 & 3 
    query = reg_data.query(f'dataset == {2} | dataset == {3}')
    sns.regplot(ax=ax1, x='latency', y='NDT', data=query)
    
    # All datasets combined
    sns.regplot(ax=ax2, x='latency', y='NDT', data=reg_data)


    ax1.set(xlabel=None)
    ax2.set(xlabel=None)

    
    ax1.set(ylabel=None)
    ax2.set(ylabel=None)
        
    ax1.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    

    ax1.set_title('Dataset 2 & 3', fontsize=22)
    ax2.set_title('All combined', fontsize=22)

    fig.supylabel('NDT (ms; $10^{th}$ RT Percentiles)', fontsize=20)
    fig.supxlabel('Latency (ms)', fontsize=20)
    

    plt.savefig(base/'reg_plots-combined.png')
    plt.close('all')


# =============================================================================
# * LINEAR REGRESSION ANALYSIS

def lin_reg_analysis():
    x = d1['latency'].to_numpy()
    y = d1['NDT'].to_numpy()
    # x = df_reg_data['latency'].to_numpy()
    # y = df_reg_data['resp. time (10th %ile)'].to_numpy()


    reg_data1 = reg_data.query('dataset == 1')
    reg_data2 = reg_data.query('dataset == 2')
    reg_data3 = reg_data.query('dataset == 3')
    reg_data4 = reg_data.query('dataset == 2 | dataset == 3')
    reg_data5 = reg_data

    concat_XY = dict()
    for idx, data in enumerate([reg_data1, reg_data2, reg_data3, reg_data4, reg_data5]):

        X = sm.add_constant(data['latency']) # adding a constant
        Y = data['NDT']
        
        concat_XY[f'reg{idx + 1}'] = (X, Y)

        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X)

        model_summary = model.summary()
        # dir(model_summary)
        # model_summary.as_text()

        with open(base/f'LinReg-Summary_{idx + 1}.txt', 'w') as f:
            f.write(model_summary.as_text())

        print(model_summary)


# %%
# ! RUNNING THE SCRIPT 
# =============================================================================

NDT_Latency_plots()
N200_plots()
separate_combined_plots()
lin_reg_analysis()
