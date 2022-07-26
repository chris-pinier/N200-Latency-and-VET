# ORDER OF IMPORTS -> mpl > mpl.use('Qt5Agg'), pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')  # specificies use of an interactive backend
import matplotlib.pyplot as plt
import itertools
import mne
import numpy as np
import os
import pickle
import pandas as pd
import pathlib
from pathlib import Path
from PIL import Image
from pprint import pprint
from pymatreader import read_mat
import re
import scipy.io
import shutil
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore',category=DeprecationWarning)

default_figsize = (12.8, 12.8)

def pickle_import(f_name):
    with open(f_name, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
    print('Pickle import done.')


def pickle_export(data, path, f_name="data"):
    # import sys
    # print(sys.getsizeof(raw_data))
    f_name += '.pickle'
    file = path/f_name

    with open(file, 'wb') as f:
        pickle.dump(data, f)

    print('Pickle export done.')


def show_all(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df)


def make_montage(type, path):

    def std_montage():
        return mne.channels.make_standard_montage('biosemi128')

    def custom_montage():
        chanlist = np.arange(0, 128)

        headmodel = 'C:/Users/chris/Videos/EEG_data/test/eginn128hm.mat'
        locdic = scipy.io.loadmat(headmodel)['EGINN128']

        nasion_chan = 17 - 1
        lpa_chan = 48 - 1
        rpa_chan = 113 - 1
        locdicchan = locdic[0][0][0][0][0][4][0:-1]
        chan_pos = dict()

        for i, j in enumerate(locdicchan[chanlist]):
            chan_pos[str(chanlist[i] + 1)] = j * 0.1

        return mne.channels\
            .make_dig_montage(
                                        ch_pos=chan_pos,
                                        nasion=locdicchan[nasion_chan] * 0.1,
                                        lpa=locdicchan[lpa_chan] * 0.1,
                                        rpa=locdicchan[rpa_chan] * 0.1
                                        )

    err_str = {'wrong_montage_name': (
                                        'Choose a montage between: standard'
                                        '("std") and custom ("custom")'
                                        )}

    return dict(
        std=std_montage,
        custom=custom_montage
        ).get(type, err_str['wrong_montage_name'])()


def modify_folder(directory, action, f_type=None, contains=None, pattern=None):
    if f_type:
        files = [f for f in os.listdir(directory) if f.endswith('.' + f_type)]
    else:
        files = [f for f in directory.iterdir() if f.is_file()]

    if contains:
        search = [os.path.join(directory, f)
                  for f in files if re.search(contains, f)]

    if action == 'remove':
        for f in files:
            os.remove(directory/f)

    # create nested folders
    elif action == 'pack':
        unique = sorted(set([re.search(pattern, f.name)[0] for f in files]))

        for name in unique:
            # REPLACE str(f) by f.name
            old_paths = [f for f in files if name in f.name]
            for f in old_paths:
                files.remove(f)

            new_dir = directory/name
            new_dir.mkdir()

            new_paths = [new_dir/f.name for f in old_paths]

            for path in zip(old_paths, new_paths):
                path[0].replace(path[1])

    # unpack nested folders
    elif action == 'unpack':
        subfolders = [fold for fold in directory.iterdir()]
        old_paths = [list(fold.iterdir()) for fold in subfolders]
        old_paths = list(itertools.chain.from_iterable(old_paths))

        new_paths = [directory/f.name for f in old_paths]

        for path in zip(old_paths, new_paths):
            path[0].replace(path[1])

        for sub in subfolders:
            sub.rmdir()

    # search file type
    elif action == 'search':
        # pprint(search)
        return search


def generate_templates(template_path, headmodel_path, save_path, 
                        figsize=default_figsize, template_rad=600):
    '''
    - template_path: type = str | pathlib.WindowsPath;
    - headmodel_path: type = str | pathlib.WindowsPath;
    - save_path: type = str | pathlib.WindowsPath; 
    - figsize: type = tuple; 
    - template_rad: type = int; radius used for the head model of the 
                    topographic maps
    
    return: 
        - wave_template: pandas.Series;
        - template_img_arr: numpy.ndarray
    '''
    
    if type(template_path) != pathlib.WindowsPath:
        template_path = pathlib.Path(template_path)
        
    templates = read_mat(template_path)

    # WAVE TEMPLATE
    wave_template = templates['N200wavetemplate']  # Samp. rate = 1000 Hz
    stim_onset = 100  # milliseconds

    # ## Add time points as index  /!\ -> start at 0.000 or 0.001?
    wave_template = pd.Series(wave_template.flatten(),
                              index=np.arange(0, wave_template.shape[0], 1))

    # ## Lock on stimulus onset and reset the index
    wave_template = wave_template.loc[stim_onset:]
    wave_template.index -= stim_onset  # index in ms -> 0 - 999 ms

    # Plotting the wave template
    plt.figure()
    plt.plot(wave_template)
    plt.savefig(save_path/'template_wave.png', dpi = 300)
    # plt.show(block=False)

    # TOPOMAP TEMPLATE
    # [:-1] -> omit the stimulus channel
    topo_template = templates['N200template'][:-1]

    # Set up the montage
    chanlist = [str(i) for i in range(1, 129)]
    
    locdic = read_mat(headmodel_path)['EGINN128']

    locdicchan = locdic['Electrode']['CoordOnSphere'][:-1] * 0.1

    chan_pos = dict(zip(chanlist, locdicchan))

    nasion = chan_pos['17']
    lpa = chan_pos['48']
    rpa = chan_pos['113']

    montage = mne.channels.make_dig_montage(ch_pos=chan_pos,
                                            nasion=nasion,
                                            lpa=lpa,
                                            rpa=rpa)

    montage.plot(show_names=False, show=False)  # (x, y, z, radius)
    plt.savefig(save_path/'template_montage.png', dpi=300)

    template_info = mne.create_info(ch_names=montage.ch_names, sfreq=1000,
                                    ch_types='eeg')

    rng = np.random.RandomState(0)
    fake_data = rng.normal(size=(len(chanlist), 1)) * 1e-6

    fake_evoked = mne.EvokedArray(fake_data, template_info)
    fake_evoked.set_montage(montage)

    template_sphere = mne.make_sphere_model(
        r0='auto', head_radius='auto', info=fake_evoked.info)
    r = template_sphere.radius
    # pprint(dict(template_sphere.items()))

    # Plot the topographic template

    fig, ax = plt.subplots(figsize=figsize)

    # img = mne.viz.plot_topomap(topo_template, fake_evoked.info,
    #                     sensors=False, extrapolate='auto', outlines='head',
    #                     show=False, axes=ax, res=1280, sphere=r,
    #                     contours=6) #cmap='summer',
    # plt.savefig(save_path/f'original_topo_template.png')

    img = mne.viz.plot_topomap(topo_template, fake_evoked.info,
                               sensors=False, extrapolate='auto', outlines=None,
                               show=False, axes=ax, res=1280, sphere=r,
                               cmap='summer', contours=0)

    draw_circle = matplotlib.patches.Circle((0, 0), r, fill=False, lw=0,
                                            transform=ax.transData)

    ax.add_patch(draw_circle).set_alpha(0.3)
    img[0].set_clip_path(draw_circle)
    plt.savefig(save_path/'template_sphere.png', transparent=False)
    # plt.show(block=False)
    
    template_img = Image.open(str(save_path/'template_sphere.png'))
    template_img_arr = np.asarray(template_img)

    # template_center = (656, 601)
    # template_rad = 496

    template_img_arr = ajdust_topo_plot(template_img_arr, transp=False,
                                        template_rad=template_rad,
                                        save_path=str(
                                            save_path/'template_sphere.png'))

    template_img = Image.open(str(save_path/'template_sphere.png')).convert('L')

    # Selecting the lower half of the image (posterior region of the brain)
    template_img_arr = np.asarray(template_img)

    # Replace white pixels by NaN values and vectorize the matrix
    template_img_arr = np.where(
        template_img_arr == 255, np.nan, template_img_arr).flatten()

    # Checking pixel values in grayscale
    # val_counts = np.unique(template_img_arr, return_counts=True)
    # pprint( {px_val:count for px_val, count in zip(val_counts[0], val_counts[1])} )

    # rename var template_img_arr to topo_img_arr /!\

    return wave_template, template_img_arr


def N200_detection(sess, evoked, dir_figs, n200_window, 
                    nb_chans, nb_comps, wave_template, df_topo_tpl, 
                    figsize=default_figsize, dir_pickle=None,
                    condition=None, show_topo_target=None):
    '''
    - sess: type = str; name of the subj and/or session (used for naming the
            files and plots)
    - evoked: type = mne.evoked.EvokedArray; ERP data in MNE format 
                (channels x timepoints ? -->  NEED TO CHECK)
    - dir_figs: type = pathlib.WindowsPath; directory used to save the figures, 
                must be created beforehand
    - dir_pickle: type = pathlib.WindowsPath; directory used to save the 
                    .pickle files containing information about the N200 signal
                    in a dictionnary
    - n200_window: type = list; 2 elements -> start and end time (ms) of the 
                    desired time window to identify the N200; both indices 
                    included
    - wave_template: type = pandas.core.series.Series; time-series values of 
                    the template N200 potential / signal
    - df_topo_tpl: type = pandas.core.frame.DataFrame;
    - figsize: type = tuple; default = (12.8, 12.8); figure size used for 
                certain matplolib and plots    
    
    returns: 
    '''
    
    # need to implement 'condition' and 'show_topo_target' args
    print('     processing data')


    evoked.plot(gfp=True, spatial_colors=True, show=False)
    plt.savefig(dir_figs/f'{sess}_evoked.png', dpi=300)


    # evoked_arr = evoked.get_data()
    evoked_df = evoked.to_data_frame(
                        picks=None, index='time', scalings=None, copy=True,
                        long_format=False, time_format='ms'
                        )

    # Select data within n200_window
    # .reset_index(drop=True)
    evoked_df_N200 = evoked_df.loc[slice(*n200_window)]

    evoked_arr = evoked_df_N200.to_numpy()

    u, s, v = np.linalg.svd(evoked_arr, full_matrices=0)

    # df_u = pd.DataFrame(u.T, columns=evoked_df_N200.index, index=evoked.ch_names)
    # df_u.columns.name = 'time'
    # df_u.index.name = 'components'
    #

    df_v = pd.DataFrame(v, columns=evoked.ch_names)
    df_v.columns.name = 'channels'
    df_v.index.name = 'components'


    # Resample and scale the wave template to match the data more closely
    # ## length = 150 ms -> 125-274 ms (or according to n200_window)

    # Reframe the wave template on the n200_window
    adjusted_wave_tpl = wave_template.loc[slice(*n200_window)]
    # adjusted_wave_tpl = wave_template.iloc[90:300]

    # Resampling the wave template
    resamp_factor = adjusted_wave_tpl.shape[0] / evoked_df_N200.shape[0]

    adjusted_wave_tpl = mne.filter\
                                    .resample(adjusted_wave_tpl.T, up=1,
                                              down=resamp_factor, npad=100,
                                              axis=- 1, window='boxcar',
                                              n_jobs=1, pad='reflect_limited',
                                              verbose=None)\
                                    .flatten()

    # Rescaling the wave template
    # adjusted_wave_tpl *= 10 ** -3
    # magnitude_factor = u[start:end + 1, :nb_comps].min() / adjusted_wave_tpl.min()
    # adjusted_wave_tpl *= magnitude_factor

    df_wave_tpl = pd.DataFrame(adjusted_wave_tpl)

    # data[sess]['adjusted_wave_tpl'] = adjusted_wave_tpl

    u_comps = u.T[:nb_comps, :]
    u_comps = np.concatenate((u_comps, u_comps * - 1), axis=0)

    v_comps = v[:nb_comps, :]
    v_comps = np.concatenate((v_comps, v_comps * - 1), axis=0)

    components_str = [f'comp_{i}' if i <= nb_comps else f'comp_{i - nb_comps}-inv'
                      for i in range(1, nb_comps * 2 + 1)]

    mapped_comps = dict(enumerate(components_str))

    wave_corr = []

    for i in range(u_comps.shape[0]):
        df_wave = pd.DataFrame(u_comps[i, :])

        corr = df_wave.corrwith(df_wave_tpl, drop=False, method='pearson')[0]
        corr = round(corr, 3)

        # corr2 = df_topo.corrwith(df_topo_template, axis=1, drop=False, method='pearson').mean()
        # corr1, corr2 = [round(x, 3) for x in [corr1, corr2]]

        wave_corr.append(corr)

    ################################################################################

    sphere_model = mne.make_sphere_model(r0='auto', head_radius='auto',
                                         info=evoked.info, verbose=False)

    r = sphere_model.radius

    # Extracting the arrays of the topographic map images
    topo_arrs = []

    for idx, comp in enumerate(v_comps):

        name = f'{sess}_topo-{mapped_comps[idx]}.png'

        # Plot the topographic map associated with the component

        fig, ax = plt.subplots(figsize=figsize)

        # Make sure to keep outlines=None & contours=0 for adjust_topo_plot to work`
        topo_img = mne.viz.plot_topomap(comp, evoked.info, sensors=False,
                                        extrapolate='auto', outlines=None, show=False,
                                        axes=ax, res=1280, contours=0, sphere=r, cmap='summer')

        ''' Draw a circle around the data with a radius = radius of the
            subject's head and crop the image around that circle '''
        draw_circle = matplotlib.patches.Circle(
            (0, 0), r, fill=False, lw=0, transform=ax.transData)
        ax.add_patch(draw_circle)
        topo_img[0].set_clip_path(draw_circle)

        # grab the pixel buffer and dump it into a numpy array
        fig.canvas.draw()
        img_arr = np.array(fig.canvas.renderer.buffer_rgba()).astype(np.uint8)

        ''' Adjust the topographic map images so that they are centered and scaled
            on the template then save the figures and store the corresponding arrays
            in topo_arrs'''
        # new_img_arr = ajdust_topo_plot(img_arr, save_path=str(dir_figs/name))
        new_img_arr = ajdust_topo_plot(img_arr)

        topo_arrs.append(new_img_arr)

        fig, ax = plt.subplots(figsize=figsize)

        # Make sure to keep outlines=None & contours=0 for adjust_topo_plot to work`
        topo_img = mne.viz.plot_topomap(comp, evoked.info, sensors=False,
                                        extrapolate='auto', outlines='head', show=False,
                                        axes=ax, res=1280, contours=6, sphere=r)

        plt.savefig(dir_figs/f'{name[:-4]}.png')

    # Convert the images into grayscale and extract the new arrays
    topo_arrs_gray = [np.asarray(Image.fromarray(arr).convert('L'))
                      for arr in topo_arrs]

    topo_corr = []
    for arr in topo_arrs_gray:

        # Replace white pixels by NaN values and vectorize the matrix
        arr = np.where(arr == 255, np.nan, arr).flatten()

        # convert the array into a DataFrame
        df_topo = pd.DataFrame(arr)

        # print('checking that length df_topo = df_topo_tpl ->',
        #         len(df_topo) == len(df_topo_tpl))

        # Compute Pearson correlation btw component's topomap & template topomap
        corr = df_topo.corrwith(df_topo_tpl, drop=False, method='pearson')[0]
        corr = round(corr, 3)

        topo_corr.append(corr)

    df_corr = pd.DataFrame({
                            'component': list(mapped_comps.keys()),
                            'topo_corr': topo_corr,
                            'wave_corr': wave_corr,
                            })

    df_corr.index = [sess for _ in range(len(df_corr))]

    df_corr['prod'] = df_corr['topo_corr'] * df_corr['wave_corr']

    picked_comp = df_corr.sort_values(by='prod',
                                        ascending=False)['component'][0]

    df_u_comps = pd.DataFrame(u_comps,
                                index = [i for i in range(u_comps.shape[0])],
                                columns = evoked_df_N200.index)

    u_comp_latency = df_u_comps.loc[picked_comp].sort_values(ascending=True).index[0]

    # plt.figure()
    # plt.plot(evoked_df_N200.index, u_comps[picked_comp])
    # plt.plot(ch_data * (u_comps.min() / ch_data.min().min() ))

    if picked_comp >= 10:
        picked_chs = df_v.iloc[picked_comp - 10]\
                                    .sort_values(ascending=True)\
                                    .index[:nb_chans].to_list()

    else:
        picked_chs = df_v.iloc[picked_comp]\
                                    .sort_values(ascending=False)\
                                    .index[:nb_chans].to_list()

    ch_data = evoked_df_N200[picked_chs]

    # select the {nb_chans} that contribute the most to the component
    # picked_chs = []

    # picked_chs.extend(channels)

    # ch_data = evoked.to_data_frame(picks=picked_chs, time_format='ms')
    # ch_data = ch_data.set_index('time')

    # peak_ch = picked_chs[0]

    peak_ch = ch_data.min(axis=0).sort_values(ascending=True).index[0]

    peak = ch_data[peak_ch].sort_values(ascending=True)
    peak_latency = peak.index[0]
    peak_val = peak.iloc[0]


    n200_data = {
                    'v_comps': v_comps,
                    'u_comps': u_comps,
                    'topo_arrs': topo_arrs,
                    'df_corr': df_corr,
                    'picked_comp': picked_comp,
                    'picked_chs': picked_chs,
                    'peak': {
                            'ch': peak_ch,
                            'latency': peak_latency,
                            'peak_val': peak_val},
                    'u_peak': {'latency': u_comp_latency}
                    }

    check_ch = peak_ch == evoked_df_N200.min(axis=0).sort_values().index[0]
    # WRONG, NEED TO REWRITE THIS: PEAK LATENCY SHOULD BE LATENCY OF peak_ch
    check_lat = peak_latency == evoked_df_N200.min(axis=1).sort_values().index[0]
    check_val = peak_val == evoked_df_N200.min().min()

    print('Checking if selected channel has lowest value and late on N200 interval')
    print(f'{check_ch = } | {check_lat = } | {check_val = }')


    # 1st set of plots
    mid_comp = int(np.ceil(nb_comps / 2))

    timepoints = np.linspace(*n200_window, adjusted_wave_tpl.shape[0])

    mpl_data = [
                [timepoints, u_comps.T[:, :mid_comp]],
                [timepoints, u_comps.T[:, mid_comp:mid_comp * 2]],
                [timepoints, u_comps.T[:, picked_comp]],
                ]

    y_lims = [
                [u_comps.min(), u_comps.max()],
                [u_comps.min(), u_comps.max()],
                [None]
                ]

    mpl_kwargs = [
                    {'label': [i + 1 for i in range(0, mid_comp)]},
                    {'label': [i + 1 for i in range(mid_comp, mid_comp * 2)]},
                    {'label': '_template'}
                ]

    mpl_titles = [
                f'{sess} - SVD Components (U): 0-{mid_comp}',
                f'{sess} - SVD Components (U): {mid_comp + 1}-{nb_comps}',
                f'Selected SVD Component: {mapped_comps[picked_comp]}',
                # f'Selected SVD Component(s) # {", ".join([str(i) for i in picked_comps])}',
                # f'Electrodes from SVD component # {picked_comp}',
                # f'Electrodes from SVD component(s) # {", ".join([str(i) for i in picked_comps]) }',
                ]

    mpl_fnames = [
                f'{sess}_SVD1.png',
                f'{sess}_SVD2.png',
                f'{sess}_SVD3-selected_comp.png',
                # f'{sess}_2-n200-channels.png',
                ]

    fig, ax = plt.subplots(figsize=figsize)
    for fig_data, y_lim, kwargs, title, fname in zip(mpl_data, y_lims, mpl_kwargs, mpl_titles,
                                                     mpl_fnames):
        ax.plot(*fig_data, **kwargs)
        ax.set_ylim(*y_lim)
        ax.set_title(title)
        # plt.show(block=False)
        ax.legend()
        plt.savefig(dir_figs/fname)
        ax.clear()



    # 2n set of plots
    y_lim = [evoked_df.min().min(), evoked_df.max().max()]
    topo_times = [0.125, 0.15, 0.175, 0.2, 0.225, 0.25]

    mne_plot_args = [
            dict(show=False, times=topo_times, title=sess,
                 ts_args={'spatial_colors': True, 'ylim': dict(eeg=y_lim)}),

            dict(show=False, picks=picked_chs, times=topo_times, title=sess,
                 ts_args={'spatial_colors': True, 'ylim': dict(eeg=y_lim)}),

            dict(show=False, times=topo_times,
                 picks=list((ch_data.min(axis=0).sort_values().index[:2])),
                 title=sess, ts_args={'spatial_colors': True,
                                      'ylim': dict(eeg=y_lim)})
                ]
    # 'ylim': dict(eeg=[-15, 10])

    mne_fnames = [
                    f'{sess}_channels1-all.png',
                    f'{sess}_channels2-selected_chans.png',
                    f'{sess}_channels3-N200_chans.png',
                    ]

    evoked.info['bads'].extend([peak_ch])
    evoked.plot(show=False, spatial_colors=False, ylim=dict(eeg=y_lim),
                picks=evoked.ch_names, exclude=[])
    plt.savefig(dir_figs/f'{sess}_channels4-N200_chan.png', dpi=300)
    evoked.info['bads'].remove(peak_ch)

    for kwargs, fname in zip(mne_plot_args, mne_fnames):
        evoked.plot_joint(**kwargs)
        # plt.show(block=False)
        plt.savefig(dir_figs/fname, dpi=300)
        plt.close()

    fig, ax = plt.subplots(figsize=figsize)

    topo_img = mne.viz.plot_topomap(v_comps[picked_comp, :], evoked.info, sensors=False,
                                    extrapolate='auto', outlines='head', show=False,
                                    axes=ax, res=1280, contours=6, sphere=r)

    plt.savefig(
        dir_figs/f'{sess}_topo-Selected-{mapped_comps[picked_comp]}.png', dpi=300)

    plt.close('all')
    
    if dir_pickle:
        pickle_export(n200_data, dir_pickle, f'{sess}-N200-data_dict')

    return n200_data


def ajdust_topo_plot(img_arr, transp=False, blank=[255, 255, 255, 255], template_rad=600, save_path=None, verbose=False):

    # Create a (pixels x pixels) mask where all white pixels = True
    mask = np.all(img_arr == blank, axis=-1)

    # y coordinate for the center of the circle:
    # ## Count the number of blank pixels in each row
    val_count_rows = np.count_nonzero(mask, axis=1)

    # ## Get the indices of the rows with the mininmum number of blank pixels
    center_rows = [idx for idx, row in enumerate(
        val_count_rows) if row == val_count_rows.min()]

    # ## Select the 'middle' row from these indices
    center_row = int(
        np.ceil(center_rows[0] + (center_rows[-1] - center_rows[0]) / 2))

    # x coordinate for the center of the circle:
    # ## Count the number of blank pixels in each column
    val_count_cols = np.count_nonzero(mask, axis=0)

    # ## Get the indices of the columns with the mininmum number of blank pixels
    center_cols = [idx for idx, col in enumerate(
        val_count_cols) if col == val_count_cols.min()]

    # ## Select the 'middle' column from these indices
    center_col = int(
        np.ceil(center_cols[0] + (center_cols[-1] - center_cols[0]) / 2))

    def checkConsecutive(l):
        return sorted(l) == list(range(min(l), max(l) + 1))

    if not checkConsecutive(center_cols):
        print('some "center columns" indices are not consecutive')

    if not checkConsecutive(center_rows):
        print('some "center rows" indices are not consecutive')

    center = (center_col, center_row)

    rad = np.where(mask[center_row:center_row + 1, center_col:] == True)[1][0]

    recenter = {i: j for i, j in zip(['col', 'row'],
                                     [640 - coord for coord in center])}

    blank_col = np.tile(blank, (1280, abs(recenter['col']), 1))

    if recenter['col'] < 0:
        new_img_arr = np.append(img_arr, blank_col, axis=1)
        new_img_arr = np.delete(
            new_img_arr, np.s_[:abs(recenter['col'])], axis=1)
    else:
        new_img_arr = np.append(blank_col, img_arr, axis=1)
        new_img_arr = np.delete(new_img_arr, np.s_[1280:], axis=1)

    blank_row = np.tile(blank, (abs(recenter['row']), 1280, 1))

    if recenter['row'] < 0:
        new_img_arr = np.append(new_img_arr, blank_row, axis=0)
        new_img_arr = np.delete(
            new_img_arr, np.s_[:abs(recenter['row'])], axis=0)
    else:
        new_img_arr = np.append(blank_row, new_img_arr, axis=0)
        new_img_arr = np.delete(new_img_arr, np.s_[1280:], axis=0)

    # SCALING ####

    new_img = Image.fromarray(new_img_arr.astype(np.uint8), mode='RGBA')

    scaling_ratio = template_rad / rad
    new_size = tuple(round(dim * scaling_ratio) for dim in new_img.size)

    new_img_re1 = new_img.resize(new_size, Image.ANTIALIAS)

    def cropping(img, bg_size=(1280, 1280), ratio=scaling_ratio):
        if scaling_ratio > 1:
            '''crop the image '''
            anchor = tuple(
                map(lambda x: int((x[0] - x[1]) / 2), zip(img.size, bg_size)))

            #                              (left, top, right, bottom)
            new_img = img.crop((anchor[0], anchor[0], anchor[0] + bg_size[0],
                                anchor[0] + bg_size[1]))

        elif scaling_ratio <= 1:
            '''return a white-background-color image having the img in exact center'''
            anchor = tuple(
                map(lambda x: int((x[0] - x[1]) / 2), zip(bg_size, img.size)))
            new_img = Image.new('RGBA', bg_size, color=(255, 255, 255, 255))
            new_img.paste(img, anchor)

        return new_img

    new_img_re2 = cropping(new_img_re1)

    if save_path != None:
        new_img_re2.save(save_path)

    plt.close('all')

    recenter = tuple(recenter.values())
    new_center = tuple(center[i] + recenter[i] for i in range(2))

    if verbose:
        print(f'Original topomap:\n center: {center}, radius: {rad}')
        print('New topomap: \n'
              f' center: {new_center}, '
              f'radius: {round(rad * scaling_ratio, 2)}\n')

    return np.asarray(new_img_re2).astype(np.uint8)


def reg_analysis(x, y, save_dir, show=False):
    m, b = np.polyfit(x, y, 1)

    line_equation = f'{round(m,2)}x + {round(b, 2)}'

    plt.figure()

    # Data points
    plt.plot(x, y, 'o')

    # Regression Line
    plt.plot(x, m*x + b, label=line_equation)

    plt.xlabel('N200 Latency')
    plt.ylabel('resp. time (10th Percentile)')

    plt.title('Resp. Times vs. N200 latencies')

    plt.legend()

    if save_dir:
        plt.savefig(save_dir/'lin_reg.png', dpi=300)

    if show:
        plt.show(block=False)


    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)

    reg = LinearRegression().fit(x, y)

    score = reg.score(x, y)
    coef = reg.coef_
    intercept = reg.intercept_
    # reg.predict(np.array([[3]]))

    print(
            f'{line_equation = }'
            f'{score = }'
            f'{coef = }'
            f'{intercept = }'
            )

    return reg


def select_files(directory, pattern) -> list:
    
    if type(pattern) == str :
        pattern = [pattern]
    
    if type(directory) != pathlib.WindowsPath:
        directory = pathlib.Path(directory)

    selection = []
    
    for pat in pattern:
        
        files = [f for f in directory.iterdir() if (re.search(pat, f.name) is 
                    not None) & (f.is_file())]
        
        selection.extend(files)
        
    new_dir = directory/'Selected_Files'
    
    new_dir.mkdir()
    
    new_paths = [new_dir/f.name for f in selection]
    
    for original, copy in zip(selection, new_paths):
        shutil.copy(original, copy)
    
    # NEXT -> Copy files to new_dir


def data_struct(data, save_dir):
    structure = [(k, type(data[sess][k])) for k in data[sess].keys()]
    structure[-1][1] == dict
    print(tabulate(structure))


def tree(directory):

    print(f'+ {directory}')
    
    for path in sorted(directory.rglob('*')):

        depth = len(path.relative_to(directory).parts)  - 1
        spacer = '\t' * depth

        # print(f'{spacer}+ {path.name}')

        if path.is_file():
            print(f'{spacer}f {path.name}')
        else:
            print(f'{spacer}d {path.name}')
