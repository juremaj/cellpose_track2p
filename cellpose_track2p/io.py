
import os 
import numpy as np
import tifffile as tf
from tifffile import TiffFile

def tiff_loader(tiffs_path_chN, tiff_files_chN):
    full_tiff = []
    for tiff_file in tiff_files_chN:
        tiff_path = os.path.join(tiffs_path_chN, tiff_file)
        # read tiff but not just the first frame
        print(f'Loading {tiff_path}')
        # Load a subset of slices
        with TiffFile(tiff_path) as tif:
            n_pages = len(tif.pages)
            # close the file
            tif.close()
            
        tiff_np_subset = tf.imread(tiff_path, key=slice(0, n_pages))  # Load first 10 slices

        print(tiff_np_subset.shape)  # Should print (10, 512, 512)
        # put first dimension to last
        # tiff_np_subset = np.moveaxis(tiff_np_subset, 0, -1)
        full_tiff.append(tiff_np_subset)
    
    full_tiff = np.concatenate(full_tiff, axis=0)
    print(full_tiff.shape)
    return full_tiff

def get_session_s2p_paths(subject_id, take_first_nsessions=None):
    
    # get all suite2p/plane0 paths
    subject_path = os.path.join('data_proc', subject_id[:2], subject_id)

    # now get all sub-directories that end with '_a'
    session_paths = [os.path.join(subject_path, f) for f in os.listdir(subject_path) if f.endswith('_a')]
    session_paths.sort()

    take_first_nsessions = len(session_paths) if take_first_nsessions is None else take_first_nsessions

    session_paths = session_paths[:take_first_nsessions]
    print(session_paths)

    all_s2p_path = []

    for session_path in session_paths:
        s2p_path = os.path.join(session_path, 'suite2p', 'plane0')
        if os.path.exists(s2p_path):
            all_s2p_path.append(s2p_path)

    return session_paths, all_s2p_path

def load_all_ds_stat_iscell(all_ds_path, nplanes=1, iscell_thr=0):
    all_ds_stat_iscell = []
    for (i, ds_path) in enumerate(all_ds_path):
        ds_stat_iscell = []
        for j in range(nplanes):
            stat = np.load(os.path.join(ds_path, 'suite2p', f'plane{j}', 'stat.npy'), allow_pickle=True)
            iscell = np.load(os.path.join(ds_path, 'suite2p', f'plane{j}', 'iscell.npy'), allow_pickle=True)
            if iscell_thr==None:
                stat_iscell = stat[iscell[:,0]==1]
            else: 
                stat_iscell = stat[iscell[:,1]>iscell_thr]
            ds_stat_iscell.append(stat_iscell)
        all_ds_stat_iscell.append(ds_stat_iscell)

    return all_ds_stat_iscell

# functions from move_deve

def get_s2p_t2p_alldays(t2p_save_path, t2p_all_ds_path, t2p_iscell_thr, manually_cur=False, plane='plane0'):
    
    # t2p: getting cell match matrix and track_ops
    t2p_match_mat = np.load(os.path.join(t2p_save_path, f'{plane}_match_mat.npy'), allow_pickle=True)

    print('Datasets used for t2p:')
    for ds_path in t2p_all_ds_path:
        print(ds_path)

    # t2p: getting matches across all days
    t2p_match_mat_allday = t2p_match_mat[~np.any(t2p_match_mat==None, axis=1), :]
    print(f'\nNumber of cells matched across all days: {t2p_match_mat_allday.shape[0]}\nNumber of days: {t2p_match_mat_allday.shape[1]}')

    # s2p: loading stat, f and ops for all days indexed and reordered by matches on all days
    iscell_thr = t2p_iscell_thr # use the same threshold as when running the algo (to be consistent with indexing)

    all_stat_t2p = []
    all_f_t2p = []
    all_f_neu_t2p = []
    all_spks_t2p = []   
    all_ops = [] # ops dont change

    for (i, ds_path) in enumerate(t2p_all_ds_path):
        ops = np.load(os.path.join(ds_path, 'suite2p', plane, 'ops.npy'), allow_pickle=True).item()
        stat = np.load(os.path.join(ds_path, 'suite2p', plane, 'stat.npy'), allow_pickle=True)
        f = np.load(os.path.join(ds_path, 'suite2p', plane, 'F.npy'), allow_pickle=True)
        fneu = np.load(os.path.join(ds_path, 'suite2p', plane, 'Fneu.npy'), allow_pickle=True)
        spks = np.load(os.path.join(ds_path, 'suite2p', plane, 'spks.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(ds_path, 'suite2p', plane, 'iscell.npy'), allow_pickle=True)
        
        if manually_cur:
            iscell_bool = iscell[:,0].astype(bool)
        else:
            iscell_bool = iscell[:,1]>iscell_thr

        stat_iscell = stat[iscell_bool]
        f_iscell = f[iscell_bool, :]
        f_neu_iscell = fneu[iscell_bool, :]
        spks_iscell = spks[iscell_bool, :]
        
        stat_t2p = stat_iscell[t2p_match_mat_allday[:,i].astype(int)]
        f_t2p = f_iscell[t2p_match_mat_allday[:,i].astype(int), :]  
        f_neu_t2p = f_neu_iscell[t2p_match_mat_allday[:,i].astype(int), :]
        spks_t2p = spks_iscell[t2p_match_mat_allday[:,i].astype(int), :]

        all_stat_t2p.append(stat_t2p)
        all_f_t2p.append(f_t2p)
        all_f_neu_t2p.append(f_neu_t2p)
        all_spks_t2p.append(spks_t2p)
        all_ops.append(ops)

    return all_stat_t2p, all_f_t2p, all_f_neu_t2p, all_spks_t2p, all_ops