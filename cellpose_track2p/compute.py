import os 
import numpy as np
import matplotlib.pyplot as plt
from suite2p.extraction import dcnv

def str_gad(string, subject_id):
    # wherever there is a subject_id, in string replace it with subject_id string but added string `gad`
    return string.replace(subject_id, subject_id + '_gad')

def get_spks(F, Fneu, ops):
    # get spks
    Fc = F - ops['neucoeff'] * Fneu

    # baseline operation
    Fc = dcnv.preprocess(
        F=Fc,
        baseline=ops['baseline'],
        win_baseline=ops['win_baseline'],
        sig_baseline=ops['sig_baseline'],
        fs=ops['fs'],
        prctile_baseline=ops['prctile_baseline']
    )

    # get spikes
    spks = dcnv.oasis(F=Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])
    
    return spks

def generate_all_meantiff(all_s2p_path):
    subject_id = all_s2p_path[0].split(os.sep)[2]
    print(subject_id)
    all_ch2_meanimg = []

    for s2p_path in all_s2p_path:
        ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()

        ch2_meanimg = ops['meanImg_chan2']
        all_ch2_meanimg.append(ch2_meanimg)

        ch2_meanimg_uint16 = (ch2_meanimg*2).astype(np.uint16)
        tiff_path = os.path.join(s2p_path, 'meanImg_chan2.tiff')
        plt.imsave(str_gad(tiff_path, subject_id), ch2_meanimg_uint16, cmap='gray')

def masks_to_stat(masks, session_paths, session_paths_cellpose):

    for i, img_masks in enumerate(masks):
        stat = []
        for u in np.unique(img_masks)[1:]:
            ypix, xpix = np.where(img_masks==u)
            # get the centroid
            med = [int(np.median(ypix)), int(np.median(xpix))]
            npix = len(xpix)
            lam = np.ones(npix, np.float32)
            radius = np.sqrt(npix/np.pi)
            stat.append({'xpix': xpix, 'ypix': ypix, 'med': med, 'npix': npix, 'lam': lam, 'radius': radius})
        
        iscell = np.ones((len(stat), 2))
        redcell = np.copy(iscell)
        
        ops = np.load(os.path.join(session_paths[i], 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()

        stat_save_path = os.path.join(session_paths_cellpose[i], 'suite2p', 'plane0', 'stat.npy')
        iscell_save_path = os.path.join(session_paths_cellpose[i], 'suite2p', 'plane0', 'iscell.npy')
        redcell_save_path = os.path.join(session_paths_cellpose[i], 'suite2p', 'plane0', 'redcell.npy')
        ops_save_path = os.path.join(session_paths_cellpose[i], 'suite2p', 'plane0', 'ops.npy')

        # TODO: load ops and make sure to save it in the same path

        os.makedirs(os.path.dirname(stat_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(iscell_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(redcell_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(ops_save_path), exist_ok=True)

        np.save(stat_save_path, stat)
        np.save(iscell_save_path, iscell)
        np.save(redcell_save_path, redcell)
        np.save(ops_save_path, ops)