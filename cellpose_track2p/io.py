
import os 
import numpy as np
import tifffile as tf
from tifffile import TiffFile

def tiff_loader(tiffs_path_chN, tiff_files_chN):
    full_tiff = []
    for tiff_file in tiff_files_chN:
        tiff_path = os.path.join(tiffs_path_chN, tiff_file)
        # read tiff but not just the first frame

        # Load a subset of slices
        with TiffFile(tiff_path) as tif:
            n_pages = len(tif.pages)
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