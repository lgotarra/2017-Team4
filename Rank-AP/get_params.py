import os
import pandas as pd
import numpy as np

def get_params():

    '''
    Define dictionary with parameters
    '''
    params = {}


    # Source data
    params['root'] = '/home/oscarlinux/Escritorio/UPC/Q5/GDSA/'
    params['database'] = 'TerrassaBuildings900'

    # To generate

    # 'root_save' directory goes under 'root':
    params['root_save'] = 'save'

    # All the following go under 'root_save':
    params['image_lists'] = 'image_lists'
    params['feats_dir'] = 'features'
    params['rankings_dir'] = 'rankings'
    params['codebooks_dir'] = 'codebooks'
    params['kaggle_dir'] = 'kaggle'


    # Parameters
    params['split'] = 'val'
    params['descriptor_size'] = 1024 # Number of clusters
    params['descriptor_type'] = 'SIFT'
    params['keypoint_type'] = 'SIFT'
    params['max_size'] = 500 # Widht size
    params['distance_type'] = 'euclidean'
    params['save_for_kaggle'] = True


    # Normalization of local descriptors
    params['whiten'] = False
    params['normalize_feats'] = False
    params['scale'] = False


    # We read the training annotations to know the set of possible labels
    data = pd.read_csv(os.path.join(params['root'],params['database'],'train','annotation.txt'), sep='\t', header = 0)

    # Store them in the parameters dictionary for later use
    params['possible_labels'] = np.unique(data['ClassID'])

    create_dirs(params)

    return params


def make_dir(dir):
    '''
    Creates a directory if it does not exist
    dir: absolute path to directory to create
    '''
    if not os.path.isdir(dir):
        os.makedirs(dir)

def create_dirs(params):

    '''
    Create directories specified in params
    '''
    save_dir = os.path.join(params['root'], params['root_save'])

    make_dir(save_dir)
    make_dir(os.path.join(save_dir,params['image_lists']))
    make_dir(os.path.join(save_dir,params['feats_dir']))
    make_dir(os.path.join(save_dir,params['rankings_dir']))
    make_dir(os.path.join(save_dir,params['codebooks_dir']))
    make_dir(os.path.join(save_dir,params['kaggle_dir']))

    make_dir(os.path.join(save_dir,params['rankings_dir'],params['descriptor_type']))
    make_dir(os.path.join(save_dir,params['rankings_dir'],params['descriptor_type'],params['split']))



if __name__ == "__main__":
    params = get_params()
