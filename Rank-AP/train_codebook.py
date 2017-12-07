from get_params import get_params
import os
import pickle

from sklearn.cluster import MiniBatchKMeans


def train_codebook(params,X):

        # Init kmeans instance/ A cada descriptor li assignem un centroide 
        km = MiniBatchKMeans(params['descriptor_size'])


    # Training the model with our descriptors

        km.fit(X)
        # Save to disk
        pickle.dump(km,open(os.path.join(params['root'],params['root_save'],
                                     params['codebooks_dir'],'codebook_'
                                     + str(params['descriptor_size']) + "_"
                                     + params['descriptor_type']
                                     + "_" + params['keypoint_type'] + '.cb'),'wb'))
        
        return km