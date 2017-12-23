# -*- coding: cp1252 -*-
from get_params import get_params
import sys
import os, time
import numpy as np
import pickle
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


def get_features(params,pca=None,scaler=None):

    # Read image names
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'],params['split'] + '.txt'),'r') as f:
        image_list = f.readlines()

    # Initialize keypoint detector and feature extractor
    #detector, extractor = init_detect_extract(params)

    # Initialize feature dictionary
    features = {}

    # Get trained codebook
    km = pickle.load(open(os.path.join(params['root'],params['root_save'],
                                     params['codebooks_dir'],'codebook_'
                                     + str(params['descriptor_size']) + "_"
                                     + params['descriptor_type']
                                     + "_" + params['keypoint_type'] + '.cb'),'rb'))

    for image_name in image_list:
        image_name=image_name.replace('\n','')
        # Read image
        im = cv2.imread(os.path.join(params['root'],params['database'],params['split'],'images',image_name))

        # Resize image
        im = resize_image(params,im)

        # Extract local features
        f_orb = image_local_features(im)
        f_sift=sift_features(im)

        # posem f_i que sigui igual que la dimensió f_o
        f_sift.resize(f_orb.shape)
        feats=np.concatenate((f_sift,f_orb))

        #f_root = image_local_features_root(im,detector,extractor)
        f_orb.resize(f_sift.shape)
        feats=np.concatenate((f_sift,f_orb))

        if feats is not None:

            if params['normalize_feats']:
                feats = normalize(feats)

            # If we scaled training features
            if scaler is not None:
                scaler.transform(feats)

            # Whiten if needed
            if pca is not None:

                pca.transform(feats)

            # Compute assignemnts
            assignments = get_assignments(km,feats)

            # Generate bow vector
            feats = bow(assignments,km)
        else:
            # Empty features
            feats = np.zeros(params['descriptor_size'])

        '''
        print f_sift[1]
        print f_orb[1]
        print len(f_orb)
        print len(f_sift)
        '''
        #print len(feats)

        # Add entry to dictionary
        features[image_name] = feats

    # Save dictionary to disk with unique name
    save_file = os.path.join(params['root'],params['root_save'],params['feats_dir'],
                             params['split'] + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p')

    pickle.dump(features,open(save_file,'wb'))


def resize_image(params,im):

    # Get image dimensions
    height, width = im.shape[:2]

    # If the image width is smaller than the proposed small dimension, keep the original size !
    resize_dim = min(params['max_size'],width)

    # We don't want to lose aspect ratio:
    dim = (resize_dim, height * resize_dim/width)

    # Resize and return new image
    return cv2.resize(im,dim)

'''
class RootSIFT:
    def __init__(self):
    	# initialize the SIFT feature extractor
        self.extractor = cv2.DescriptorExtractor_create("SIFT")
    def compute(self, image, kps, eps=1e-7):
    	# compute SIFT descriptors
    #	(kps, descs1) = self.extractor1.compute(image, kps)
        (kps, descs) = self.extractor.compute(image, kps)
        #desc2.resize(desc1.shape)
        #descs=np.concatenate((desc2,desc1))
    	# if there are no keypoints or descriptors, return an empty tuple
    	if len(kps) == 0:
    	   return ([], None)
    	# apply the Hellinger kernel by first L1-normalizing and taking the
    	# square-root
    	descs /= (descs.sum(axis=1, keepdims=True) + eps)
    	descs = np.sqrt(descs)
        return kps, descs
'''
def sift_features(image):
    if not image is None:
        #Extract sift descriptors
        sift = cv2.SIFT(200000)
        kp_sift = sift.detect(image,None)
        kp_sift, des_sift = sift.compute(image, kp_sift)
        #print len(kp_sift)
        #print len(des_sift)
        return des_sift

def image_local_features(image):
    #llegim la imatge:
    #img = cv2.imread(image)
    #Cambiem la mida de la imatge:
    if not image is None:

        #linea que soluciona un bug de opencv a python3
        #cv2.ocl.setUseOpenCL(False)

        # Creem l'objecte ORB que tindrà 200k keypoints. (Perametre que podem modificar per no saturar el programa)
        orb = cv2.ORB(200000)

        # Detectem els keypoints:
        kp_o = orb.detect(image,None)

        # Calculem els descriptors amb els keypoints trobats.
        kp, des= orb.compute(image, kp_o)

        return des

def stack_features(params):

    '''
    Get local features for all training images together
    '''

    # Init detector and extractor
    #detector, extractor = init_detect_extract(params)

    # Read image names
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'],params['split'] + '.txt'),'r') as f:
        image_list = f.readlines()

    X = []
    for image_name in image_list:
        image_name=image_name.replace('\n','') #Afegim el replace per ignorar el caràcter /n

        # Read image
        im = cv2.imread(os.path.join(params['root'],params['database'],params['split'],'images',image_name))

        # Resize image
        im = resize_image(params,im)

        #feats = image_local_features(im,detector,extractor)
        f_o=image_local_features(im)
        f_i=sift_features(im)
        # posem f_i que sigui igual que la dimensió f_o
        f_i.resize(f_o.shape)
        feats=np.concatenate((f_i,f_o))
        #detector,extractor=init_detect_extract(params)
        #f_r=image_local_features_root(im,detector,extractor)
        #f_o.resize(f_r.shape)
        #feats=np.concatenate((f_r,f_o))
        # Stack all local descriptors together

        if feats is not None:
            if len(X) == 0:

                X = feats
            else:
                X = np.vstack((X,feats))

    if params['normalize_feats']:
        X = normalize(X)

    if params['whiten']:
                            #n_components=128
        pca = PCA(whiten=True)
        pca.fit_transform(X)

    else:
        pca = None

    # Scale data to 0 mean and unit variance
    if params['scale']:

        scaler = StandardScaler()

        scaler.fit_transform(X)
    else:
        scaler = None

    return X, pca, scaler

def train_codebook(params,X):

        # Init kmeans instance
        km = MiniBatchKMeans(params['descriptor_size'])


    # Training the model with our descriptors

    # normalize dataset for easier parameter selection
        #X = StandardScaler().fit_transform(X)
        #kg = kneighbors_graph(X,n_neighbors=params['descriptor_size'], mode='distance',include_self=False)
        # make kg symmetric
        #kg = 0.5 * (kg + kg.T)

        #km = AgglomerativeClustering(connectivity=kg,linkage='ward',n_clusters=params['descriptor_size'])
        km.fit(X)
        # Save to disk
        pickle.dump(km,open(os.path.join(params['root'],params['root_save'],
                                     params['codebooks_dir'],'codebook_'
                                     + str(params['descriptor_size']) + "_"
                                     + params['descriptor_type']
                                     + "_" + params['keypoint_type'] + '.cb'),'wb'))

        return km

def get_assignments(km,descriptors):

    assignments = km.predict(descriptors)

    return assignments


def bow(assignments,km):

    # Initialize empty descriptor of the same length as the number of clusters
    descriptor = np.zeros(np.shape(km.cluster_centers_)[0])

    # Build vector of repetitions
    for a in assignments:

        descriptor[a] += 1

    # L2 normalize
    descriptor = normalize(descriptor)

    return descriptor



if __name__ == "__main__":

    params = get_params()

    # Change to training set
    params['split'] = 'train'

    print "Stacking features together..."
    # Save features for training set
    t = time.time()
    X, pca, scaler = stack_features(params)
    print "Done. Time elapsed:", time.time() - t
    print "Number of training features", np.shape(X)

    print "Training codebook..."
    t = time.time()
    train_codebook(params,X)
    print "Done. Time elapsed:", time.time() - t

    print "Storing bow features for train set..."
    t = time.time()
    get_features(params, pca,scaler)
    print "Done. Time elapsed:", time.time() - t

    params['split'] = 'val'

    print "Storing bow features for validation set..."
    t = time.time()
    get_features(params)
print "Done. Time elapsed:", time.time() - t