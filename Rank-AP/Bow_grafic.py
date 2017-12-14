from get_params import get_params
import numpy as np
from get_features import get_features
from eval_rankings import eval_rankings
import matplotlib.pyplot as plt

def grafic (params,etiqueta):
    
    params['split']=etiqueta
    get_features(params)
    ap_list,dict_=eval_rankings(params)
    mean=np.mean(dict_)
    return mean
    
if __name__ == "__main__":
    
    params = get_params()
    params['descriptor_size'] = 1024
    mean=grafic(params,'train')
    meant=grafic(params,'val')
    train=[mean]
    val=[meant]
    

    params['descriptor_size'] = 256
    mean=grafic(params,'train')
    meant=grafic(params,'val')
    train.append(mean)
    val.append(meant)
    
    
    
    params['descriptor_size'] = 2048
    mean=grafic(params,'train')
    meant=grafic(params,'val')
    train.append(mean)
    val.append(meant)
    
    
    params['descriptor_size'] = 512
    mean=grafic(params,'train')
    meant=grafic(params,'val')
    train.append(mean)
    val.append(meant)
    
    
    params['descriptor_size'] = 3072
    mean=grafic(params,'train')
    meant=grafic(params,'val')
    train.append(mean)
    val.append(meant)


size=[256,512,1024,2048,3072]

plt.figure('scatter')
plt.scatter(train,size)

plt.figure('plot')

plt.plot(val,size)
plt.show()