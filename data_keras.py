from tensorflow import keras
import os
import math
from PIL import Image as image
import numpy as np
voc_colormap = [[0, 0, 0], [245,222,179]]
class keras_data(keras.utils.Sequence):
    def __init__(self,root='data/',image_set='train',batch_size=2):
        super(keras_data,self).__init__()
        self.root=os.path.expanduser(root)
        self.image_set=image_set
        self.batch_size=batch_size
        voc_dir=os.path.join(self.root)
        image_dir=os.path.join(voc_dir,'image')
        mask_dir=os.path.join(voc_dir,'mask')
        splits_f=os.path.join(self.root, self.image_set + '.txt')
        with open(os.path.join(splits_f),'r') as f:
            self.file_name=[x.strip() for x in f.readlines()]

        self.image=[os.path.join(image_dir,x+'.jpg') for x in self.file_name]
        self.mask=[os.path.join(mask_dir,x+'.png') for x in self.file_name]
        assert (len(self.image)==len(self.mask))
    def __len__(self):
        return math.ceil(len(self.file_name)/self.batch_size)
    def __getitem__(self, item):
        try:
           img=self.image[item*self.batch_size:(item+1)*self.batch_size]
           mask=self.mask[item*self.batch_size:(item+1)*self.batch_size]
           mask=np.array([np.array(image.open(i))/255 for i in mask])
           mask_t=mask[:,:,:,None]
           img_t=np.array([np.array(image.open(i).convert('RGB')) for i in img])
        except Exception as e:
            print(e)
        return img_t,[mask_t]*5


def hist(label_true,label_pred,num_cls):
    # mask=(label_true>=0)&(label_true<num_cls)
    hist=np.bincount(label_pred.astype(int)*num_cls+label_true.astype(int),minlength=num_cls**2).reshape(num_cls,num_cls)
    return hist
def label_acc_score(label_true,label_pred,num_cls=2):
    hist_matrix=np.zeros((num_cls,num_cls))
    tmp=0
    for i,j in zip(label_true,label_pred):
        hist_matrix+=hist(i.cpu().numpy().flatten(),j.cpu().numpy().flatten(),num_cls)
        tmp+=1
    diag=np.diag(hist_matrix)
    # acc=diag/hist_matrix.sum()
    acc_cls=diag/hist_matrix.sum(axis=0)
    m_iou=diag/(hist_matrix.sum(axis=1)+hist_matrix.sum(axis=0)-diag)
    return acc_cls,m_iou,hist_matrix,tmp