import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import *

# from tqdm import tqdm

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        '''
        Initialize  Dataset
        path = [imgs_path, masks_path, labels_path, bboxes_path]     
        '''     
        f = h5py.File(path[0], 'r+')
        self.image = f['data'][()]
        f.close()
        f = h5py.File(path[1], 'r+')
        self.mask = f['data'][()]
        f.close()
        self.label = np.load(path[2],allow_pickle=True)
        self.bbox = np.load(path[3],allow_pickle=True)
        self.mask = self.unflatten_mask()
        # check shapes
        print("Image ",self.image.shape,"\nMask ",len(self.mask),"\nLabels ",self.label.shape,"\nBBox ",self.bbox.shape)
        
        self.preprocess_images = T.Compose([
            lambda x: np.transpose(x, (1, 2, 0)),
            lambda x: x/255,
            T.ToTensor(),
            T.Resize(size = (800,1066)),
            T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
            T.Pad((11,0)),
            lambda x: x.type(torch.float)
            # lambda x: x.unsqueeze(0)
        ])
        self.preprocess_mask = T.Compose([
            lambda x: x.astype(float),
            T.ToTensor(),
            T.Resize(size = (800,1066),antialias=True),
            T.Pad((11,0)),
            lambda x: x.squeeze(0)
        ])

    def __getitem__(self, index):
        '''    
        In this function for given index we rescale the image and the corresponding  masks, boxes
        and we return them as output
        output:
            transed_img
            label
            transed_mask
            transed_bbox
            index        
        return transformed images,labels,masks,boxes,index
        '''
        if torch.is_tensor(index):
            index = index.tolist()
        img, msk, bbox = self.pre_process_batch(self.image[index],self.bbox[index],self.mask[index])
        assert img.shape == (3,800,1088)
        assert bbox.shape[0] == msk.shape[0]
        # sample = {'image': im, 'label': label, 'bounding_box': bbox, 'mask':m}
        # sample = (im,label,bbox,m)
        return img, torch.tensor(np.array(self.label[index])), bbox, msk, index

    def pre_process_batch(self, img, bbox,mask):
        '''
        Apply the correct transformation to the images,masks,boxes
        This function preprocess the given image, mask, box by rescaling them appropriately
        output:
               img: (3,800,1088)
               mask: (n_box,800,1088)
               box: (n_box,4)
        '''
        img = self.preprocess_images(img)
        temp = []
        for j in range(len(mask)):
            temp.append(self.preprocess_mask(mask[j]))
        mask = torch.stack(temp)
        scale = np.tile(np.tile(np.flipud(np.divide([800, 1088], [300, 400])),2),len(bbox)).reshape(len(bbox),4)
        bbox = torch.tensor((np.array(bbox)*scale))
        assert img.squeeze(0).shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        return img.squeeze(0), mask, bbox
    
    def __len__(self):
        return len(self.image)

    def unflatten_mask(self):
        '''Create global lists for masks considering flattened sizes'''
        global_counter = 0
        all_mask = []
        for i in range(len(self.label)):
            temp_mask = []
            for j in range(len(self.label[i])):
                m = self.mask[global_counter]
                global_counter += 1
                temp_mask.append(m)
            all_mask.append(temp_mask)
        return all_mask


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        '''    
        output:
         dict{images: (bz, 3, 800, 1088)
              labels: list:len(bz)
              masks: list:len(bz){(n_obj, 800,1088)}
              bbox: list:len(bz){(n_obj, 4)}
              index: list:len(bz)
        '''
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def collect_fn(self,data):
        image, label, bounding_box, mask, index  = list(zip(*data))
        return {'images':torch.stack(image),
                'labels':label, 
                'masks':mask,
                'bbox':bounding_box,
                'index':index}

    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)
    
    
    def plot_tranformed_image(self,index):
        image,labels,bbox,mask, _ = self.dataset[index]
        img = (image.permute(1,2,0).numpy()).copy()
        img = np.clip(img, 0, 1)

        # class Vehicle - red, class People - blue, class Animals - green  
        c = ['purple','blue','green'] 
        c_maps = ['jet','Blues','Greens']

        thickness = 3

        plt.imshow(img)
        for k in range(len(labels)):
            c_index = labels[k]-1
            x1,y1,x2,y2 = bbox[k]
            m = mask[k]
            rect=patches.Rectangle((int(x1),int(y1)),int(x2-x1),int(y2-y1), 
                                fill=False,
                                color=c[c_index],
                                linewidth=thickness)
            plt.gca().add_patch(rect)
            plt.imshow(m,cmap=c_maps[c_index],vmin=0,vmax=1,alpha=0.6*m)

        plt.title(f"Transformed Targets - Label {labels}")
        plt.show()
 