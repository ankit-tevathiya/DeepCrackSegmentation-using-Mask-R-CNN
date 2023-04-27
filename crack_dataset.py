import torch
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader

import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class CrackDataset(torch.utils.data.Dataset):
    def __init__(self,path,mask_path):
        '''
        Initialize  Dataset
        path = [imgs_path, masks_path, labels_path, bboxes_path]     
        '''   
        self.path = path  
        self.mask_path = mask_path
        img_list = os.listdir(self.path)
        self.img_list = pd.Series(img_list).str.split(".").apply(lambda x: x[0]).to_list()

        # check shapes
        print("Image ",len(self.img_list))

        self.preprocess_images = trans.Compose([
            lambda x: x/255,
            trans.ToTensor(),
            trans.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
            lambda x: x.type(torch.float)
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

        img = Image.open(self.path+str(self.img_list[index])+'.jpg')
        img = self.preprocess_images(np.array(img))

        mask = Image.open(self.mask_path+str(self.img_list[index])+'.png')
        mask = (np.array(mask)>0).astype(np.uint8)

        target = {}
        target['boxes'] = self.get_box(mask)
        target['labels'] = torch.tensor([1])
        target['masks'] = torch.tensor(mask).unsqueeze(0).type(torch.uint8)
        target['image_id'] = torch.tensor(index)
        target['area'] = area =(target['boxes'][:, 2]-target['boxes'][:, 0])*(target['boxes'][:, 3]-target['boxes'][:, 1])
        target['iscrowd'] = torch.zeros((target['boxes'].shape[0],), dtype=torch.int64)
        return img, target

    def get_box(self, mask):
        ind = np.where(mask==1)
        x1, x2 = min(ind[1]), max(ind[1])
        y1, y2 = min(ind[0]), max(ind[0])
        return torch.tensor([[x1,y1,x2,y2]])
    
    def __len__(self):
        return len(self.img_list)


class CrackDataLoader(torch.utils.data.DataLoader):
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
        image, target  = list(zip(*data))
        return (image,target)

    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)
    
    
 