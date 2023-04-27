import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, Tensor
from dataset import *
from utils import *
import torchvision
from collections import OrderedDict
import random


class RPNHead(torch.nn.Module):
    # The input of the initialization of the RPN is:
    # Input:
    #       computed_anchors: the anchors computed in the dataset
    #       num_anchors: the number of anchors that are assigned to each grid cell
    #       in_channels: number of channels of the feature maps that are outputed from the backbone
    #       device: the device that we will run the model
    def __init__(self, num_anchors=3, in_channels=256, device='cuda',
                 anchors_param=dict(ratio=[[1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]],
                                    scale=[32, 64, 128, 256, 512],
                                    grid_size=[(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)],
                                    stride=[4, 8, 16, 32, 64])
                 ):
        
        self.image_height = 800
        self.image_width = 1088
        self.device = device

        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'])
        self.ground_dict={}

        # TODO initialize RPN
        # Define Intermediate Layer
        self.intermediate = nn.Sequential(OrderedDict([
            ('conv',nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),padding='same')),
            ('norm',nn.BatchNorm2d(256)),
            ('relu',nn.ReLU()),
        ]))

        # Define Proposal Classifier Head
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=3,kernel_size=(1,1),padding='same'),
            nn.Sigmoid(),
            )

        # Define Proposal Regressor Head
        self.regressor = nn.Sequential(
           nn.Conv2d(in_channels=256,out_channels=4*3,kernel_size=(1,1),padding='same')
        )


    def forward(self, X):
        '''
        Forward each level of the FPN output through the intermediate layer and the RPN heads
        Input:
            X: list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
        Ouput:
            logits: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
            bbox_regs: list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
        '''
        logits = []
        bbox_regs = []
        for i in range(len(X)):
            temp = self.forward_single(X[i])
            logits.append(temp[0])
            bbox_regs.append(temp[1])
        return logits, bbox_regs


    def forward_single(self, feature):
        '''
        Forward a single level of the FPN output through the intermediate layer and the RPN heads
        Input:
            feature: (bz,256,grid_size[0],grid_size[1])}
        Ouput:
            logit: (bz,1*num_acnhors,grid_size[0],grid_size[1])
            bbox_regs: (bz,4*num_anchors, grid_size[0],grid_size[1])
        '''
        #forward through the Intermediate layer
        X = self.intermediate(feature)
        #forward through the Classifier Head
        logit = self.classifier(X)
        #forward through the Regressor Head
        bbox_reg = self.regressor(X)

        return logit, bbox_reg


    def create_anchors(self, aspect_ratio, scale, grid_size, stride):
        ''' 
        This function creates the anchor boxes for all FPN level
            Input:
                aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
                scale:        list:len(FPN)
                grid_size:    list:len(FPN){tuple:len(2)}
                stride:        list:len(FPN)
            Output:
                anchors_list: list:len(FPN){(grid_size[0]*grid_size[1]*num_anchors,4)}
        '''
        anchors_list = []
        for i in range(len(grid_size)):
            anchors_list.append(self.create_anchors_single(aspect_ratio[i],scale[i],grid_size[i]))
        return anchors_list

    def create_anchors_single(self, ratio:list, scale, grid_sizes):
        '''
        This function creates the anchor boxes for one FPN level
        Input:
                aspect_ratio: list:len(number_of_aspect_ratios)
                scale: scalar
                grid_size: tuple:len(2)
                stride: scalar

        Output:
                Cross boundry: Which anchors to consider in the image grid
                possible anchor centers: 
                possible anchors: centers
                w: width of anchors
                h: height of anchors
        '''
        Sy = grid_sizes[0]
        Sx = grid_sizes[1]

        center_y, center_x = np.meshgrid([(self.image_height/Sy/2)+(self.image_height/Sy)*i for i in range(Sy)],[(self.image_width/Sx/2)+(self.image_width/Sx)*i for i in range(Sx)],indexing='ij')
        
        anchors = []

        for aspect_ratio in ratio:
                w,h = int((scale**(2)/aspect_ratio)**(1/2)*aspect_ratio),int((scale**(2)/aspect_ratio)**(1/2))
                cross_boundry = np.where(center_y - h/2<0,0,1) * np.where(center_y + h/2>self.image_height,0,1) * np.where(center_x - w/2<0,0,1) * np.where(center_x + w/2>self.image_width,0,1)
                # append center_x, center_y, w,h
                anchors.append(torch.tensor(np.stack([center_x*cross_boundry,center_y*cross_boundry,cross_boundry*np.ones_like(center_x)*w,cross_boundry*np.ones_like(center_x)*h],axis=2),device=self.device))
        return torch.stack(anchors)

    def get_anchors(self):
        return self.anchors


    def create_batch_truth(self, bboxes_list, indexes):
        '''
        This function creates the ground truth for a batch of images
        Input:
            bboxes_list: list:len(bz){(number_of_boxes,4)}
            indexes: list:len(bz)
            image_shape: list:len(bz){tuple:len(2)}
        Ouput:
            ground: list:len(FPN){(bz,num_anchors,grid_size[0],grid_size[1])}
            ground_coord: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
        '''
        # create ground truth for a batch of images
        ground_clas = []
        ground_coord = []
        anchors = self.get_anchors()

        for i in range(len(bboxes_list)):
            temp = self.create_ground_truth(bboxes_list[i],indexes[i],anchors)
            ground_clas.append(temp[0])
            ground_coord.append(temp[1])
        
        ground_coord = [torch.stack([j[i] for j in ground_clas]).shape for i in range(5)]
        ground_clas = [torch.stack([j[i] for j in ground_coord]).shape for i in range(5)]

        return ground_clas, ground_coord


    def create_ground_truth(self, bbox, index, anchors_list):
        '''
        This function create the ground truth for one image for all the FPN levels
        It also caches the ground truth for the image using its index
        Input:
            bboxes:      (n_boxes,4)
            index:       scalar (the index of the image in the total dataset)
            grid_size:   list:len(FPN){tuple:len(2)}
            anchor_list: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
        Output:
            ground_clas: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
            ground_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}
        '''
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        class_list = []
        coord_list = []
        for anchor in anchors_list:
            ground_clas = []
            ground_coord = []

            Sy, Sx = anchor.shape[1:3]

            # x1,y1,x2,y2
            conv_anchor = torch.stack([anchor[:,:,:,0] - (anchor[:,:,:,2]/2), anchor[:,:,:,1] - (anchor[:,:,:,3]/2), anchor[:,:,:,0] + (anchor[:,:,:,2]/2), anchor[:,:,:,1] + (anchor[:,:,:,3]/2)],dim=3)

            for anc in range(anchor.shape[0]):
                ind = conv_anchor[anc].any(axis=2)
                ious = iou(bbox,conv_anchor[anc][ind])
                max_ious = torch.max(ious,dim=1).values
                # given a max value - do or b/w max or iou > 0.7
                positive_labels = torch.logical_or(torch.where(ious > 0.7,1,0),torch.stack([torch.where(ious[i,:]==max_ious[i],1,0) for i in range(ious.shape[0])])).int()
                # do and b/w non positive and iou < 0.3
                negative_labels = torch.argwhere(torch.logical_and(torch.logical_not(positive_labels),torch.where(ious<0.3,1,0)).int()==1)
                positive_labels = torch.argwhere(positive_labels==1)
                ground_truth = torch.zeros((1,Sx,Sy),device=self.device)
                # negative lavels are assigned -1
                indices = torch.divide(anchor[anc][ind][negative_labels[:,1]][:,:2],torch.tensor([self.image_width/Sx,self.image_height/Sy],device=self.device)).long()
                ground_truth[:,indices[:,0],indices[:,1]]=-1
                # positive labels are assigned 1
                indices = torch.divide(anchor[anc][ind][positive_labels[:,1]][:,:2],torch.tensor([self.image_width/Sx,self.image_height/Sy],device=self.device)).long()
                ground_truth[:,indices[:,0],indices[:,1]]=1
                ground_clas.append(torch.transpose(ground_truth,1,2))

                ground_truth = torch.zeros((4,Sx,Sy),dtype=torch.float64,device=self.device)
                w = anchor[anc][ind][:,2][0].item()
                h = anchor[anc][ind][:,3][0].item()
                
                # for each bounding box
                for i in range(ious.shape[0]):
                    center = torch.stack(((torch.sum(bbox[i][[0,2]])/2) ,(torch.sum(bbox[i][[1,3]])/2)))
                    xy = torch.subtract(center,anchor[anc][ind][positive_labels[:,1][torch.where(positive_labels[:,0]==i)]][:,:2])
                    # encode and assign 
                    ground_truth[:,indices[torch.where(positive_labels[:,0]==i)][:,0],indices[torch.where(positive_labels[:,0]==i)][:,1]] = torch.stack((xy[:,0]/w,xy[:,1]/h, torch.repeat_interleave(torch.log(torch.diff(bbox[i][[0,2]])/w),len(xy)),torch.repeat_interleave(torch.log(torch.diff(bbox[i][[1,3]])/h),len(xy))))
                
                ground_coord.append(torch.transpose(ground_truth,1,2))

            class_list.append(torch.vstack(ground_clas))
            coord_list.append(torch.vstack(ground_coord))
        
        # store in dict
        self.ground_dict[key] = (class_list, coord_list)

        return class_list, coord_list

    def loss_class(self, p_out, n_out):
        '''
            Compute the loss of the classifier
            Input:
                 p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
                 n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
        '''
        # torch.nn.BCELoss()
        sum_count = len(p_out)
        loss = nn.BCELoss(reduction = 'sum')(torch.concat((p_out,n_out)),torch.concat((torch.ones(len(p_out),device=self.device),torch.zeros(len(n_out),device=self.device))))

        return loss, sum_count


    def loss_reg(self, pos_target_coord, pos_out_r):
        '''
            Compute the loss of the regressor
            Input:
                  pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
                  pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
        '''
        # torch.nn.SmoothL1Loss()
        sum_count = len(pos_target_coord)
        loss = nn.SmoothL1Loss(reduction = 'sum')(pos_target_coord,pos_out_r)

        return loss, sum_count


    def compute_loss(self, clas_out_list, regr_out_list, targ_clas_list, targ_regr_list, l=1, effective_batch=300):
        '''
        Compute the total loss for the FPN heads
        Input:
            clas_out_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
            regr_out_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
            targ_clas_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
            targ_regr_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
            l: weighting lambda between the two losses
        Output:
            loss: scalar
            loss_c: scalar
            loss_r: scalar
        ''' 
        # compute the total loss
    
        ground_coord, ground_clas, _ = output_flattening(targ_regr_list,targ_clas_list, None)
        pred_coord, pred_clas, _ = output_flattening(regr_out_list,clas_out_list, None)

        pos_indices = torch.where(ground_clas==1)
        if len(pos_indices[0]) > effective_batch/2:
            ind = sorted(random.sample(range(len(pos_indices[0])), int(effective_batch/2)))
            pos_indices = tuple(pos_indices[0][ind])

        neg_indices = torch.where(ground_clas==-1)
        if len(neg_indices[0]) > (effective_batch - len(pos_indices[0])):
            ind = sorted(random.sample(range(len(neg_indices[0])), int(effective_batch - len(pos_indices[0]))))
            neg_indices = tuple(neg_indices[0][ind])
        
        p_out = pred_clas[pos_indices]
        n_out = pred_clas[neg_indices]

        loss_c,pos = self.loss_class(p_out,n_out)
        loss_r,pos = self.loss_reg(ground_coord[pos_indices,:],pred_coord[pos_indices,:])

        # print("losses",loss_c,loss_r)
        
        loss = (loss_c/pos) + ((l*loss_r)/pos)


        return loss, loss_c, loss_r



    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=3):
        # postprocess a batch of images
        '''
        Post process for the outputs for a batch of images
        Input:
            out_c: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
            out_r: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
            IOU_thresh: scalar that is the IOU threshold for the NMS
            keep_num_preNMS: number of masks we will keep from each image before the NMS
            keep_num_postNMS: number of masks we will keep from each image after the NMS
        Output:
            nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
            nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinate of the boxes that the NMS kept)      
        '''
        nms_clas_list = []
        nms_prebox_list = []

        for i in range(out_c.shape[0]):
            temp = self.postprocessImg(out_c[i],out_r[i],keep_num_preNMS, keep_num_postNMS)
            nms_clas_list.append(temp[0])
            nms_prebox_list.append(temp[1])

        return torch.stack(nms_clas_list), torch.stack(nms_prebox_list)


    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        '''    
        Post process the output for one image
        Input:
            mat_clas: list:len(FPN){(1,1*num_anchors,grid_size[0],grid_size[1])}  (score of the output boxes)
            mat_coord: list:len(FPN){(1,4*num_anchors,grid_size[0],grid_size[1])} (encoded coordinates of the output boxess)
        Output:
            nms_clas: (Post_NMS_boxes)
            nms_prebox: (Post_NMS_boxes,4)
        '''
        
        anchors_list = self.get_anchors()
        flatten_out, flatten_clas, flatten_anchors = output_flattening(mat_coord,mat_clas, anchors_list)
        proposed_box = output_decoding(flatten_out, flatten_anchors,device=self.device)

        # postprocess a single image
        indices = torch.where(proposed_box<0)
        proposed_box[:,indices[1]] = 0
        boxes = proposed_box.T
        indices = torch.all(boxes, dim=1)
        boxes = boxes[indices]
        scores = flatten_clas[indices]

        # pre NMS
        scores, indices = torch.sort(scores,descending=True)
        scores = scores[:keep_num_preNMS]
        pre_box = boxes[indices][:keep_num_preNMS]

        scores = self.NMS(scores,pre_box)

        # post NMS
        scores, indices = torch.sort(scores,descending=True)
        nms_clas = scores[:keep_num_postNMS]
        nms_prebox = pre_box[indices][:keep_num_postNMS]
        return (nms_clas, nms_prebox)


    def NMS(self,scores,pre_box,method='gauss', gauss_sigma=0.5):
        '''
        Input:
          scores: (top_k_boxes) (scores of the top k boxes)
          prebox: (top_k_boxes,4) (coordinate of the top k boxes)
        Output:
          nms_clas: (Post_NMS_boxes)
          nms_prebox: (Post_NMS_boxes,4)
        '''
        # perform NMS
        n = len(scores)
        ious = iou(pre_box,pre_box)
        ious = ious.fill_diagonal_(0)
        ious_cmax = ious.max(0)[0].expand(n, n).T
        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)
        decay = decay.min(dim=0)[0]
        return scores * decay
    
    
    def proposal_plot(self,images):
        '''
        Input: 
            Images
        Output:
            Plot top poropsals output from network 
        '''
        pred_clas,pred_coord = self.forward(images.type(torch.float))
        nms_clas_list, nms_prebox_list = self.postprocess(pred_clas.detach(),pred_coord.detach())
        for i in range(images.shape[0]):
            plot_proposed_box(images[i],nms_prebox_list[i],nms_clas_list[i])
            
