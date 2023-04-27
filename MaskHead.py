import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable

from .utils import *


class MaskHead(torch.nn.Module):
    def __init__(self,Classes=3,P=14):
        self.C=Classes
        self.P=P
        

        self.backbone = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same')),
            ('norm1',nn.BatchNorm2d(256)), ##Possible change
            ('relu1',nn.ReLU()),
            ('pool1',nn.MaxPool2d(2, stride=2)), ##Possible change
            ('conv2',nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same')),
            ('norm2',nn.BatchNorm2d(256)),  
            ('relu2',nn.ReLU()),  
            ('pool2',nn.MaxPool2d(2, stride=2)),
            ('conv3',nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same')),
            ('norm3',nn.BatchNorm2d(256)),   
            ('relu3',nn.ReLU()), 
            ('pool3',nn.MaxPool2d(2, stride=2)),
            ('conv4',nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same')),
            ('norm4',nn.BatchNorm2d(256)),  
            ('relu4',nn.ReLU()),  
            ('pool4',nn.MaxPool2d(2, stride=2)),
            ('deconv1', nn.ConvTranspose2d(in_channels=256,out_channels=256, stride = 2, padding = 1))
            ('conv5',nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1)),
            ('sig', nn.Sigmoid),
            
        ]))



    def preprocess_ground_truth_creation(self, class_logits, box_regression, gt_labels ,masks , keep_num_preNMS=1000, keep_num_postNMS=100):
        '''
            This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
            and create the ground truth for the Mask Head
            
            Input:
                class_logits: (total_proposals,(C+1))
                box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
                gt_labels: list:len(bz) {(n_obj)}
                bbox: list:len(bz){(n_obj, 4)}
                masks: list:len(bz){(n_obj,800,1088)}
                IOU_thresh: scalar (threshold to filter regressed with low IOU with a bounding box)
                keep_num_preNMS: scalar (number of boxes to keep pre NMS)
                keep_num_postNMS: scalar (number of boxes to keep post NMS)
            Output:
                boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
                scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
                labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
                gt_masks: list:len(bz){(post_NMS_boxes_per_image,2*P,2*P)}
                
        '''
        final_pred = []
        final_label = []
        final_conf = []
        final_mask = []

        for j in {1,2,3}:
            ind = torch.where(class_logits==j)[0]
            if (len(ind)>0):
                if len(ind>keep_num_preNMS):
                    score, box = self.sort_subset(class_logits[ind],box_regression[ind],keep_num_preNMS)
                else:
                    score, box = class_logits[ind],box_regression[ind]

                if len(ind)>keep_num_postNMS:
                    if gt_labels[ind]==j:
                        scores,pred = self.NMS(score,box)
                        final_pred.append(pred[:keep_num_postNMS])
                        final_conf.append(scores[:keep_num_postNMS])
                        final_label.append(torch.repeat_interleave(torch.tensor(j),keep_num_postNMS))
                        final_mask.append(masks)
                else:
                    final_pred.append(box)
                    final_conf.append(score)
                    final_label.append(torch.repeat_interleave(torch.tensor(j),len(ind)))
        
        boxes, scores, labels, masks = torch.hstack(final_label), torch.hstack(final_conf), torch.vstack(final_pred),torch.vstack(masks)

        return boxes, scores, labels, final_mask



    def flatten_inputs(self,input_list):
        '''
        general function that takes the input list of tensors and concatenates them along the first tensor dimension
        Input:
            input_list: list:len(bz){(dim1,?)}
        Output:
            output_tensor: (sum_of_dim1,?)
        
        '''
        return torch.flatten(start_dim=0,end_dim=1)



    def postprocess_mask(self, masks_outputs, boxes, labels, image_size=(800,1088)):
        '''
        This function does the post processing for the result of the Mask Head for a batch of images. It project the predicted mask
        back to the original image size
        Use the regressed boxes to distinguish between the images
        Input:
            masks_outputs: (total_boxes,C,2*P,2*P)
            boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
            labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
            image_size: tuple:len(2)
        Output:
            projected_masks: list:len(bz){(post_NMS_boxes_per_image,image_size[0],image_size[1]
            
        '''
        masks_outputs = F.interpolate(masks_outputs,size=image_size,mode='bilinear')
        masks_outputs = (masks_outputs>0.5).to(masks_outputs.type)
        projected_masks=[]
        for b in range(boxes.shape[0]):
            for j in range(boxes.shape[1]):
                projected_masks.append(masks_outputs[b+j])

        return projected_masks





    def compute_loss(self,mask_output,labels,gt_masks):
        '''
        
        Compute the total loss of the Mask Head
        Input:
            mask_output: (total_boxes,C,2*P,2*P)
            labels: (total_boxes)
            gt_masks: (total_boxes,2*P,2*P)
        Output:
            mask_loss
        '''
        if labels.size():
        
            pos_indices = torch.where(labels>0)[0]
            positive_labels = labels[pos_indices.data].long()
            indices = torch.stack((pos_indices, positive_labels), dim=1)

            y_true = mask_output[indices[:,0].data,:,:]
            y_pred = gt_masks[indices[:,0].data,indices[:,1].data,:,:]

            mask_loss = F.binary_cross_entropy(y_pred, y_true)

        else:
            mask_loss = Variable(torch.FloatTensor([0]), requires_grad=False)
            if labels.is_cuda:
                mask_loss = mask_loss.cuda()

            return mask_loss




    def forward(self, features):
        '''
        Forward the pooled feature map Mask Head
        Input:
            features: (total_boxes, 256,P,P)
        Outputs:
            mask_outputs: (total_boxes,C,2*P,2*P)        
        '''

        mask_outputs = self.backbone(features)

        return mask_outputs

