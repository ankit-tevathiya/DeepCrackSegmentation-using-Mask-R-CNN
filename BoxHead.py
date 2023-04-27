import random 
import torch
import torch.nn.functional as F
from torch import nn

from .utils import *

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(BoxHead,self).__init__()

        self.C=Classes
        self.P=P
        self.device = device

        self.image_height = 800
        self.image_width = 1088
        # initialize BoxHead

        # intermediatse layer
        self.intermediate_layer = nn.Sequential(
            nn.Linear(256*(self.P**2),1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
        )
        # classifier layer
        self.classifier_layer = nn.Sequential(
            nn.Linear(1024,self.C+1),
        )

        # regressor layer
        self.regressor_layer = nn.Sequential(
            nn.Linear(1024,self.C*4),           
        )

        self.class_loss = nn.CrossEntropyLoss()
        self.regressor_loss = nn.SmoothL1Loss(reduction = 'sum')


    def create_ground_truth(self,proposals,gt_labels,bbox,th=0.5):
        '''
        This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
        Input:
            proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            gt_labels: list:len(bz) {(n_obj)}
            bbox: list:len(bz){(n_obj, 4)}
            th: threshold for IOU
        Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
            labels: (total_proposals,1) (the class that the proposal is assigned)
            regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
        '''
        labels = []
        regressor_target = []
        for i in range(len(proposals)):
            ious = iou(proposals[i],bbox[i])
            final_iou, indices = torch.max(ious, dim=1)   
            labels.append(torch.where(final_iou>th,1,0) * gt_labels[i][indices])

            gt_regress = bbox[i][indices]
            regressor_target.append(torch.vstack((((gt_regress[:,0] + gt_regress[:,2])/2 - (proposals[i][:,0]+proposals[i][:,2])/2)/(proposals[i][:,2]-proposals[i][:,0]),
            ((gt_regress[:,1] + gt_regress[:,3])/2 - (proposals[i][:,1]+proposals[i][:,3])/2)/(proposals[i][:,3]-proposals[i][:,1]),
            torch.log((gt_regress[:,2] - gt_regress[:,0])/ (proposals[i][:,2] - proposals[i][:,0])),
            torch.log((gt_regress[:,3] - gt_regress[:,1])/ (proposals[i][:,3] - proposals[i][:,1])))).T)            

        return torch.hstack(labels).to(self.device),torch.vstack(regressor_target).to(self.device)


    def postprocess_detections(self,prediction, conf_thresh=0.5,keep_topK = 200, keep_num_preNMS=20, keep_num_postNMS=5):
        '''
        This function does the post processing for the results of the Box Head for a batch of images
        Use the proposals to distinguish the outputs from each image
        Input:
                class_logits: (total_proposals,(C+1))
                box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
                proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
                conf_thresh: scalar
                keep_num_preNMS: scalar (number of boxes to keep pre NMS)
                keep_num_postNMS: scalar (number of boxes to keep post NMS)
        Output:
                boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
                scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
                labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        '''
        conf_score,pred_labels = torch.max(F.softmax(prediction['class_logits'],dim=1),dim=1)
        
        # indices = torch.where(prediction['labels']>0)
        # print("Total Class Accuracy\t\t: ",np.round((torch.where(pred_labels==prediction['labels'],1,0).sum()/len(pred_labels)*100).item(),2))
        # print("No Background - Class Recall\t: ",np.round((torch.where(prediction['labels'][indices] == pred_labels[indices],1,0).sum()/len(indices[0])).item()*100,2))
        # indices = torch.where(pred_labels>0)
        # print("No Background - Class Precision\t: ",np.round((torch.where(prediction['labels'][indices] == pred_labels[indices],1,0).sum()/len(indices[0])).item()*100,2))

        pos_indices = torch.where(pred_labels>0)[0]
        if len(pos_indices)==0:
          final_pred, final_conf, final_label = [None], [None], [None]
          return final_pred, final_conf, final_label
        all_indices = (pred_labels[pos_indices].reshape(-1,1)-1)*4
        all_indices = torch.hstack((all_indices,all_indices+1,all_indices+2,all_indices+3))
        box_pred = prediction['box_pred'][pos_indices]
        box_pred = output_decoding_boxhead(torch.stack([box_pred[i][all_indices[i]] for i in range(len(pos_indices))]),torch.vstack(prediction['proposals'])[pos_indices], device=self.device)
        gt_box_pred = output_decoding_boxhead(prediction['regressor_target'][pos_indices],torch.vstack(prediction['proposals'])[pos_indices], device=self.device)
    

        # postprocess a single image
        '''Remove gegative coners'''
        indices = torch.where(box_pred<0)
        box_pred[indices[0],:] = 0
        '''Remove ones outside positive edge of image'''
        indices = torch.where((box_pred[:,0]>self.image_width)|(box_pred[:,2]>self.image_width)|(box_pred[:,1]>self.image_height)|(box_pred[:,3]>self.image_height)) 
        box_pred[indices[0],:] = 0
        '''Remove low confident scores'''
        indices = torch.where(conf_score[pos_indices]<conf_thresh)
        box_pred[indices[0],:] = 0
        '''find the ones with 0'''
        indices = torch.all(box_pred, dim=1)

        box_pred = box_pred[indices]
        gt_box_pred = gt_box_pred[indices]
        conf_score = conf_score[pos_indices][indices]
        pred_labels = pred_labels[pos_indices][indices]

        image_index = torch.div(pos_indices, keep_topK, rounding_mode='trunc')
        image_index = image_index[indices]
        
        final_pred = []
        final_label = []
        final_conf = []
        for i in range(int(prediction['labels'].shape[0]/keep_topK)):
            indices = torch.where(image_index==i)[0]
            label,conf,pred = self.single_image_postprocess_detections(pred_labels[indices],conf_score[indices],box_pred[indices],keep_num_postNMS,keep_num_preNMS)
            final_pred.append(pred)
            final_label.append(label)
            final_conf.append(conf)
        
        # plot_predictions(images,final_pred,final_label,bboxes)

        return final_pred, final_conf, final_label
    
    def single_image_postprocess_detections(self,pred_labels,conf_score,box_pred, keep_num_postNMS = 2, keep_num_preNMS = 15):
        '''Individual image'''
        final_pred = []
        final_label = []
        final_conf = []

        for j in {1,2,3}:
            ind = torch.where(pred_labels==j)[0]
            if (len(ind)>0):
                # pre NMS
                if len(ind>keep_num_preNMS):
                    score, box = self.sort_subset(conf_score[ind],box_pred[ind],keep_num_preNMS)
                else:
                    score, box = conf_score[ind],box_pred[ind]

                # post NMS
                if len(ind)>keep_num_postNMS:
                    scores,pred = self.NMS(score,box)
                    final_pred.append(pred[:keep_num_postNMS])
                    final_conf.append(scores[:keep_num_postNMS])
                    final_label.append(torch.repeat_interleave(torch.tensor(j),keep_num_postNMS))
                else:
                    final_pred.append(box)
                    final_conf.append(score)
                    final_label.append(torch.repeat_interleave(torch.tensor(j),len(ind)))
        return torch.hstack(final_label), torch.hstack(final_conf), torch.vstack(final_pred)



    def compute_loss(self,class_logits, box_pred, labels, regressor_target,l=1,effective_batch=150):    
        '''        
        Compute the total loss of the classifier and the regressor
        Input:
            class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
            box_preds: (total_proposals,4*C)      (as outputed from forward)
            labels: (total_proposals,1)
            regression_targets: (total_proposals,4)
            l: scalar (weighting of the two losses)
            effective_batch: scalar
        Outpus:
            loss: scalar
            loss_class: scalar
            loss_regr: scalar
        '''

        pos_indices = torch.where(labels>0)[0]

        if len(pos_indices) > int(effective_batch*0.75):
            ind = sorted(random.sample(range(len(pos_indices)), int(effective_batch*0.75)))
            pos_indices = pos_indices[ind]

        neg_indices = torch.where(labels==0)[0]
        if len(neg_indices) > (effective_batch - len(pos_indices)):
            ind = sorted(random.sample(range(len(neg_indices)), int(effective_batch - len(pos_indices))))
            neg_indices = neg_indices[ind]

        all_indices = torch.cat((pos_indices,neg_indices))
        n = len(all_indices)
        
        # classification loss
        loss_class =  self.class_loss(class_logits[all_indices],labels[all_indices])

        all_indices = (labels[pos_indices].reshape(-1,1)-1)*4
        all_indices = torch.hstack((all_indices,all_indices+1,all_indices+2,all_indices+3))
        box_pred = box_pred[pos_indices]

        # regression loss
        loss_regr = self.regressor_loss(torch.stack([box_pred[i][all_indices[i]] for i in range(len(pos_indices))]),regressor_target[pos_indices]) /n 

        # total loss
        loss = loss_class + loss_regr*l

        return loss, loss_class, loss_regr


    def forward(self, feature_vectors):
        '''
        Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
        Input:
            feature_vectors: (total_proposals, 256*P*P)
        Outputs:
            class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
                                                    CrossEntropyLoss you should not pass the output through softmax here)
            box_pred:     (total_proposals,4*C)
        '''
        inter = self.intermediate_layer(feature_vectors)    
        class_logits = self.classifier_layer(inter)
        box_pred     = self.regressor_layer(inter) 
        return class_logits, box_pred
        
    
    def sort_subset(self,score,box,keep_num_preNMS):
        score, ind = torch.sort(score,descending=True)
        return score[:keep_num_preNMS],box[ind][:keep_num_preNMS]

    
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
        scores, indices = torch.sort(scores,descending=True)
        pre_box = pre_box[indices]
        ious = iou(pre_box,pre_box)
        ious = ious.fill_diagonal_(0)
        ious_cmax = ious.max(0)[0].expand(n, n).T
        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)
        decay = decay.min(dim=0)[0]
        scores = scores * decay

        scores, indices = torch.sort(scores,descending=True)
        pre_box = pre_box[indices]
        return scores,pre_box

