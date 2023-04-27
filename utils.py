import numpy as np
import torch
import torchvision
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))


def single_box_iou(boxA, boxB):
    '''
    compute the IOU between the boxA, boxB boxes
    '''
    x1, y1 = torch.max(boxA[0], boxB[0]), torch.min(boxA[1], boxB[1])
    x2, y2 = torch.min(boxA[2], boxB[2]), torch.max(boxA[3], boxB[3])
    intersection = torch.abs(x1-x2) * torch.abs(y1-y2)  
    ar1 = torch.abs(boxA[0]-boxA[2]) * torch.abs(boxA[1]-boxA[3])
    ar2 = torch.abs(boxB[0]-boxB[2]) * torch.abs(boxB[1]-boxB[3])
    return intersection/(ar1+ar2-intersection)


def iou(bbox, possible_anchor):
    '''
    Vectorized IoU for all pred, target
    '''
    pred_area = ((bbox[:,3]-bbox[:,1]) * (bbox[:,2]-bbox[:,0]).T).reshape(-1,1)
    area_gt = ((possible_anchor[:,3]-possible_anchor[:,1]) * (possible_anchor[:,2]-possible_anchor[:,0]).T).reshape(-1,1)
    intersect_width = torch.maximum(torch.minimum(bbox[:,2].reshape(-1,1),possible_anchor[:,2].reshape(-1,1).T)- torch.maximum(bbox[:,0].reshape(-1,1),possible_anchor[:,0].reshape(-1,1).T),torch.tensor(0))
    intersect_height = torch.maximum(torch.minimum(bbox[:,3].reshape(-1,1),possible_anchor[:,3].reshape(-1,1).T)- torch.maximum(bbox[:,1].reshape(-1,1),possible_anchor[:,1].reshape(-1,1).T),torch.tensor(0))
    # compute intersect area
    area_intersect = intersect_width * intersect_height
    # compute union area
    area_union = (pred_area + area_gt.T) - area_intersect
    return (area_intersect / area_union)



def output_flattening(out_r, out_c, anchors_list):
    '''
    This function flattens the output of the network and the corresponding anchors
    in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
    the FPN levels from all the images into 2D matrices
    Each row correspond of the 2D matrices corresponds to a specific grid cell
    Input:
        out_r: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}
        out_c: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
        anchors: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
    Output:
        flatten_regr: (total_number_of_anchors*bz,4)
        flatten_clas: (total_number_of_anchors*bz)
        flatten_anchors: (total_number_of_anchors*bz,4)
    '''
    flatten_clas = torch.hstack([v.flatten() for v in out_c])
    flatten_regr = torch.vstack([v.view(4,3,-1).view(-1,4) for v in out_r])
    if anchors_list!=None:
        flatten_anchors = torch.vstack([v.view(-1,4) for v in anchors_list])
    else:
        flatten_anchors=[]

    return flatten_regr, flatten_clas, flatten_anchors


def output_decoding(flatten_out, flatten_anchors, device='cpu'):
    '''
    This function decodes the output that are given in the [t_x,t_y,t_w,t_h] format
    into box coordinates where it returns the upper left and lower right corner of the bbox
    Input:
        flatten_out: (total_number_of_anchors*bz,4)
        flatten_anchors: (total_number_of_anchors*bz,4)
    Output:
        box: (total_number_of_anchors*bz,4)
    '''
    x_b = torch.add(flatten_out[:,0,:]*flatten_anchors[:,2],flatten_anchors[:,0])
    y_b = torch.add(flatten_out[:,1,:]*flatten_anchors[:,3],flatten_anchors[:,1])
    w_b = torch.exp(flatten_out[:,2,:])*flatten_anchors[:,2]
    h_b = torch.exp(flatten_out[:,3,:])*flatten_anchors[:,3]

    x1 = torch.subtract(x_b, w_b/2)
    x2 = torch.add(x_b, w_b/2)
    y1 = torch.subtract(y_b, h_b/2)
    y2 = torch.add(y_b, h_b/2)
    box = torch.stack((x1,y1,x2,y2),dim=1)

    return box


def output_decoding_boxhead(regressor_target,proposals, device='cuda:0'):
    '''
    This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
    into box coordinates where it return the upper left and lower right corner of the bbox
    Input:
        regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
        flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
    Output:
        box: (total_proposals,4) ([x1,y1,x2,y2] format)
    '''
    if type(proposals)==list:
        proposals = torch.vstack(proposals).clone()
    w = torch.exp(regressor_target[:,2]) * (proposals[:,2] - proposals[:,0])
    h = torch.exp(regressor_target[:,3]) * (proposals[:,3] - proposals[:,1])
    x = regressor_target[:,0]*(proposals[:,2] - proposals[:,0]) + (proposals[:,0]+proposals[:,2])/2
    y = regressor_target[:,1]*(proposals[:,3] - proposals[:,1]) + (proposals[:,1]+proposals[:,3])/2
    box = torch.stack((x-(w/2), y-(h/2), x+(w/2), y+(h/2)),dim=1)
    return box.to(device)


def MultiScaleRoiAlign(self, fpn_feat_list,proposals):
        '''
        This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
        a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
        Input:
            fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
            proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            P: scalar
        Output:
            feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
        '''
        proposals = torch.stack(proposals).clone()
        widths = (proposals[:,:,2] - proposals[:,:,0])
        heights = (proposals[:,:,3] - proposals[:,:,1])
        k = torch.clip(torch.floor(4+ torch.log2(torch.sqrt(widths*heights)/224)),2,5).int()
        # [[fpn_feat_list[j-2][i] for j in k[i]] for i in range(len(k))]

        feature_vectors = []
        # for each image in the batch
        for i in range(len(k)):
            fpn_subset = [(fpn_feat_list[j-2][i]).unsqueeze(0) for j in k[i]]
            strides = torch.tensor(np.array([(self.image_width/v.shape[-1],self.image_height/v.shape[-2]) for v in fpn_subset]),device=self.device)

            proposals[i][:,0]  = proposals[i][:,0] / strides[:,0]
            proposals[i][:,1]  = proposals[i][:,1] / strides[:,1]
            proposals[i][:,2]  = proposals[i][:,2] / strides[:,0]
            proposals[i][:,3]  = proposals[i][:,3] / strides[:,1]

            temp = torch.stack([torchvision.ops.roi_align(fpn_subset[j], [proposals[i][j].reshape(1,-1)], output_size=self.P, 
                                                spatial_scale=1,
                                                sampling_ratio=-1).view(-1) for j in range(proposals.shape[1])])
            feature_vectors.append(temp)

        return torch.vstack(feature_vectors).to(self.device)

def plot_proposed_box(image,selected_anchor,conf):
    '''Plot selected anchors'''
    img = (image.cpu().permute(1,2,0).numpy()).copy()
    img = np.clip(img, 0, 1)
    thickness = 3
    plt.imshow(img)
    for k in range(len(selected_anchor)):
        x1,y1,x2,y2 = selected_anchor[k]
        rect=patches.Rectangle((int(x1),int(y1)),int(x2-x1),int(y2-y1), 
                            fill=False,
                            color='red',
                            linewidth=thickness)
        plt.gca().add_patch(rect)
        rx, ry = rect.get_xy()
        plt.annotate(np.round(conf[k].item(),2)*100, (rx, ry), color='w', weight='bold',fontsize=6, ha='center', va='center')
    
    plt.title("Proposed Anchors")
    plt.show()


