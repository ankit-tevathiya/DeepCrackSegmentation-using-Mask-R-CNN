from torch.optim import lr_scheduler, optimizer
from .utils import *
from .BoxHead import *
from .rpn import *
from .backbone import *

import torch
import torchvision
from torchvision.models.detection.image_list import ImageList

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks

import matplotlib.pyplot as plt

# PreTrained model
backbone = Resnet50Backbone(device="cuda:0", eval=False)
rpn = RPNHead().load_state_dict("content/rpn_model.pth")

class TrainModel(pl.LightningModule):

    def __init__(self,keep_topK=200):
        super(TrainModel,self).__init__()
        
        # FasterRCNN
        self.keep_topK = keep_topK
        self.faster_rcnn = BoxHead()

        # Loss List
        self.train_loss_list = {'loss':[], 'loss_class':[],'loss_regr':[]}
        self.val_loss_list = {'loss':[], 'loss_class':[],'loss_regr':[]}

    def training_step(self, batch, batch_idx):
        images, label, _, bboxes, _  = batch.values()
        backout = backbone(images.type(torch.float))
        # im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
        # rpnout = rpn(im_lis, backout)
        feature_pyramid = [v.detach() for v in backout.values()] 
        pred_clas,pred_coord = rpn.forward(feature_pyramid)
        proposals, _ = rpn.postprocess(pred_clas,pred_coord)

        proposals=[proposal[0:self.keep_topK,:] for proposal in proposals]
        fpn_feat_list= list(backout.values())

        labels,regressor_target = self.faster_rcnn.create_ground_truth(proposals,label,bboxes)
        feature_vectors = MultiScaleRoiAlign(fpn_feat_list,proposals)
        class_logits, box_pred = self.faster_rcnn.forward(feature_vectors)

        loss, loss_class, loss_regr = self.faster_rcnn.compute_loss(class_logits, box_pred, labels, regressor_target,l=6)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_class", loss_class)
        self.log("train_loss_regr", loss_regr)

        return {"loss":loss,"loss_class":loss_class,"loss_regr":loss_regr}  
    
    def training_epoch_end(self, training_step_outputs):
        temp1, temp2, temp3 = [],[],[]
        step_number = []
        for out in training_step_outputs:
            temp1.append(out['loss'].item())
            temp2.append(out['loss_class'].item())
            temp3.append(out['loss_regr'].item())
            step_number.append(1)
        self.train_loss_list['loss'].append(torch.sum(torch.tensor(temp1)).item()/sum(step_number))
        self.train_loss_list['loss_class'].append(torch.sum(torch.tensor(temp2)).item()/sum(step_number))
        self.train_loss_list['loss_regr'].append(torch.sum(torch.tensor(temp3)).item()/sum(step_number))
    
    def validation_step(self, batch, batch_idx):
        images, label, _, bboxes, _  = batch.values()
        backout = backbone(images.type(torch.float))
        # im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
        # rpnout = rpn(im_lis, backout)
        feature_pyramid = [v.detach() for v in backout.values()] 
        pred_clas,pred_coord = rpn.forward(feature_pyramid)
        proposals, _ = rpn.postprocess(pred_clas,pred_coord)

        proposals=[proposal[0:self.keep_topK,:] for proposal in proposals]
        fpn_feat_list= list(backout.values())

        labels,regressor_target = self.faster_rcnn.create_ground_truth(proposals,label,bboxes)
        feature_vectors = MultiScaleRoiAlign(fpn_feat_list,proposals)
        class_logits, box_pred = self.faster_rcnn.forward(feature_vectors)

        loss, loss_class, loss_regr = self.faster_rcnn.compute_loss(class_logits, box_pred, labels, regressor_target,l=6)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_class", loss_class)
        self.log("val_loss_regr", loss_regr)

        return {"loss":loss,"loss_class":loss_class,"loss_regr":loss_regr}
    
    def validation_epoch_end(self, outputs):
        temp1, temp2, temp3 = [],[],[]
        step_number = []
        for out in outputs:
            temp1.append(out['loss'].item())
            temp2.append(out['loss_class'].item())
            temp3.append(out['loss_regr'].item())
            step_number.append(1)
        self.val_loss_list['loss'].append(torch.sum(torch.tensor(temp1)).item()/sum(step_number))
        self.val_loss_list['loss_class'].append(torch.sum(torch.tensor(temp2)).item()/sum(step_number))
        self.val_loss_list['loss_regr'].append(torch.sum(torch.tensor(temp3)).item()/sum(step_number))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.7,patience=2)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler,"monitor": "val_loss"}

    
    def loss_plots(self):
      for i in self.val_loss_list.keys():
        self.val_loss_list[i] = self.val_loss_list[i][1:]

      loss = ['Total Loss','Class Loss','Regression Loss']
      for i in range(len(self.train_loss_list)):
          plt.plot(self.train_loss_list[list(self.train_loss_list.keys())[i]],linewidth=2.5,color='coral',label='Train Loss')
          plt.plot(self.val_loss_list[list(self.train_loss_list.keys())[i]],linewidth=1.5,color='deepskyblue',label='Test Loss')
          plt.legend(['Train Loss','Val Loss'])
          plt.xlabel('Epochs')
          plt.ylabel('Loss')
          plt.title(loss[i])
          plt.show()
    
    def forward(self,batch):
        images, label, _, bboxes, _  = batch.values()
        backout = backbone(images.type(torch.float))
        feature_pyramid = [v.detach() for v in backout.values()] 
        pred_clas,pred_coord = rpn.forward(feature_pyramid)
        proposals, _ = rpn.postprocess(pred_clas,pred_coord)

        proposals=[proposal[0:self.keep_topK,:] for proposal in proposals]
        fpn_feat_list= list(backout.values())

        labels,regressor_target = self.faster_rcnn.create_ground_truth(proposals,label,bboxes)
        feature_vectors = MultiScaleRoiAlign(fpn_feat_list,proposals)
        class_logits, box_pred = self.faster_rcnn.forward(feature_vectors)

        return {"labels":labels,"class_logits":class_logits,"regressor_target":regressor_target,"box_pred":box_pred,"proposals":proposals} 

    