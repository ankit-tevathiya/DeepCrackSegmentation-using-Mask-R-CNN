from .utils import *
from .rpn import *
from .backbone import *

import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks

import matplotlib.pyplot as plt

backbone = Resnet50Backbone(device="cuda:0", eval=False)

class BasicModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
        self.rpn_net = RPNHead()
        self.train_loss_list = {'loss':[], 'loss_c':[],'loss_r':[]}
        self.val_loss_list = {'loss':[], 'loss_c':[],'loss_r':[]}
    
    def training_step(self, batch, batch_idx):
        images, _, _, bboxes, indexes  = batch.values()
        feature_pyramid = [v.detach() for v in backbone(images.type(torch.float)).values()] 
        pred_clas,pred_coord = self.rpn_net.forward(feature_pyramid)
        ground_clas, ground_coord = self.rpn_net.create_batch_truth(bboxes,indexes)
        loss, loss_c, loss_r = self.rpn_net.compute_loss(pred_clas,pred_coord,ground_clas,ground_coord)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss":loss,"loss_c":loss_c,"loss_r":loss_r}
    
    def training_epoch_end(self, training_step_outputs):
        temp1, temp2, temp3 = [],[],[]
        step_number = []
        for out in training_step_outputs:
            temp1.append(out['loss'].item())
            temp2.append(out['loss_c'].item())
            temp3.append(out['loss_r'].item())
            step_number.append(1)
        self.train_loss_list['loss'].append(torch.sum(torch.tensor(temp1)).item()/sum(step_number))
        self.train_loss_list['loss_c'].append(torch.sum(torch.tensor(temp2)).item()/sum(step_number))
        self.train_loss_list['loss_r'].append(torch.sum(torch.tensor(temp3)).item()/sum(step_number))
    
    def validation_step(self, batch, batch_idx):
        images, _, _, bboxes, indexes  = batch.values()
        images, _, _, bboxes, indexes  = batch.values()
        feature_pyramid = [v.detach() for v in self.backbone(images.type(torch.float)).values()] 
        pred_clas,pred_coord = self.rpn_net.forward(feature_pyramid)
        ground_clas, ground_coord = self.rpn_net.create_batch_truth(bboxes,indexes)
        loss, loss_c, loss_r = self.rpn_net.compute_loss(pred_clas,pred_coord,ground_clas,ground_coord)
        self.log("val_loss", loss, prog_bar=True)
        return {"loss":loss,"loss_c":loss_c,"loss_r":loss_r}

    def validation_epoch_end(self, outputs):
        temp1, temp2, temp3 = [],[],[]
        step_number = []
        for out in outputs:
            temp1.append(out['loss'].item())
            temp2.append(out['loss_c'].item())
            temp3.append(out['loss_r'].item())
            step_number.append(1)
        self.val_loss_list['loss'].append(torch.sum(torch.tensor(temp1)).item()/sum(step_number))
        self.val_loss_list['loss_c'].append(torch.sum(torch.tensor(temp2)).item()/sum(step_number))
        self.val_loss_list['loss_r'].append(torch.sum(torch.tensor(temp3)).item()/sum(step_number))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
            # plt.savefig(f'/Users/smrutichourasia/Desktop/Study/Fall 22/CIS680/HW4/loss/{list(self.train_loss_list.keys())[i]}_loss.jpg')
    
    def proposal_plot(self,images):
      self.rpn_net.proposal_plot(images)
    
    