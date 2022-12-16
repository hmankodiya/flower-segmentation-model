import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchvision.models as models
from torchmetrics import JaccardIndex


def load_base(pretrained=True):
    base = models.segmentation.fcn_resnet50(pretrained=pretrained).to(device='cuda')
    for param in base.parameters():
        param.requires_grad = True
    base.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1))
    return base

class ResNetModel(pl.LightningModule):
    def __init__(self, resnet, optimizer, learning_rate=0.01) -> None:
        super(ResNetModel, self).__init__()
        self.resnet = resnet
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        # self.jaccard_index = JaccardIndex(task='multiclass', num_classes=2, average='weighted')
        self.jaccard_index = JaccardIndex(task='binary', num_classes=1, average='weighted')
    
    def _calc_loss(self, predicted_tensor, target_tensor):
        '''
        Calculates cross entropy loss.

        Args:
            predicted_tensor (tensor): model predictions.
            target_tensor (tensor): target tensors from dataset.

        Returns:
            tensor: Value tensor. 
        '''
        return self.loss(predicted_tensor, target_tensor) 
    
    def _evaluate_jaccard_index(self, prediction, target):
        '''
        Calculates intersection over union.        

        Args:
            prediction (tensor): predicted model outputs.
            target (tensor): target tensors from dataset.

        Returns:
            tensor: value tensor.
        '''
        return self.jaccard_index(prediction, target)
    
    def training_step_end(self, step_output):
        train_loss_step = step_output['loss']
        self.log_dict({'train_loss_step': train_loss_step})
        
    def training_epoch_end(self, outputs):
        train_loss_epoch = torch.as_tensor([i['loss'] for i in outputs]).mean()
        train_jaccard_epoch = torch.as_tensor([i['jaccard'] for i in outputs]).mean()

        self.log_dict({'train_loss_epoch': train_loss_epoch, 
                       'train_jaccard_epoch': train_loss_epoch}, prog_bar=True)
        
    def training_step(self, batch, batch_idx):
        '''
        Training the model on batch.

        Args:
            batch (list): image tensor, segmentation image tensor, input image path, segmentation image path.
            batch_idx (int): batch index.

        Returns:
            dict: {'loss':...}
        '''
        # input image tensor, segmentation image tensor, input image path, segmentation image path
        input_image_tensor, target_segmentation_tensor = batch[0], batch[1]
        model_output = self.resnet(input_image_tensor)['out']
        loss = self._calc_loss(model_output, target_segmentation_tensor)
        model_output_softmax = self.softmax(model_output)
        train_jaccard = self._evaluate_jaccard_index(target=target_segmentation_tensor, prediction=model_output_softmax)
        # self.log('jaccard', train_jaccard, prog_bar=True)
        log_dict = {'loss': loss, 'jaccard': train_jaccard}
        self.log_dict(log_dict, prog_bar=True)
        return log_dict
    
    def validation_step(self, batch, batch_idx):
        input_image_tensor, target_segmentaion_tensor = batch[0], batch[1]
        model_output = self.resnet(input_image_tensor)['out']
       
        val_log_dict = {'validation_loss': self._calc_loss(model_output, target_segmentaion_tensor), 
                'validation_jaccard_index': self._evaluate_jaccard_index(target=target_segmentaion_tensor, prediction=self.softmax(model_output))}
        self.log_dict(val_log_dict, prog_bar=True)
        
    def test_step(self, batch, batch_idx):        
        input_image_tensor, target_segmentaion_tensor = batch[0], batch[1]
        model_output = self.resnet(input_image_tensor)['out']
        test_log_dict = {'test_loss': self._calc_loss(model_output, target_segmentaion_tensor), 'test_jaccard_index': self._evaluate_jaccard_index(target=target_segmentaion_tensor, 
                                                                                                                                                   prediction=self.softmax(model_output))}
        self.log_dict(test_log_dict, prog_bar=True)
            
    def forward(self, input_images):
        return torch.argmax(self.softmax(self.resnet(input_images)['out']), dim=1)
    
    def predict_step(self, batch, batch_idx):
        '''
        Function to run prediction on images.

        Args:
            batch (tuple): batch of input images which are to be segmented.
            batch_idx (int): batch index.
        '''
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer