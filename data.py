import os
import numpy as np
import pandas as pd
import glob
import argparse

import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from src.utils.utils import make_dir

SRC = './dataset'
IMAGE_SRC = './dataset/102flowers/jpg'
SEGMENTATION_SRC_REMOVED = './dataset/102segmentations/segmim-removed/'
PREPROCESSED_SEGMENTATION1 = './dataset/102segmentations/preprocessed-thresh-thresh'
PREPROCESSED_SEGMENTATION2 = './dataset/102segmentations/preprocessed-thresh-thresh-close' 

class Channelize:
    def __init__(self) -> None:
        pass
    
    def __call__(self, segmentation):
        assert segmentation.shape[0]==1, f'Segmentation must be single channeled, {segmentation.shape[0]} found.'
        ones = segmentation == 1
        zeros = segmentation == 0
        segmentation = torch.concatenate((segmentation, torch.zeros_like(segmentation)), dim=0)
        segmentation[0] = 0
        segmentation[1, ones[0, :, :]] = 1.0
        segmentation[0, zeros[0, :, :]] = 1.0
        return segmentation
    
class ImageData(torch.utils.data.Dataset):
    def __init__(self, image_src, segmentation_src, height=512, width=600) -> None:
        '''
        Dataset initialization.

        Args:
            image_src (str): image source directory.
            segmentation_src (str): segmentation source directory.
        '''
        
        # super(ImageData, self).__init__()
        self.image_src = image_src
        self.segmentation_src = segmentation_src
        self._populate()
        
        self.height = height
        self.width = width
        self.transforms = transforms.Compose([
            transforms.Resize((self.height, self.width)),
        ])
        
        self.channelize = Channelize()
    
    def _set_size(self, height, width):
        self.height = height
        self.width = width
        self.transforms = transforms.Compose([
            transforms.Resize((self.height, self.width)),
        ])
    
    def __len__(self):
        '''
        Checks if total segmentations and total images are same and returns that number.

        Returns:
            int: length of dataset
        '''
        
        assert len(os.listdir(self.image_src))==len(os.listdir(self.segmentation_src)), 'total segmentations and total images donot match'
        return len(os.listdir(self.image_src))
    
    def _populate(self):
        '''
        Funtion creates a numpy array of image names and having shape (2, total images).
        '''
        
        self.image_segmentation = np.array([sorted(glob.glob(self.image_src+'/*')), sorted(glob.glob(self.segmentation_src+'/*'))])
    
    def __getitem__(self, idx):
        '''
        Single item from the original dataset directory.

        Returns:
            list: dataset image, ground truth segmentation.
        '''
        # return single item from the original dataset directory
        image_path = self.image_segmentation[0, idx]
        segmentation_path = self.image_segmentation[1, idx]

        image = self.transforms(torchvision.io.read_image(image_path))/255.0
        segmentation = self.transforms(torchvision.io.read_image(segmentation_path)).to(dtype=torch.float32)
        segmentation = self.channelize(segmentation)
                
        return image, segmentation, image_path, segmentation_path
    
    
class ImageDataLoaders(pl.LightningDataModule):
    def __init__(self, dataset) -> None:
        '''
        Image data loader converts the image dataset to pytorch Dataloader to generate training, validation and testing batchs.
        '''
        super(ImageDataLoaders, self).__init__()
        self.dataset = dataset
        self.total_indices = np.arange(len(self.dataset))
        np.random.shuffle(self.total_indices)
    
    def __len__(self):
        '''
        Length of dataset.

        Returns:
            int: length
        '''
        return len(self.dataset)
    
    def set_size(self, height, width):
        '''
        Setting image size. 

        Args:
            height (int): height of image.
            width (int): width of image.
        '''
        self.dataset._set_size(height, width)
    
    def prepare_data(self, test_split=0.2, val=True, val_split=0.2):
        '''
        Prepares training, validation and testing dataset.

        Args:
            dataset (ImageData): Image dataset generator
            test_split (float, optional): Defaults to 0.2.
            val (bool, optional): Defaults to True.
            val_split (float, optional): Defaults to 0.2.
        '''
        
        self.test_size = int(len(self.dataset)*test_split)
        self.train_size = len(self.dataset)-self.test_size
        self.val = True
        if val:
            self.val_size = int(self.train_size*val_split)
            self.train_size = self.train_size - self.val_size

        training_indices = self.total_indices[:self.train_size]
        validation_indices = None
        self.validation_dataframe = None
        if val:
            validation_indices = self.total_indices[self.train_size:self.train_size+self.val_size]
            testing_indices = self.total_indices[self.train_size+self.val_size:]
        else:
            testing_indices = self.total_indices[self.train_size:]
        
        self.training_dataframe = pd.DataFrame({'training_indices': training_indices, 'training_image_paths': self.dataset.image_segmentation[0, training_indices], 
                                                'training_segmentation_paths': self.dataset.image_segmentation[1, training_indices]})
        
        self.testing_dataframe = pd.DataFrame({'testing_indices': testing_indices, 'testing_image_paths': self.dataset.image_segmentation[0, testing_indices], 
                                                'testing_segmentation_paths': self.dataset.image_segmentation[1, testing_indices]})
        if val:
            assert validation_indices is not None, 'validation indices cannot be None'
            self.validation_dataframe = pd.DataFrame({'validation_indices': validation_indices, 'validation_image_paths': self.dataset.image_segmentation[0, validation_indices], 
                                                    'validation_segmentation_paths': self.dataset.image_segmentation[1, validation_indices]})       
        
        if val:
            assert not (np.array_equal(self.training_dataframe['training_indices'].values, self.testing_dataframe['testing_indices'].values) and
                    np.array_equal(self.training_dataframe['training_indices'].values, self.validation_dataframe['validation_indices'].values) and 
                    np.array_equal(self.testing_dataframe['testing_indices'].values, self.validation_dataframe['validation_indices'].values)), "Data leakage found"
        else:
            assert not np.array_equal(self.training_dataframe['training_indices'].values, self.testing_dataframe['testing_indices'].values), "Data leakage found"            
            
    
    def save_dataframe(self, training_source, testing_source, validation_source=None):
        self.training_dataframe.to_csv(training_source, index=False)
        self.testing_dataframe.to_csv(testing_source, index=False)
        if self.validation_dataframe is not None: 
            self.validation_dataframe.to_csv(validation_source, index=False)
    
    def load_dataframes(self, training_dataframe_path='./dataset/split1/training_dataframe.csv', testing_dataframe_path='./dataset/split1/testing_dataframe.csv',
                        validation_dataframe_path='./dataset/split1/validation_dataframe.csv'):
        self.training_dataframe = pd.read_csv(training_dataframe_path)
        self.testing_dataframe = pd.read_csv(testing_dataframe_path)
        self.validation_dataframe = pd.read_csv(validation_dataframe_path)
    
    def train_dataloader(self, batch_size=2, **kwargs):
        train_sampler = SubsetRandomSampler(self.training_dataframe['training_indices'].values)
        return torch.utils.data.DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, **kwargs)
    
    def test_dataloader(self, **kwargs):
        test_sampler = SubsetRandomSampler(self.testing_dataframe['testing_indices'].values)
        return torch.utils.data.DataLoader(self.dataset, sampler=test_sampler, **kwargs)
        
    def val_dataloader(self, **kwargs):
        assert self.validation_dataframe is not None, 'Validation set not created.'
        validation_sampler = SubsetRandomSampler(self.validation_dataframe['validation_indices'].values)
        return torch.utils.data.DataLoader(self.dataset, sampler=validation_sampler, **kwargs)
        
    def predict_dataloader(self, data=None, from_test=True, **kwargs):
        if from_test:
            return self.test_dataloader(**kwargs)
        else:
            # use data
            pass            
        # return super().predict_dataloader()

    
if __name__=='__main__':
    
    args = argparse.ArgumentParser()
    
    args.add_argument('--img', default=IMAGE_SRC, help="Training image source")   
    args.add_argument('--seg', default=PREPROCESSED_SEGMENTATION2, help="Training segmentation source")   
    args.add_argument('--dataframes_src', default=SRC, help="Dataframes source folder")
    args.add_argument('--dataframe_folder_name', required=True, help="Dataframe folder name")
    
    parsed_args = args.parse_args()
    imagedata = ImageData(parsed_args.img, parsed_args.seg)
    imagedataloaders = ImageDataLoaders(imagedata)
    imagedataloaders.prepare_data()
    
    dataframe_folder_name = os.path.join(parsed_args.dataframes_src, parsed_args.dataframe_folder_name)
    make_dir(dataframe_folder_name)
    print(f'Creating folder {dataframe_folder_name} for dataframes')
        
    imagedataloaders.save_dataframe(os.path.join(dataframe_folder_name, 'training_dataframe.csv'),
                                    os.path.join(dataframe_folder_name, 'testing_dataframe.csv'),            
                                    os.path.join(dataframe_folder_name, 'validation_dataframe.csv'))            
    
    print(f'CSVs {os.listdir(dataframe_folder_name)} created')