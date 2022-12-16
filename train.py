import os
import numpy as np
import pandas as pd 
import gc
import argparse

from data import ImageData, ImageDataLoaders

import pytorch_lightning as pl
import torch
import mlflow

from model import load_base, ResNetModel


torch.hub.set_dir('./')


SRC = './dataset'
IMAGE_SRC = './dataset/102flowers/jpg'
SEGMENTATION_SRC_REMOVED = './dataset/102segmentations/segmim-removed/'
PREPROCESSED_SEGMENTATION1 = './dataset/102segmentations/preprocessed-thresh-thresh'
PREPROCESSED_SEGMENTATION2 = './dataset/102segmentations/preprocessed-thresh-thresh-close'

TRAINING_DATAFRAME = './dataset/split1/training_dataframe.csv'
VALIDATION_DATAFRAME = './dataset/split1/validation_dataframe.csv'
TESTING_DATAFRAME = './dataset/split1/testing_dataframe.csv'

# height, width
shapes_array = [(64, 75), (128, 150), (256, 300)]

if __name__=='__main__':
    
    args = argparse.ArgumentParser()   
    args.add_argument('--img', default=IMAGE_SRC, help="Training image source")
    args.add_argument('--seg', default=PREPROCESSED_SEGMENTATION2, help="Training segmentation source")
    args.add_argument('--training_dataframe', default=TRAINING_DATAFRAME)
    args.add_argument('--testing_dataframe', default=TESTING_DATAFRAME)
    args.add_argument('--validation_dataframe', default=VALIDATION_DATAFRAME)
    args.add_argument('--run_name', default='trial-run')
    args.add_argument('--height', default=64, type=int)
    args.add_argument('--width', default=75, type=int)
    args.add_argument('--epochs', default=5, type=int)
    args.add_argument('--lr', default=0.01)
    args.add_argument('--num_workers', default=8, type=int)
    args.add_argument('--tensorboard_logger_dir', default='.')
    args.add_argument('--batch_size', default=2, type=int)
    args.add_argument('--single_run', default=False, type=bool)
    parsed_args = args.parse_args()
    
    if parsed_args.single_run:
        imagedata = ImageData(parsed_args.img, parsed_args.seg, height=parsed_args.height, width=parsed_args.width)
        imagedataloaders = ImageDataLoaders(imagedata)
        imagedataloaders.load_dataframes(training_dataframe_path=parsed_args.training_dataframe,
                                        testing_dataframe_path=parsed_args.testing_dataframe,
                                        validation_dataframe_path=parsed_args.validation_dataframe)
        
        train_loader = imagedataloaders.train_dataloader(num_workers=parsed_args.num_workers, batch_size=parsed_args.batch_size)
        test_loader = imagedataloaders.test_dataloader(num_workers=parsed_args.num_workers)
        val_loader = imagedataloaders.val_dataloader(num_workers=parsed_args.num_workers)

        print(f'dataset size {len(imagedata)}, training size {len(train_loader)}, test size {len(test_loader)}, validation size {len(val_loader)}')
        
        torch.cuda.empty_cache()
        gc.collect()
        
        optimizer = torch.optim.Adam
        base = load_base(pretrained=True)
        model = ResNetModel(resnet=base, optimizer=optimizer, learning_rate=parsed_args.lr).to(device='cuda')

        
        mlflow.set_tag('mlflow.runName', parsed_args.run_name)
        mlflow.pytorch.autolog(log_every_n_epoch=parsed_args.epochs, log_models=True)
        tblogger = pl.loggers.TensorBoardLogger(parsed_args.tensorboard_logger_dir, version = parsed_args.run_name)
        trainer = pl.Trainer(max_epochs=parsed_args.epochs, accelerator='gpu', logger=tblogger)
        trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
        
        trainer.test(model=model, dataloaders=test_loader)
    
    else:
        mlflow.set_tag('mlflow.runName', parsed_args.run_name)
        mlflow.pytorch.autolog(log_every_n_epoch=parsed_args.epochs, log_models=True)
        for index, shape in enumerate(shapes_array):
            
            print(f"############## TRAINING FOR IMAGE SHAPE {shape} {index} ##############")
            
            imagedata = ImageData(parsed_args.img, parsed_args.seg, height=shape[0], width=shape[1])
            imagedataloaders = ImageDataLoaders(imagedata)
            imagedataloaders.load_dataframes(training_dataframe_path=parsed_args.training_dataframe,
                                            testing_dataframe_path=parsed_args.testing_dataframe,
                                            validation_dataframe_path=parsed_args.validation_dataframe)
            
            train_loader = imagedataloaders.train_dataloader(num_workers=parsed_args.num_workers, batch_size=parsed_args.batch_size)
            test_loader = imagedataloaders.test_dataloader(num_workers=parsed_args.num_workers)
            val_loader = imagedataloaders.val_dataloader(num_workers=parsed_args.num_workers)

            print(f'dataset size {len(imagedata)}, training size {len(train_loader)}, test size {len(test_loader)}, validation size {len(val_loader)}')
            
            torch.cuda.empty_cache()
            gc.collect()
            
            optimizer = torch.optim.Adam
            base = None
            
            if index==0:
                base = load_base(pretrained=True)
            else:
                base = model.resnet
                
            model = ResNetModel(resnet=base, optimizer=optimizer, learning_rate=parsed_args.lr).to(device='cuda')
        
            tblogger = pl.loggers.TensorBoardLogger(parsed_args.tensorboard_logger_dir, version = f'{parsed_args.run_name}_{shape}')
            trainer = pl.Trainer(max_epochs=parsed_args.epochs, accelerator='gpu', logger=tblogger)
            trainer.fit(model=model,
                        train_dataloaders=train_loader,
                        val_dataloaders=val_loader)
            
            
            trainer.test(model=model, dataloaders=test_loader)
    