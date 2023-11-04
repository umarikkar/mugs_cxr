import numpy as np
from torch.utils.data import Dataset

import pandas as pd
import os
from PIL import Image

import glob

from tqdm import tqdm

class CXR_Pretrain_Dataset_v1(Dataset):

    def __init__(self,
                 datapath,
                 transform=None,
                 seed=0,
                 ):

        super().__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        
        image_list = []
        
        self.data_mimic = os.path.join(datapath, 'mimic_cxr/data_raw/physionet.org/files/mimic-cxr-jpg/2.0.0')
        image_list.append(self.get_mimic_csv())
        
        self.data_nih = os.path.join(datapath, 'nih_cxr8/NIH')
        image_list.append(self.get_nih_csv())
        
        self.data_chex = os.path.join(datapath, 'CheXpert')
        image_list.append(self.get_chex_csv())

        self.transform = transform
        self.all_images = pd.concat(image_list).reset_index().rename(columns={0: 'Path'})
        
        
    def get_mimic_csv(self):
        imgpath = os.path.join(self.data_mimic, 'files')    
        csv = pd.read_csv(os.path.join(self.data_mimic, 'mimic-cxr-2.0.0-metadata.csv'))
        return csv.apply(lambda row: os.path.join(imgpath, "p" + str(row["subject_id"])[:2], 
                        "p" + str(row["subject_id"]), 
                        "s" + str(row["study_id"]), 
                        str(row["dicom_id"]) + ".jpg"), axis=1)
    
    def get_nih_csv(self):
        imgpath = os.path.join(self.data_nih, 'images-224')        
        csv = pd.read_csv(os.path.join(self.data_nih, 'train_val.csv'))
        return csv.apply(lambda row: os.path.join(imgpath, str(row["Image Index"])), axis=1)
    
    def get_chex_csv(self):
        csv = pd.read_csv(os.path.join(self.data_chex, 'CheXpert-v1.0-small/train.csv'))
        return csv.apply(lambda row: os.path.join(self.data_chex, str(row["Path"])), axis=1)


    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        
        img_path = self.all_images['Path'][idx]

        img = Image.open(img_path)
        
        if img.mode != "L":
            img = img.convert("L")
        
        if self.transform is not None:
            img = self.transform(img)
            
        # if img.shape[0]!=1:
        #     print('ha')

        return img, []



class CXR_Pretrain_Dataset_v2(Dataset):

    def __init__(self, datapath='/vol/research/datasets/radiology', 
                        transform=None, channel=3):
        
        datapath2 = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir), 'datasets')
        
        ds_covid_train = glob.glob(os.path.join(datapath2, 'COVID_CXR/train/*g'))
        ds_covid_test = glob.glob(os.path.join(datapath2, 'COVID_CXR/test/*g'))
        print(f'Loaded Covid Dataset: len {len(ds_covid_train) + len(ds_covid_test)}')
        
        ds_spir_train = glob.glob(os.path.join(datapath2, 'SPIR/kaggle/kaggle/train/*.png'))
        ds_spir_test = glob.glob(os.path.join(datapath2, 'SPIR/kaggle/kaggle/test/*.png'))
        print(f'Loaded SPIR Dataset: len {len(ds_spir_train) + len(ds_spir_test)}')

        ds_Monty = glob.glob(os.path.join(datapath2, 'MONTGOMERY/images/images/*g'), recursive=True)
        print(f'Loaded Montgomery Dataset: len {len(ds_Monty)}')
        
        ds_Shenz = glob.glob(os.path.join(datapath2, 'ShenZhen/images/images/*g'), recursive=True)
        print(f'Loaded Shenzhen Dataset: len {len(ds_Shenz)}')
        
        ds_Belarus = glob.glob(os.path.join(datapath2, 'tbcnn-master/belarus/belarus/*g'), recursive=True)
        print(f'Loaded Belarus Dataset: len {len(ds_Belarus)}')

        ds_NIH = glob.glob(os.path.join(datapath, 'nih_cxr8/NIH/images/*.png'), recursive=True)
        print(f'Loaded NIH Dataset: len {len(ds_NIH)}')
        
        csv = pd.read_csv(os.path.join(datapath, 'CheXpert/CheXpert-v1.0-small/train.csv'))
        csv = csv[csv['Frontal/Lateral']=='Frontal']
        ds_chexpert_train = csv.apply(lambda row: os.path.join(datapath, 'CheXpert', str(row["Path"])), axis=1).to_list()
        
        csv = pd.read_csv(os.path.join(datapath, 'CheXpert/CheXpert-v1.0-small/valid.csv'))
        csv = csv[csv['Frontal/Lateral']=='Frontal']
        ds_chexpert_test = csv.apply(lambda row: os.path.join(datapath, 'CheXpert', str(row["Path"])), axis=1).to_list()

        print(f'Loaded CHEXPERT Dataset: len {len(ds_chexpert_train) + len(ds_chexpert_test)}')

        
        self.all_imgs_path = ds_covid_train + ds_covid_test + ds_spir_train + ds_spir_test + ds_chexpert_train + ds_chexpert_test + ds_NIH + ds_Monty + ds_Shenz + ds_Belarus

        self.label = -1
        self.channel = channel

        print(f"Image counts: Mode:PRETRAIN disease:NONE")
        print(len(self.all_imgs_path))
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs_path)

    def __getitem__(self, index):
        
        path = self.all_imgs_path[index]
        if self.channel == 1:
            img = Image.open(path).convert('L')
        elif self.channel == 3:
            img = Image.open(path).convert('RGB')    

        if self.transform:
            img, weak_flag = self.transform(img)
            
        return img,  weak_flag