# -*- coding: utf-8 -*-
import os
import cv2
import PIL
from PIL import Image
import numpy as np
import pandas as pd
from .utils import cs_images_folder, meta_smiles_path, read_image
from .utils import load_l5, prune_labels, generate_folds
from .drugdataset import DrugDataset

from torchvision import transforms

def get_cs_images(n_fold=3, **kwargs):
    dataset = load_l5()
    labels = dataset["label_df"]
    labels = prune_labels(labels, 11, 1)
    
    cs_meta = pd.read_csv(meta_smiles_path, index_col=0, dtype=str)
    cs_meta = cs_meta.loc[labels.index.values]
    train_fold, test_fold, labels = generate_folds(labels.index.values.tolist(), labels, n_fold)
    column_names = list(labels.columns.values)
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
#                               transforms.ColorJitter(hue=.05, saturation=.05),
#                               transforms.RandomCrop((224, 224)),
#         transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        transforms.ToTensor()
    ])
    
    test_transforms = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor()
                            ])
    
    for fold_i in range(n_fold):
        train_dataset = []
        test_dataset = []
        train_ids = train_fold[fold_i]
        test_ids = test_fold[fold_i]
    
        for _id in train_ids:
            pubchem_id = cs_meta.loc[_id, 'pubchem_cid']
            cs_img_path = os.path.join(cs_images_folder, pubchem_id+'.png')
            x = read_image(cs_img_path)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = Image.fromarray(x)
            y = labels.loc[_id].values.astype(np.float32)
            train_dataset.append((x, y))
    
        for _id in test_ids:
            pubchem_id = cs_meta.loc[_id, 'pubchem_cid']
            cs_img_path = os.path.join(cs_images_folder, pubchem_id+'.png')
            x = read_image(cs_img_path)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = Image.fromarray(x)
            y = labels.loc[_id].values.astype(np.float32)
            test_dataset.append((x, y))
    
        output_size = labels.shape[1]
        train = DrugDataset(train_dataset, output_size, train_ids, None, column_names, transform=train_transforms)
        test = DrugDataset(test_dataset, output_size, test_ids, None, column_names, transform=test_transforms)
        yield train, test