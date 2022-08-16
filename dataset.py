import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import numpy as np
import os

def regular_data(info_file, root_dir):
    ages = []
    face_frame = []
    with open(info_file, 'r') as fin:
        for line in fin:
            split_content = line.split()
            face_frame.append(split_content[0])
            ages.append(float(split_content[1]))

    return ages, face_frame


class AgeEstimationDataset(Dataset):
    """Face dataset."""

    def __init__(self, info_file, root_dir, f_getData, transform=None):
        self.ages, self.face_frame = f_getData(info_file, root_dir)

        print('info file:', info_file)
        print('len of img', len(self.face_frame))
        self.ages = torch.Tensor(self.ages)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.face_frame[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')

        age = self.ages[idx]
        image224=self.transform[0](image)

        return image224, age

def load_data(train_bath_size, list_root, pic_root_dir, RANDOM_SEED, val_batch_size=10):
    transform_train_224=transforms.Compose([transforms.Resize([256,256]),
                                      transforms.RandomCrop([224,224]),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                     ])
    transform_val_224 = transforms.Compose([transforms.Resize([256,256]),
                                      transforms.CenterCrop([224,224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
     ])

    transformed_train_dataset = AgeEstimationDataset(info_file=list_root + '_train.txt',
                                                     root_dir=pic_root_dir,
                                                     f_getData=regular_data,
                                                     transform=[transform_train_224, None])


    transformed_valid_dataset = AgeEstimationDataset(info_file=list_root + '_val.txt',
                                                     root_dir=pic_root_dir,
                                                     f_getData=regular_data,
                                                     transform=[transform_val_224,None])

    # Loading dataset into dataloader
    train_loader = DataLoader(transformed_train_dataset, batch_size=train_bath_size,
                              shuffle=True, num_workers=4, worker_init_fn=np.random.seed(RANDOM_SEED))

    val_loader = DataLoader(transformed_valid_dataset, batch_size=val_batch_size,
                            shuffle=False, num_workers=4, worker_init_fn=np.random.seed(RANDOM_SEED))

    return train_loader, val_loader