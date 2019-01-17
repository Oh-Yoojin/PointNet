import os
import numpy as np
from torch.utils.data import DataLoader
import h5py


class DataLoader_20_clf(DataLoader):
    def __init__(self, data_path, f_name='ply_data_train.h5'):
        self.train_path = os.path.join(data_path, f_name)

        train_f = h5py.File(self.train_path, 'r')
        self.image_ = list(train_f['data'])
        self.label_ = list(train_f['label'])
        self.n_label_ = len(set(self.label_))

    def __len__(self):
        return len(self.image_)

    def __getitem__(self, idx):
        image_f, label_f = self.image_[idx], self.label_[idx]
        #label_f = self.build_onehot(label_f, self.n_label_)
        return (image_f, label_f)

    def build_onehot(self, num, n_labels):
        onehot = np.zeros((n_labels))
        onehot[num] = 1
        return onehot


def get_loader(data_path, batch_size=32, num_worker=4, train_=True):
    if train_:
        dataloader_20 = DataLoader_20_clf(data_path)
        data_loader = DataLoader(dataset=dataloader_20, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    else:
        dataloader_20 = DataLoader_20_clf(data_path, f_name='ply_data_test.h5')
        data_loader = DataLoader(dataset=dataloader_20, batch_size=1, shuffle=False, num_workers=num_worker)
    return data_loader, dataloader_20.n_label_
