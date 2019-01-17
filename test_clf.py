import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets import get_loader
from pointnet import PointNet_cls
import torch.nn.functional as F
import h5py
from sklearn.metrics import accuracy_score


def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_no)
    print(opt)

    test_loader, num_classes = get_loader(data_path=opt.data_path,
                                           batch_size=opt.batch_size,
                                           num_worker=opt.num_workers,
                                           train_=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pointnet_clf = PointNet_cls(n_class=num_classes).to(device)

    load_model = os.path.join(opt.data_path, 'pointnet_pytorch', '{}ep_model.ckpt'.format(opt.load_epoch))
    state_dict = torch.load(load_model)
    pointnet_clf.load_state_dict(state_dict)

    result_mat = np.array([])
    pointnet_clf.eval()
    for i, (point, label) in enumerate(test_loader):
        point = point.view(opt.batch_size, -1, opt.num_points).to(device)

        raw, _, output = pointnet_clf(point)
        result_mat = np.append(result_mat, output.data.cpu().numpy())


    result_mat = result_mat.reshape(-1, 4)[:-2]
    test_f = h5py.File(os.path.join(opt.data_path ,'ply_data_test.h5'), 'r')
    labels_true = list(test_f['label'])[:-2]
    labels_pred = np.argmax(result_mat, axis=1)

    print('Accuracy :', accuracy_score(np.array(labels_true), labels_pred))

    if opt.save_tf:
        np.save(os.path.join(opt.save_path, 'result_softmax.npy'), result_mat)
        np.save(os.path.join(opt.save_path, 'result_pred.npy'), labels_pred)
        np.save(os.path.join(opt.save_path, 'result_ture.npy'), labels_true)

        result_df = pd.DataFrame(result_mat)
        result_df.columns = ['softmax_dim{}'.format(i) for i in range(1,5)]
        result_df['True_labels'] = labels_true
        result_df['Pred_labels'] = labels_pred

        result_df.to_csv(os.path.join(opt.save_path, 'result_df.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--gpu_no', type=int, default=0)
    parser.add_argument('--load_epoch', type=int, default=80)
    parser.add_argument('--save_tf', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='.')

    opt = parser.parse_args()
    main(opt)
