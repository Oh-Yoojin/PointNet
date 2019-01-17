import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets import get_loader
from pointnet import PointNet_cls
import torch.nn.functional as F


def adjust_learning_rate(learning_rate, optimizer, epoch, decay_epoch):
    lr = learning_rate * (0.1 ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_no)
    print(opt)

    train_loader, num_classes = get_loader(data_path=opt.data_path,
                                           batch_size=opt.batch_size,
                                           num_worker=opt.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pointnet_clf = PointNet_cls(n_class=num_classes)
    pointnet_clf.to(device)

    if opt.load_model != '':
        pointnet_clf.load_state_dict(torch.load(opt.load_model))

    optimizer = optim.Adam(pointnet_clf.parameters(), lr=opt.lr)

    for epoch in range(opt.num_epochs):
        for iter, (points, labels) in enumerate(train_loader):
            points = points.view(opt.batch_size, -1, opt.num_points).to(device)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            pointnet_clf = pointnet_clf.train()
            pred, _, pred_softmax = pointnet_clf(points)
            loss = F.cross_entropy(pred, labels)
            loss.backward()
            optimizer.step()

            print("EPOCH [{}/{}] // ITER [{}/{}] // LOSS [{}]".format(epoch+1, opt.num_epochs, iter+1, len(train_loader), loss.item()))

        if (epoch+1) % opt.decay_epochs == 0:
            adjust_learning_rate(opt.lr, optimizer, epoch, opt.decay_epochs)

        if (epoch+1) % opt.save_per_epochs == 0:
            torch.save(pointnet_clf.state_dict(), "{}ep_model.ckpt".format(epoch+1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='.')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--gpu_no', type=int, default=0)
    parser.add_argument('--save_per_epochs', type=int, default=100)
    parser.add_argument('--load_model', type=str, default='')

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--decay_epochs', type=int, default=30)

    opt = parser.parse_args()
    main(opt)
