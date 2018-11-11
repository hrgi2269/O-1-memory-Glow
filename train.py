import os
from os.path import join, exists
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import datasets, transforms
from model import Glow

# Parse args
print('==> Args')
parser = argparse.ArgumentParser()
parser.add_argument('--datasets_dir', default='./', type=str,
                    help='Directory of datasets')
parser.add_argument('--out_dir', default='./out', type=str,
                    help='Directory to put the training result')
parser.add_argument('--channels_h', default=512, type=int,
                    help='Number of channels of hidden layers of conv-nets')
parser.add_argument('--K', default=32, type=int,
                    help='Depth of flow')
parser.add_argument('--L', default=3, type=int,
                    help='Number of levels')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--weight_decay', default=1e-6, type=float,
                    help='Weight decay')
parser.add_argument('--batch_size', default=512, type=int,
                    help='Mini-batch size')
parser.add_argument('--epochs', default=10**3, type=int,
                    help='Number of epochs to train totally')
parser.add_argument('--save_memory', action='store_true',
                    help='Enables memory-saving backpropagation')
parser.add_argument('--display_interval', default=1, type=int,
                    help='Steps between logging training details')
parser.add_argument('--sample_interval', default=1, type=int,
                    help='Epochs between sampling')
parser.add_argument('--temperature', default=0.7, type=float,
                    help='Temperature of distribution to sample from')
parser.add_argument('--save_model_interval', default=5, type=int,
                    help='Epochs between saving model')
args = parser.parse_args()
print(vars(args))

# Device
print('==> Device')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
print(device)

# Dataset
print('==> Dataset')
transform = transforms.ToTensor()
train_dataset = datasets.CIFAR10(args.datasets_dir, train=True,
                                 transform=transform, download=True)
test_dataset = datasets.CIFAR10(args.datasets_dir, train=False,
                                transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)
image_size = train_dataset[0][0].size()
print('size of train data: %d' % len(train_dataset))
print('size of test data: %d' % len(test_dataset))
print('image size: %s' % str(image_size))

# Model
print('==> Model')
model = Glow(image_size, args.channels_h, args.K, args.L,
             save_memory=args.save_memory).to(device)
#print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(epoch):
    # warmup
    lr = min(args.lr * epoch / 10, args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()
    sum_loss = 0
    count = 0
    for iteration, batch in enumerate(train_loader, 1):
        batch = batch[0].to(device)
        optimizer.zero_grad()
        loss = -model.log_prob(batch)
        mean_loss = loss.mean()
        mean_loss.backward()
        optimizer.step()

        sum_loss += loss.sum().item()
        if iteration % args.display_interval == 0:
            print('[%6d][%6d] | loss: %.4f' % \
                  (epoch, iteration, mean_loss.item()))
    average_loss = sum_loss / len(train_dataset)
    print('==> Epoch%d Average Loss | loss: %.4f' % \
          (epoch, average_loss))
    return average_loss

def test(epoch):
    model.eval()
    sum_loss = 0
    for iteration, batch in enumerate(test_loader, 1):
        batch = batch[0].to(device)
        with torch.no_grad():
            loss = -model.log_prob(batch)
        sum_loss += loss.sum().item()

    average_loss = sum_loss / len(test_dataset)
    print('==> Epoch%d Test Loss | loss: %.4f' % \
          (epoch, average_loss))
    if epoch % args.sample_interval == 0:
        n_samples = 64
        with torch.no_grad():
            sample = model.sample(n_samples, device).detach().cpu()
        save_image(sample, join(args.out_dir, 'sample_%06d.png' % epoch), nrow=8)
    return average_loss

def dump(train_loss, test_loss):
    with open(join(args.out_dir, 'dump.csv'), mode='a') as f:
        f.write('%.4f, %.4f\n' % (train_loss, test_loss))

if __name__ == '__main__':
    if not exists(args.out_dir):
        os.mkdir(args.out_dir)
    print('==> Start learning')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        dump(train_loss, test_loss)
        if epoch % args.save_model_interval == 0:
            params = model.state_dict()
            torch.save(params, join(args.out_dir, 'model_%06d' % epoch))
