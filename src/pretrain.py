# PyTorch
import torch                                                                        # pytorchを使用(モデル定義及び学習フロー等)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models

from torchvision.datasets import CIFAR10, CIFAR100
# basic
import numpy as np
import os                                                                           # パス作成とCPUのコア数読み込みに使用
import random
from tqdm import tqdm 
import argparse
import yaml

from model import PretrainedResNet

# pretrainがないと学習できない問題を確認するために，CIFARでpretrainする
def pretrain(model, config, worker_init_fn, g):
    
    # Optimizerをyamlから設定
    def make_optimizer(params, name, **kwargs):
        return optim.__dict__[name](params, **kwargs)
    
    # ImageNetのデータセットを用意する
    def load_cifar100(transform, train:bool):
        cifar100_dataset = CIFAR100(
                            root='./data',
                            train=train,
                            download=True,
                            transform=transform
        )
        return cifar100_dataset

    train_transform = transforms.Compose([transforms.Resize(144),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])

    trainset = load_cifar100(transform=train_transform, train=True)
    train_loader = DataLoader(
        trainset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model.parameters(), **config['optimizer'])
    epochs = 500
    
    # 事前学習開始
    print('Pretrain Session')
    for epoch in range(epochs):
        loop = tqdm(train_loader, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))
        
        model.train()
        for batch in loop:
            x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), os.path.join(log_path, config['model']))

    return model

if __name__ == '__main__':
    seed = 9999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)
    worker_init_fn = worker_init_fn

    def get_args():
        parser = argparse.ArgumentParser(description='YAMLありの例')
        parser.add_argument('config_path', type=str, help='設定ファイル(.yaml)')
        args = parser.parse_args()
        return args
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    log_dir_name = config['model'] + '_' + config['name'] \
                            + '_epochs' + str(config['epochs']) \
                            + '_batch_size' + str(config['batch_size']) \
                            + '_lr' + str(config['optimizer']['lr']) \
                            + '_weight_decay' + str(config['optimizer']['weight_decay']) \
                            + '_margin' + str(config['margin']) \
                            + '_scheduler' + str(config['step_size']) \
                            + '_scale' + str(config['scale'])                         # ログを保存するフォルダ名
    base_log_path = 'pretrain_weight/' + config['base_log_path']
    log_path = os.path.join(base_log_path, log_dir_name)
    os.makedirs(log_path, exist_ok=True)
    model = PretrainedResNet(512, False)
    model = pretrain(model, config, worker_init_fn, g)
    torch.save(model.state_dict(), os.path.join(log_path, config['model']))