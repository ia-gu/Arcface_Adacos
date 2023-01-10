# hand-made module
from test_data_loader import TestDataLoader
from model import PretrainedResNet, VGG_based
# pyTorch
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# other
from scipy import spatial
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import argparse
import yaml
import os

def test(config, model, log_path):
    print('Test phase')

    test_transform = transforms.Compose([transforms.Resize((config['width'],config['height'])),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    testSet = TestDataLoader(config['test_data_path'], transform=test_transform)
    testLoader = DataLoader(testSet, batch_size=1, num_workers=1, pin_memory=config['pin_memory'])

    device = torch.device('cuda')
    correct = 0
    error = 0
    Outs = []
    Labels = None

    for _, (test1, test2, label) in enumerate(testLoader):
        model = model.to(device)

        model.eval()
        test1 = test1.to(device, non_blocking=True)
        test2 = test2.to(device, non_blocking=True)

        # face recognition(same person or not)
        out1, out2 = model.test(test1, test2)
        cos = 1 - spatial.distance.cosine(out1[0].data.cpu().numpy(), out2[0].data.cpu().numpy())
        if((cos>0.0 and label[0]==1) or (cos<=0.0 and label[0]==0)):
            correct += 1
        else:
            error += 1

        if(Labels == None):
            Labels = label
            Outs.append(cos)
        else:
            Labels = torch.cat((Labels, label), 0)
            Outs.append(cos)
      
    print('*'*70)
    print(f'Test result   correct:{correct}   error:{error}   accuracy:{correct/(correct+error):.3f}')
    logging.error(f'Test result   correct:{correct}   error:{error}   accuracy:{correct/(correct+error):.3f}')
    logging.error('')
    print('*'*70)

    # ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(Labels, Outs, drop_intermediate=False)
    fig = plt.figure()
    plt.plot(fpr, tpr, marker='o', markersize=1)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    fig.savefig(log_path+'/roc_curve.png')

if __name__ == '__main__':
    seed = 9999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # get config
    def get_args():
        parser = argparse.ArgumentParser(description='YAML')
        parser.add_argument('config_path', type=str, help='.yaml')
        args = parser.parse_args()
        return args
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda')

    if('vgg' in config['model']):
            model = VGG_based().to(device)
    elif('resnet' in config['model']):
        model = PretrainedResNet(config['pretrained']).to(device)

    model = model.to(device)
    log_dir_name = config['loss'] + '/' + config['model'] + '_' + config['name'] \
                            + '_epochs' + str(config['epochs']) + '_batch_size' + str(config['batch_size']) \
                            + '_lr' + str(config['optimizer']['lr']) + '_weight_decay' + str(config['optimizer']['weight_decay']) \
                            + '_margin' + str(config['margin']) + '_scheduler' + str(config['step_size']) \
                            + '_scale' + str(config['scale'])
    log_path = os.path.join(config['base_log_path'], log_dir_name)

    params = torch.load(os.path.join(log_path,'best_weight.pth'), map_location=device)
    model.load_state_dict(params)
    logging.basicConfig(level=logging.ERROR, filename=log_path+'/result.txt', format='%(message)s')

    test(config, model, log_path)