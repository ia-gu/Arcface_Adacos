# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch import topk
# basic
import numpy as np
import os
import logging
import random
from tqdm import tqdm
# hand made
from model import VGG_based, PretrainedResNet
from logger import Logger
from data_loader import DataLoader
from test import test
from earlystopping import EarlyStopping
from adacos import AdaCos

class AdaCosTrainer:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config['model']
        log_dir_name = self.config['loss'] + '/' + self.model_name + '_' + self.config['name'] \
                            + '_epochs' + str(self.config['epochs']) + '_batch_size' + str(self.config['batch_size']) \
                            + '_lr' + str(self.config['optimizer']['lr']) + '_weight_decay' + str(self.config['optimizer']['weight_decay']) \
                            + '_margin' + str(self.config['margin']) + '_scheduler' + str(self.config['step_size']) \
                            + '_scale' + str(self.config['scale'])
        self.log_path = os.path.join(self.config['base_log_path'], log_dir_name)
        img_size = (self.config['width'],self.config['height'])
        self.logger = Logger(self.log_path)
        num_class = self.config['num_classes']
        step_size = self.config['step_size']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_workers = 0
        if self.config['num_workers'] == -1:
            print('set num_workers to number of cpu cores :', os.cpu_count())
            num_workers = os.cpu_count()
        else:
            num_workers = self.config['num_workers']

        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.data_loader = DataLoader(data_path=self.config['train_data_path'],
                                      batch_size=self.config['batch_size'],
                                      img_size=img_size,
                                      train_ratio=self.config['train_ratio'],
                                      num_workers=num_workers,
                                      pin_memory=self.config['pin_memory'])
        if('vgg' in self.config['model']):
            self.model = VGG_based(num_class).to(self.device)
        elif('resnet' in self.config['model']):
            self.model = PretrainedResNet(num_class, config['pretrained']).to(self.device)
       
        if os.path.isfile(os.path.join(self.log_path, self.model_name)):
            print('Trained weight file exists')
            self.model.load_state_dict(torch.load(os.path.join(self.log_path, self.model_name)))

        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.metric_fn = AdaCos(self.config['embedding_size'], num_class, m=self.config['margin']).to(self.device)

        def make_optimizer(params, name, **kwargs):
            return optim.__dict__[name](params, **kwargs)
        self.optimizer = make_optimizer(list(self.model.parameters())+list(self.loss.parameters()),**self.config['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=0.1)
        
        self.earlystopping = EarlyStopping(20, path=self.log_path+'/best_weight.pth')

    def accuracy(self, output, target):
        with torch.no_grad():
            maxk = 1
            _, pred = topk(output, maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
            return correct_k

    def train(self):
        print('Train with AdaCos phase')
        logging.error('train_acc, train_loss, val_acc, val_loss')
        epochs = self.config['epochs']

        with tqdm(range(epochs)) as pbar:
            for epoch in enumerate(pbar):
                i = epoch[0]
                pbar.set_description('[Epoch %d]' % (i+1))
                loss_result = 0.0
                acc = 0.0
                val_loss_result = 0.0
                val_acc = 0.0
                j = 1

                for inputs, labels in self.data_loader.dataloaders['train']:
                    self.model.train()
                    self.metric_fn.train()
                    pbar.set_description('[Epoch %d (Iteration %d)]' % ((i+1), j))
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    feature = self.model(inputs)
                    outputs = self.metric_fn(feature, labels, True)
                    loss = self.loss(outputs, labels)
                    acc1, = self.accuracy(output=outputs, target=labels)
                    acc += acc1.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_result += loss.item()
                    j = j + 1

                else:
                    with torch.no_grad():
                        self.model.eval()
                        self.metric_fn.eval()

                        pbar.set_description('[Epoch %d (Validation)]' % (i+1))
                        for val_inputs, val_labels in self.data_loader.dataloaders['val']:
                            val_inputs = val_inputs.to(self.device, non_blocking=True)
                            val_labels = val_labels.to(self.device, non_blocking=True)
                            val_features = self.model(val_inputs)
                            val_outputs = self.metric_fn(val_features, val_labels, False)
                            val_loss = self.loss(val_outputs, val_labels)
                            val_acc1, = self.accuracy(output=val_outputs, target=val_labels)
                            val_acc += val_acc1.item()
                            val_loss_result += val_loss.item()

                    epoch_loss = loss_result / len(self.data_loader.dataloaders['train'].dataset)
                    epoch_acc = acc / len(self.data_loader.dataloaders['train'].dataset)
                    val_epoch_loss = val_loss_result / len(self.data_loader.dataloaders['val'].dataset)
                    val_epoch_acc = val_acc / len(self.data_loader.dataloaders['val'].dataset)
                    self.logger.collect_history(loss=epoch_loss, accuracy=epoch_acc, val_loss=val_epoch_loss, val_accuracy=val_epoch_acc)
                    self.logger.writer.add_scalars('losses', {'train':epoch_loss,'validation':val_epoch_loss}, (i+1))
                    self.logger.writer.add_scalars('accuracies', {'train':epoch_acc, 'validation':val_epoch_acc}, (i+1))
                    self.logger.writer.add_scalars('learning_rate', {'learning_rate':self.optimizer.param_groups[0]['lr']}, (i+1))
                    logging.error(f'{epoch_acc:.3f}, {epoch_loss:.3f}, {val_epoch_acc:.3f}, {val_epoch_loss:.3f}')
                    self.scheduler.step()

                self.earlystopping((val_epoch_loss), self.model)
                if self.earlystopping.early_stop:
                    print('Early Stopping!')
                    break

                pbar.set_postfix({'loss': epoch_loss, 'accuracy': epoch_acc, 'val_loss': val_epoch_loss, 'val_accuracy': val_epoch_acc})
        torch.save(self.model.state_dict(), os.path.join(self.log_path, self.model_name))
        self.logger.draw_graph()
        self.logger.writer.flush()

        logging.error(self.optimizer)

        test(self.config, self.model, self.log_path, self.worker_init_fn, self.g)
