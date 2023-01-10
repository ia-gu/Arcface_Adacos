# PyTorch
import torch
import torch.optim as optim
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
# PML
from pytorch_metric_learning import losses
import pytorch_metric_learning.utils.common_functions as tmp


class ArcfaceTrainer:
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
        tmp.NUMPY_RANDOM.seed(seed)

        self.data_loader = DataLoader(data_path=self.config['train_data_path'],
                                      batch_size=self.config['batch_size'],
                                      img_size=img_size,
                                      train_ratio=self.config['train_ratio'],
                                      num_workers=num_workers,
                                      pin_memory=self.config['pin_memory'],)
        if('vgg' in self.config['model']):
            self.model = VGG_based().to(self.device)
        elif('resnet' in self.config['model']):
            self.model = PretrainedResNet(config['pretrained']).to(self.device)

        if os.path.isfile(os.path.join(self.log_path, self.model_name)):
            print('Trained weight file exists')
            self.model.load_state_dict(torch.load(os.path.join(self.log_path, self.model_name)))

        '''
        # when using pretrained model parameter (by pretrain.py), need to change path
        # if os.path.isfile('/home/ueno/Arcface/pretrain_weight/logs/pretrained/resnet_sgd19_epochs200_batch_size128_lr0.01_weight_decay1e-06_margin14.3_scheduler30_scale64/resnet'):
        #     print('load weight file')
        #     self.model.load_state_dict(torch.load('/home/ueno/Arcface/pretrain_weight/logs/pretrained/resnet_sgd19_epochs200_batch_size128_lr0.01_weight_decay1e-06_margin14.3_scheduler30_scale64/resnet'))
        '''

        self.loss = losses.ArcFaceLoss(margin=self.config['margin'],
                                        scale=self.config['scale'],
                                        num_classes=self.config['num_classes'],
                                        embedding_size=self.config['embedding_size']).to(self.device)

        def make_optimizer(params, name, **kwargs):
            return optim.__dict__[name](params, **kwargs)
        self.optimizer = make_optimizer(list(self.model.parameters())+list(self.loss.parameters()),**self.config['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=0.1)
        self.earlystopping = EarlyStopping(20, path=self.log_path+'/best_weight.pth')


    def train(self):
        logging.error('train_acc, train_loss, val_acc, val_loss')
        print('Train with Arcface phase')
        epochs = self.config['epochs']

        with tqdm(range(epochs)) as pbar:
            for epoch in enumerate(pbar):
                i = epoch[0]
                pbar.set_description('[Epoch %d]' % (i+1))
                loss_result = 0.0
                acc = 0.0
                val_loss_result = 0.0
                val_acc = 0.0

                self.model.train()
                j = 1
                for inputs, labels in self.data_loader.dataloaders['train']:
                    pbar.set_description('[Epoch %d (Iteration %d)]' % ((i+1), j))
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.clone().detach()
                    labels = labels.to(self.device, non_blocking=True)
                    outputs = self.model(inputs)
                    loss = self.loss(outputs, labels)

                    # calcurate accuracy from arcface loss
                    # referring from: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/175
                    mask = self.loss.get_target_mask(outputs, labels)
                    cosine = self.loss.get_cosine(outputs)
                    cosine_of_target_classes = cosine[mask == 1]
                    modified_cosine_of_target_classes = self.loss.modify_cosine_of_target_classes(cosine_of_target_classes)
                    diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
                    logits = cosine + (mask*diff)
                    logits = self.loss.scale_logits(logits, outputs)
                    pred = logits.argmax(dim=1, keepdim=True)
                    for img_index in range(len(labels)):
                        if pred[img_index, 0] == labels[img_index]:
                            acc += 1.

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_result += loss.item()
                    j = j + 1

                else:
                    with torch.no_grad():
                        self.model.eval()
                        pbar.set_description('[Epoch %d (Validation)]' % (i+1))
                        for val_inputs, val_labels in self.data_loader.dataloaders['val']:
                            val_inputs = val_inputs.to(self.device, non_blocking=True)
                            val_labels = val_labels.clone().detach()
                            val_labels = val_labels.to(self.device, non_blocking=True)
                            val_outputs = self.model(val_inputs)
                            val_loss = self.loss(val_outputs, val_labels)

                            val_loss_result += val_loss.item()

                            mask = self.loss.get_target_mask(val_outputs, val_labels)
                            cosine = self.loss.get_cosine(val_outputs)
                            cosine_of_target_classes = cosine[mask == 1]
                            modified_cosine_of_target_classes = self.loss.modify_cosine_of_target_classes(cosine_of_target_classes)
                            diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
                            logits = cosine + (mask*diff)
                            logits = self.loss.scale_logits(logits, val_outputs)
                            pred = logits.argmax(dim=1, keepdim=True)
                            for img_index in range(len(val_labels)):
                                if pred[img_index, 0] == val_labels[img_index]:
                                    val_acc += 1.

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
                    
                pbar.set_postfix({'loss':epoch_loss, 'accuracy': epoch_acc, 'val_loss':val_epoch_loss, 'val_accuracy': val_epoch_acc})
                self.earlystopping((val_epoch_loss), self.model)
                if self.earlystopping.early_stop:
                    print('Early Stopping!')
                    break

        torch.save(self.model.state_dict(), os.path.join(self.log_path,self.model_name))
        self.logger.draw_graph()
        self.logger.writer.flush()

        logging.error(self.optimizer)

        test(self.config, self.model, self.log_path, self.worker_init_fn, self.g)
