# PyTorch
import torch                                                                        # pytorchを使用(モデル定義及び学習フロー等)
import torch.nn as nn
import torch.optim as optim
from torch import topk
# basic
import numpy as np
import os                                                                           # パス作成とCPUのコア数読み込みに使用
import logging
import random
from tqdm import tqdm                                                               # 学習の進捗表示に使用
# hand made
from model import VGG_based, PretrainedResNet                         # 自作(モデル定義に使用)
from logger import Logger                                                           # 自作(ログ保存に使用)
from data_loader import DataLoader                                                  # 自作(データ読み込みに使用)
from test import test                                                               # 自作(モデルのテストに使用)
from earlystopping import EarlyStopping
from adacos import AdaCos
# PML
import pytorch_metric_learning
from pytorch_metric_learning import losses                                          # Arcfaceに使用
import pytorch_metric_learning.utils.common_functions as tmp                        # seed固定用

# 学習全体を管理するクラス
# AdaCos
class AdaCosTrainer:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config['model']                                             # モデル名
        log_dir_name = self.config['loss'] + '/' + self.model_name \
                            + '_' + self.config['name'] \
                            + '_epochs' + str(self.config['epochs']) \
                            + '_batch_size' + str(self.config['batch_size']) \
                            + '_lr' + str(self.config['optimizer']['lr']) \
                            + '_weight_decay' + str(self.config['optimizer']['weight_decay']) \
                            + '_margin' + str(self.config['margin']) \
                            + '_scheduler' + str(self.config['step_size']) \
                            + '_scale' + str(self.config['scale'])                         # ログを保存するフォルダ名
        self.log_path = os.path.join(self.config['base_log_path'], log_dir_name)           # ログの保存先
        img_size = (self.config['width'],self.config['height'])                            # 画像サイズ
        self.logger = Logger(self.log_path)                                                # ログ書き込みを行うLoggerクラスの宣言
        num_class = self.config['num_classes']                                             # クラス数
        step_size = self.config['step_size']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # バッチ読み込みをいくつのスレッドに並列化するか指定
        # パラメータ辞書に'-1'と登録されていればCPUのコア数を読み取って指定
        num_workers = 0
        if self.config['num_workers'] == -1:
            print('set num_workers to number of cpu cores :', os.cpu_count())
            num_workers = os.cpu_count()
        else:
            num_workers = self.config['num_workers']

        # seed固定
        # https://qiita.com/north_redwing/items/1e153139125d37829d2d より，とりあえず全部入れとく
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        self.g = torch.Generator()
        self.g.manual_seed(seed)
        self.worker_init_fn = worker_init_fn
        tmp.NUMPY_RANDOM.seed(seed)

        # データローダーの定義
        # data_path : データの保管場所
        # batch_size : バッチサイズ
        # img_size : 画像サイズ(タプルで指定)
        # train_ratio : 全データ中学習に使用するデータの割合
        self.data_loader = DataLoader(data_path=self.config['train_data_path'],
                                      batch_size=self.config['batch_size'],
                                      img_size=img_size,
                                      train_ratio=self.config['train_ratio'],
                                      num_workers=num_workers,
                                      pin_memory=self.config['pin_memory'],
                                      worker_init_fn=self.worker_init_fn,
                                      generator=self.g)
        if('vgg' in self.config['model']):
            self.model = VGG_based(num_class).to(self.device)                                          # ネットワークを定義(VGGベース)
        elif('resnet' in self.config['model']):
            self.model = PretrainedResNet(num_class, config['pretrained']).to(self.device)             # ネットワークを定義(学習済みResNetベース AAPがあるため再現性がない)
       
        # 学習済み重みファイルがあるか確認しあれば読み込み
        if os.path.isfile(os.path.join(self.log_path, self.model_name)):
            print('Trained weight file exists')
            self.model.load_state_dict(torch.load(os.path.join(self.log_path, self.model_name)))

        # AdaCosを実装
        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.metric_fn = AdaCos(self.config['embedding_size'], num_class, m=self.config['margin']).to(self.device)

        # configからoptimizerを作成
        def make_optimizer(params, name, **kwargs):
            return optim.__dict__[name](params, **kwargs)
        self.optimizer = make_optimizer(list(self.model.parameters())+list(self.loss.parameters()),**self.config['optimizer'])
        # self.optimizer = make_optimizer(self.model.parameters(),**self.config['optimizer'])

        # StepLRスケジューラの設定．
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=0.1)
        
        self.earlystopping = EarlyStopping(20, verbose=False, path=self.log_path+'/best_weight.pth')

    def accuracy(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = 1
            # outputを1hot bectorに直す
            _, pred = topk(output, maxk, 1, True, True)
            pred = pred.t()
            # 教師ラベルと見比べる（多分）
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            # Tensor -> float
            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
            return correct_k
            # return res

    # 学習を行う関数
    def train(self):
        print('Train with AdaCos phase')
        logging.error('train_acc, train_loss, val_acc, val_loss')
        epochs = self.config['epochs']

        # 学習ループ開始(tqdmによって進捗を表示する)
        with tqdm(range(epochs)) as pbar:
            for epoch in enumerate(pbar):
                i = epoch[0]                                    # 現在のepoch
                pbar.set_description('[Epoch %d]' % (i+1))      # プログレスバーのタイトル部分の表示を変更
                loss_result = 0.0                               # Lossを保存しておく
                acc = 0.0                                       # Accuracyの計算に使用
                val_loss_result = 0.0                           # Validation Lossを保存しておく
                val_acc = 0.0                                   # Validation Accuracyの計算に使用
                j = 1                                           # 現在のiterationを保存しておく変数

                for inputs, labels in self.data_loader.dataloaders['train']:            # イテレータからミニバッチを順次読み出す
                    self.model.train()                                                  # モデルをtrainモード(重みが変更可能な状態)にする
                    self.metric_fn.train()
                    pbar.set_description('[Epoch %d (Iteration %d)]' % ((i+1), j))      # 現在のiterationをプログレスバーに表示
                    inputs = inputs.to(self.device, non_blocking=True)                  # 入力データをGPUメモリに送る(non_blocking=Trueによってasynchronous GPU copiesが有効になりCPUのPinned MemoryからGPUにデータを送信中でもCPUが動作できる)
                    labels = labels.to(self.device, non_blocking=True)                  # 教師ラベルをGPUメモリに送る
                    feature = self.model(inputs)
                    outputs = self.metric_fn(feature, labels, True)
                    loss = self.loss(outputs, labels)                                   # 教師ラベルとの損失を計算
                    acc1, = self.accuracy(output=outputs, target=labels)                # accuracyを計算
                    acc += acc1.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_result += loss.item()

                    j = j + 1

                else:
                    with torch.no_grad():
                        # モデルを評価モードに移行 BN, DOの動きが変わるため，忘れると正しい学習ができなくなる可能性大
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

                # val_epoch_lossを指標にearlystopping
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
