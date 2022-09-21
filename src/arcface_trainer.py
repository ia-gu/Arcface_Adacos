# PyTorch
import torch                                                                        # pytorchを使用(モデル定義及び学習フロー等)
import torch.nn as nn
import torch.optim as optim
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
# PML
import pytorch_metric_learning
from pytorch_metric_learning import losses                                          # Arcfaceに使用
import pytorch_metric_learning.utils.common_functions as tmp                        # seed固定用

# 学習全体を管理するクラス
class ArcfaceTrainer:
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
        # https://qiita.com/north_redwing/items/1e153139125d37829d2d より，とりあえず全部固定する
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
            self.model = VGG_based(num_class).to(self.device)                                              # ネットワークを定義(VGGベース)
        elif('resnet' in self.config['model']):
            self.model = PretrainedResNet(num_class, config['pretrained']).to(self.device)                 # ネットワークを定義(学習済みResNetベース AAPがあるため再現性がない)
       
        # 学習済み重みファイルがあるか確認しあれば読み込み
        if os.path.isfile(os.path.join(self.log_path, self.model_name)):
            print('Trained weight file exists')
            self.model.load_state_dict(torch.load(os.path.join(self.log_path, self.model_name)))
        
        '''
        # 学習済み重みファイルを使う場合の読み込み  pathは都度変えないとだめ
        # if os.path.isfile('/home/ueno/Arcface/pretrain_weight/logs/pretrained/resnet_sgd19_epochs200_batch_size128_lr0.01_weight_decay1e-06_margin14.3_scheduler30_scale64/resnet'):
        #     print('load weight file')
        #     self.model.load_state_dict(torch.load('/home/ueno/Arcface/pretrain_weight/logs/pretrained/resnet_sgd19_epochs200_batch_size128_lr0.01_weight_decay1e-06_margin14.3_scheduler30_scale64/resnet'))
        # else: 
        #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        '''

        # CNN部分の最適化手法の定義
        # ArcFaceLoss
        # 簡単のためpytorch_metric_learningからimportして読み込み
        # margin : クラス間の分離を行う際の最少距離(cosine類似度による距離学習を行うためmarginはθを示す)
        # scale : クラスをどの程度の大きさに収めるか
        # num_classes : ArcFaceLossにはMLPが含まれるためMLPのパラメータとして入力
        # embedding_size : 同上
        self.loss = losses.ArcFaceLoss(margin=self.config['margin'],
                                        scale=self.config['scale'],
                                        num_classes=num_class,
                                        embedding_size=self.config['embedding_size']).to(self.device)

        # configからoptimizerを作成
        def make_optimizer(params, name, **kwargs):
            return optim.__dict__[name](params, **kwargs)
        self.optimizer = make_optimizer(list(self.model.parameters())+list(self.loss.parameters()),**self.config['optimizer'])
        

        # StepLRスケジューラ設定
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=0.1)

        self.earlystopping = EarlyStopping(20, verbose=False, path=self.log_path+'/best_weight.pth')


    # 学習を行う関数
    def train(self):
        logging.error('train_acc, train_loss, val_acc, val_loss')
        print('Train with Arcface phase')
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

                self.model.train()                              # モデルをtrainモード(重みが変更可能な状態)にする
                j = 1                                           # 現在のiterationを保存しておく変数
                for inputs, labels in self.data_loader.dataloaders['train']:            # イテレータからミニバッチを順次読み出す
                    pbar.set_description('[Epoch %d (Iteration %d)]' % ((i+1), j))      # 現在のiterationをプログレスバーに表示
                    inputs = inputs.to(self.device, non_blocking=True)                  # 入力データをGPUメモリに送る(non_blocking=Trueによってasynchronous GPU copiesが有効になりCPUのPinned MemoryからGPUにデータを送信中でもCPUが動作できる)
                    labels = labels.clone().detach()
                    labels = labels.to(self.device, non_blocking=True)                  # 教師ラベルをGPUメモリに送る
                    outputs = self.model(inputs)                                        # モデルにデータを入力し出力を得る
                    loss = self.loss(outputs, labels)                                   # 教師ラベルとの損失を計算

                    # ArcFaceLossから出力を取り出してAccuracyを計算する
                    # 参考: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/175
                    mask = self.loss.get_target_mask(outputs, labels)                                                               # マスクを取得(1つだけ1で他が0の配列)
                    cosine = self.loss.get_cosine(outputs)                                                                          # 余弦を取得
                    cosine_of_target_classes = cosine[mask == 1]                                                                    # マスクにかける(マスクの1のところを取り出す)
                    modified_cosine_of_target_classes = self.loss.modify_cosine_of_target_classes(cosine_of_target_classes)
                    diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
                    logits = cosine + (mask*diff)                                                                                   # logitsはsoftmax前のNNの出力のこと(らしい)
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
                        # モデルを評価モードに移行 BN, DOの動きが変わるため，忘れると正しい学習ができなくなる可能性大
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
                # val_epoch_lossを指標にearlystopping
                self.earlystopping((val_epoch_loss), self.model) 
                if self.earlystopping.early_stop: 
                    print('Early Stopping!')
                    break

        torch.save(self.model.state_dict(), os.path.join(self.log_path,self.model_name))
        self.logger.draw_graph()
        self.logger.writer.flush()

        logging.error(self.optimizer)

        test(self.config, self.model, self.log_path, self.worker_init_fn, self.g)
