import os                                           # ディレクトリの存在確認とディレクトリ作成，パス作成に使用
import matplotlib.pyplot as plt                     # グラフの作成，保存に使用
from torch.utils.tensorboard import SummaryWriter   # TensorBoard(ログ管理ライブラリ)用のログファイルの作成，書き込みに使用

# ログの保存に関するクラス
class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        # 保存先のディレクトリが存在しなければディレクトリを作成
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.writer = SummaryWriter(log_dir=log_path)
        # 学習で計算されたLossとAccuracyを保存しておく配列
        self.loss_history = []
        self.acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    # LossとAccuracyの履歴を追加していく関数
    # クラス内変数に保存されるためフォーカス外から読み出せる
    def collect_history(self, loss, accuracy, val_loss, val_accuracy):
        self.loss_history.append(loss)                  # self.loss_historyにlossを追加
        self.acc_history.append(accuracy)               # self.acc_historyにaccuracyを追加
        self.val_loss_history.append(val_loss)          # self.val_loss_historyにval_lossを追加
        self.val_acc_history.append(val_accuracy)       # self.val_acc_historyにval_accuracyを追加

    # LossとAccuracyのグラフを描画して保存する関数
    def draw_graph(self):
        plt.plot(self.loss_history, label='loss')                   # self.loss_history(学習データに関するLoss)のプロット
        plt.plot(self.val_loss_history, label='val_loss')           # self.val_loss_history(検証データに関するLoss)のプロット
        plt.ylim([0,1.1])                                           # y軸の固定
        plt.legend()                                                # 凡例の表示
        plt.savefig(os.path.join(self.log_path, 'loss.png'))        # Lossのグラフを画像として保存
        plt.gca().clear()                                           # グラフの描画履歴を削除
        plt.plot(self.acc_history, label='accuracy')                # self.acc_history(学習データに関するAccuracy)のプロット
        plt.plot(self.val_acc_history, label='val_accuracy')        # self.val_acc_history(検証データに関するAccuracy)のプロット
        plt.legend()                                                # 凡例の表示
        plt.ylim([0,1.1])                                           # y軸の固定
        plt.savefig(os.path.join(self.log_path, 'accuracy.png'))    # Accuracyのグラフを画像として保存
