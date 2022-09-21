import torch                        # モデル定義に使用
import torch.nn.functional as F     # GAPの定義時に使用
import torch.nn as nn               # 各モジュールの宣言に使用
from torchvision import models

# モデルの定義をまとめたファイル
# 実装：VGGベースモデル，ResNet50ベースモデル，学習済みResNet50ベースモデル
# Arcfaceに渡す際，512次元である必要があるので注意

# モデル定義(VGGベースモデル)
class VGG_based(nn.Module):

    def __init__(self, num_class):
        super().__init__()      #スーパークラスの初期化関数を実行(実行しないとモジュールの宣言時にエラーが発生する)

        # 畳み込み，Batch Normalization，ReLUを3回ずつ行うものを1ブロックとし，ブロックの最後にMaxPoolingによって解像度を落とす
        # VGG NetにはBatch Normalizationは含まれていないが，学習の効率化/安定化/精度向上が期待できるため導入
        #
        # Conv2d(input_channel, output_channel, kernel_size, padding)
        # 畳み込みを行うモジュール
        # input_channel     : 入力チャネル数
        # output_channel    : 出力チャネル数
        # kernel_size       : 畳み込みカーネルの大きさ(intを入力すると(int, int)のタプルとして認識される)
        # padding           : 縁の画素に対する畳み込みを行うためのpaddingを行うか(1以上で指定した数字分だけ画像の周囲に画素値0の画素を追加する)
        #
        # BatchNorm2d(channel)
        # Batch Normalizationを行うモジュール
        # Batch Normalization : ミニバッチ内のデータを平均0，標準偏差1になるように正規化を行うこと
        # channel           : 入力チャネル数
        #
        # ReLU(inplace)
        # 活性化関数ReLUをかけるモジュール
        # inplace           : 出力の保存のために入力の変数を用いるか(x = func(x)を許容するか)
        #
        # MaxPool2d(kernel_size, stride)
        # Max Poolingを行うモジュール
        # kernel_size       : 比較を行う際に見る範囲(カーネルサイズ)の大きさ
        # stride            : カーネルを何画素移動するか
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),     #input:144×144×1      output:144×144×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),    #input:144×144×64     output:144×144×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),    #input:144×144×64     output:144×144×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, 3, padding=1),   #input:72×72×64     output:72×72×128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),  #input:72×72×128    output:72×72×128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),  #input:72×72×128    output:72×72×128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, 3, padding=1),  #input:36×36×128      output:36×36×256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  #input:36×36×256      output:36×36×256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  #input:36×36×256      output:36×36×256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, 3, padding=1),  #input:18×18×256      output:18×18×512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),  #input:18×18×512      output:18×18×512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),  #input:18×18×512      output:18×18×512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 全結合層
        # 多層パーセプトロンにより必要な特徴の重み付けを行い識別を行う
        # 多層なので非線形識別が可能
        # Linear(input_dim, output_dim)
        # 全結合層
        # input_dim     : 入力の次元数(全結合層は1次元配列が入力のため次元数はintでいい)
        # output_dim    : 出力の次元数
        self.linear = nn.Sequential(
            nn.Linear(9*9*512, 2048),              #input:9×9×256(1dim vec)    output:1×1×2058
            nn.ReLU(inplace = True),
            nn.Linear(2048, 1024),                  #input:1×1×2048      output:1×1×1024
            nn.ReLU(inplace = True),
            nn.Linear(1024, 512),                  #input:1×1×1024      output:1×1×512
            nn.ReLU(inplace = True),
        )

    # 順伝播
    # 逆伝播は自動でやってくれる
    def forward(self, x):
        x = self.feature_extractor(x)   # 特徴抽出
        x = x.view(-1, 9*9*512)         # 1次元配列に変更
        x = self.linear(x)              # 全結合層へ入力
        return x




# モデル定義(ResNet50ベース)
class ResNet_based(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Skip Connectionにより計算されない情報を送ることで層を重ねることによる初めの方の情報の欠落を防ぐ
        # 加えてデータの流れが分岐するため疑似的なアンサンブルモデルと捉えることが可能
        # 実装はhttps://www.bigdata-navi.com/aidrops/2611/ を参考に行った
        #
        # 畳み込みを1回挿む
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Block 1
        self.block0 = self._building_block(256, channel_in=64)
        self.block1 = nn.ModuleList([
            self._building_block(256) for _ in range(2)
        ])
        self.conv2 = nn.Conv2d(256, 512, 1, stride=(2, 2))
        # Block 2
        self.block2 = nn.ModuleList([
            self._building_block(512) for _ in range(4)
        ])
        self.conv3 = nn.Conv2d(512, 1024, 1, stride=(2, 2))
        # Block 3
        self.block3 = nn.ModuleList([
            self._building_block(1024) for _ in range(6)
        ])
        self.conv4 = nn.Conv2d(1024, 2048, 1, stride=(2, 2))
        # Block 4
        self.block4 = nn.ModuleList([
            self._building_block(2048) for _ in range(3)
        ])
        self.avg_pool = GlobalAvgPool2d()                           # x.viewの代わりにGAPを使用
        self.fc = nn.Linear(2048, 512)
        self.out = nn.Linear(512, 100)                                # Arcfaceが行うので不要．重みが学習されていないため，テストでの使用不可

    def pretrain(self, x):
        x = self.head(x)
        x = self.block0(x)
        for block in self.block1:
            x = block(x)
        x = self.conv2(x)
        for block in self.block2:
            x = block(x)
        x = self.conv3(x)
        for block in self.block3:
            x = block(x)
        x = self.conv4(x)
        for block in self.block4:
            x = block(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.out(x)
        return x
        
    # 順伝播
    def forward(self, x):
        x = self.head(x)
        x = self.block0(x)
        for block in self.block1:
            x = block(x)
        x = self.conv2(x)
        for block in self.block2:
            x = block(x)
        x = self.conv3(x)
        for block in self.block3:
            x = block(x)
        x = self.conv4(x)
        for block in self.block4:
            x = block(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        x = torch.relu(x)
        #x = self.out(x)
        #x = torch.log_softmax(x, dim=-1)
        return x


    # テスト時 SiameseNet
    def test(self, x1, x2):
        x1 = self.head(x1)
        x1 = self.block0(x1)
        for block in self.block1:
            x1 = block(x1)
        x1 = self.conv2(x1)
        for block in self.block2:
            x1 = block(x1)
        x1 = self.conv3(x1)
        for block in self.block3:
            x1 = block(x1)
        x1 = self.conv4(x1)
        for block in self.block4:
            x1 = block(x1)
        x1 = self.avg_pool(x1)
        x1 = self.fc(x1)
        x1 = torch.relu(x1)

        x2 = self.head(x2)
        x2 = self.block0(x2)
        for block in self.block1:
            x2 = block(x2)
        x2 = self.conv2(x2)
        for block in self.block2:
            x2 = block(x2)
        x2 = self.conv3(x2)
        for block in self.block3:
            x2 = block(x2)
        x2 = self.conv4(x2)
        for block in self.block4:
            x2 = block(x2)
        x2 = self.avg_pool(x2)
        x2 = self.fc(x2)
        x2 = torch.relu(x2)
        
        return x1, x2


    def _building_block(self, channel_out, channel_in=None):
        if channel_in is None:
            channel_in = channel_out
        return ResBlock(channel_in, channel_out)

# ResNetを構成するブロックを生成するクラス
# 実装はhttps://www.bigdata-navi.com/aidrops/2611/ を参考に行った
class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        channel = channel_out

        self.block = nn.Sequential(
            nn.Conv2d(channel_in, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel_out, 1, padding=0),
            nn.BatchNorm2d(channel_out),
        )
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.block(x)
        shortcut = self.shortcut(x)
        x = self.relu(h + shortcut)
        return x
    
    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, 1, padding=0)

# GAPを計算するクラス
# 実装はhttps://www.bigdata-navi.com/aidrops/2611/ を参考に行った
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))







# 学習済みResNet50を利用(全結合層のみ定義し直し)
# torchvisionに含まれるResNet50学習済みモデルを使用
# 実装はhttps://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html を参考に行った
# AAPが含まれており，seedが固定されないため再現性は無
class PretrainedResNet(nn.Module):
    def __init__(self, embedding_size, pretrain):
        super().__init__()
        self.pretrained_resnet = models.resnet50(pretrained=pretrain)
        self.fc = nn.Linear(2048, 512)                              #学習済みresnetの全結合層のみ入れ替え

    def forward(self, x):
        x = self.pretrained_resnet.conv1(x)
        x = self.pretrained_resnet.bn1(x)
        x = self.pretrained_resnet.relu(x)
        x = self.pretrained_resnet.maxpool(x)
        x = self.pretrained_resnet.layer1(x)
        x = self.pretrained_resnet.layer2(x)
        x = self.pretrained_resnet.layer3(x)
        x = self.pretrained_resnet.layer4(x)
        x = self.pretrained_resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # テスト時はSiameseNet
    def test(self, x1, x2):
        x1 = self.pretrained_resnet.conv1(x1)
        x1 = self.pretrained_resnet.bn1(x1)
        x1 = self.pretrained_resnet.relu(x1)
        x1 = self.pretrained_resnet.maxpool(x1)
        x1 = self.pretrained_resnet.layer1(x1)
        x1 = self.pretrained_resnet.layer2(x1)
        x1 = self.pretrained_resnet.layer3(x1)
        x1 = self.pretrained_resnet.layer4(x1)
        x1 = self.pretrained_resnet.avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.fc(x1)

        x2 = self.pretrained_resnet.conv1(x2)
        x2 = self.pretrained_resnet.bn1(x2)
        x2 = self.pretrained_resnet.relu(x2)
        x2 = self.pretrained_resnet.maxpool(x2)
        x2 = self.pretrained_resnet.layer1(x2)
        x2 = self.pretrained_resnet.layer2(x2)
        x2 = self.pretrained_resnet.layer3(x2)
        x2 = self.pretrained_resnet.layer4(x2)
        x2 = self.pretrained_resnet.avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc(x2)

        return x1, x2