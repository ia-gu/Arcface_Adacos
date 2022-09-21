from torch import utils                         # pytorch内のDataLoaderを使用
from torchvision import datasets, transforms    # ImageFolderと前処理の設定に使用

# 学習用データローダ
class DataLoader:
    def __init__(self, data_path, batch_size, img_size, train_ratio, num_workers, pin_memory, worker_init_fn, generator):
        self.dataset_path = data_path                                                       # データセットのパス
        self.img_size = img_size                                                            # 画像サイズ(タプルで指定)
        self.batch_size = batch_size                                                        # バッチサイズ
        self.worker_init_fn = worker_init_fn
        self.generator = generator
        
        # 画像の変換方法の選択
        # 学習に使用するLFW Databaseは1クラスあたりのデータ数が非常に少ないためData Augmentationを行うことで疑似的に学習データを増やす
        # RandomHorizontalFlip  : 水平反転を行ったり行わなかったりする
        # RandomRotation        : ランダムに傾ける(今回は±3°(畳み込みのカーネルサイズが3×3であるため3°傾けるだけで畳み込みの結果が大きく異なるためこの程度で十分))
        # ToTensor              : 画像データからTorch Tensor(PyTorchで使用できるテンソル)に変換
        # Normalize             : 正規化(今回は各チャネルについて平均0.5，標準偏差0.5に正規化(0.5±0.5程度に収まるため値が0～1になり計算が容易になる効果が期待される))
        self.train_transform = transforms.Compose([transforms.Resize(self.img_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    # transforms.RandomRotation(3),
                                                    transforms.RandomAffine((-3,3),(0.02, 0.02)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # DataLoaderを作成
        # batch_size    : バッチサイズ
        # train_ratio   : 全データの内どれだけを学習データとして用いるか(0～1)
        # num_workers   : ミニバッチの読み込みを並列に行うプロセス数
        # pin_memory    : Automatic memory pinningを使用(CPUのメモリ領域をページングする処理を行わなくなるため読み込みの高速化が期待できる)
        #                 参考：https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
        #                      https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        self.dataloaders = self.import_image(batch_size=batch_size, train_ratio=train_ratio, num_workers=num_workers, pin_memory=True)



    # 学習に使うデータの読み込みを行う関数
    # Argument
    # dataset_path   : データセットが格納されているディレクトリのパス
    # batch_size     : バッチサイズ
    # train_ratio    : データ全体から学習に使うデータの割合(デフォルト値: 0.8)
    # img_size       : 画像のサイズ タプルで指定すること
    # num_workers    : ミニバッチの読み込みを並列に行うプロセス数
    # pin_memory     : Automatic memory pinningを使用
    def import_image(self, batch_size, train_ratio=0.8, num_workers=0, pin_memory=True):
        # torchvision.datasets.ImageFolderで画像のディレクトリ構造を元に画像読み込みとラベル付与を行ってくれる
        # transformには前処理を記述
        data = datasets.ImageFolder(root=self.dataset_path, transform=self.train_transform)

        train_size = int(train_ratio * len(data))                                       # 学習データ数
        val_size = len(data) - train_size                                               # 検証データ数

        train_data, val_data = utils.data.random_split(data, [train_size, val_size])    # torcn.utils.data.random_splitで重複なしにランダムな分割が可能

        # 学習データの読み込みを行うイテレータ
        # shuffle    :学習データの順番をランダムに入れ替えるか
        train_loader = utils.data.DataLoader(train_data,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                worker_init_fn=self.worker_init_fn,
                                                generator=self.generator)
        # 検証データの読み込みを行うイテレータ
        # 検証データは学習を行わないためシャッフルは必要ない
        val_loader = utils.data.DataLoader(val_data,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                worker_init_fn=self.worker_init_fn,
                                                generator=self.generator)
        dataloaders = {'train':train_loader, 'val':val_loader}                          # それぞれのデータローダーをディクショナリに保存
        return dataloaders
