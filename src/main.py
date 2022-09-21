from arcface_trainer import ArcfaceTrainer   # 自作(Arcfaceを用いた学習に使用)
from adacos_trainer import AdaCosTrainer     # 自作(AdaCosを用いた学習に使用)
from mktrain_data import mktrain_data        # 自作(データローダに使用)
from mktest_data import mktest_data          # 自作(データローダに使用)
from test import test                        # 自作(テスト時に使用)
import os                                    # ファイルの存在確認に使用
import argparse                              # config取得に使用
import yaml                                  # yaml読み込みに使用
import logging                               # テキストログに使用


# main関数
def main():
    # train_data valid_data, test_dataを用意（既にあるなら実行しない）
    if(not os.path.exists('train_data')):
        mktrain_data()
    if(not os.path.exists('test_data')):
        mktest_data()

    # config取得 引数はyamlのみ
    def get_args():
        parser = argparse.ArgumentParser(description='YAMLありの例')
        parser.add_argument('config_path', type=str, help='設定ファイル(.yaml)')
        args = parser.parse_args()
        return args
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 学習のうんぬんかんぬんを管理するクラスを変数として宣言
    if(config['loss']=='arcface'):
        arcface_trainer = ArcfaceTrainer(config)   
        logging.basicConfig(level=logging.ERROR, filename=arcface_trainer.log_path+'/result.txt', format='%(message)s')
        if (not os.path.isfile(os.path.join(arcface_trainer.log_path, arcface_trainer.model_name))):
            print('Trained weight file does not exist')
            arcface_trainer.train()
        elif(config['name']=='debug'):
            print('DEBUG MODE')
            arcface_trainer.train()
        else:
            test(arcface_trainer.config, arcface_trainer.model, arcface_trainer.log_path, arcface_trainer.worker_init_fn, arcface_trainer.g)

    else:
        adacos_trainer = AdaCosTrainer(config)
        logging.basicConfig(level=logging.ERROR, filename=adacos_trainer.log_path+'/result.txt', format='%(message)s')
        if (not os.path.isfile(os.path.join(adacos_trainer.log_path, adacos_trainer.model_name))):
            print('Trained weight file does not exist')
            adacos_trainer.train()
        elif(config['name']=='debug'):
            print('DEBUG MODE')
            adacos_trainer.train()
        else:
            test(adacos_trainer.config, adacos_trainer.model, adacos_trainer.log_path, adacos_trainer.worker_init_fn, adacos_trainer.g)

if __name__ == '__main__':
    main()
