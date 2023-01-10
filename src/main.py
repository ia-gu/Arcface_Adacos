from arcface_trainer import ArcfaceTrainer
from adacos_trainer import AdaCosTrainer
from mktrain_data import mktrain_data
from mktest_data import mktest_data
from test import test
import os
import argparse
import yaml
import logging

def main():
    if(not os.path.exists('train_data')):
        mktrain_data()
    if(not os.path.exists('test_data')):
        mktest_data()

    def get_args():
        parser = argparse.ArgumentParser(description='YAML')
        parser.add_argument('config_path', type=str, help='.yaml')
        args = parser.parse_args()
        return args
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

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
