import shutil
import os
# homeから学習用データを引っ張ってくる(mktest_data()とほとんど一緒)
def mktrain_data():
    print('start train_data installation')
    
    os.makedirs('train_data', exist_ok=True)

    path = '/home/data/LFW/lfw'
    man_list = []
    for manPath in os.listdir(path):
        man_list.append(manPath)
    f = open('../peopleDevTrain.txt', 'r')
    data = f.read()
    data = data.split()
    f.close()
    for i in range(len(data)):
        if(i%2==0):
            continue
        if(data[i] in man_list):
            for charPath in os.listdir(os.path.join(path,data[i])):
                os.makedirs('train_data/'+data[i], exist_ok=True)
                shutil.copy(os.path.join(path,data[i],charPath), 'train_data/'+data[i])

    print('train_data loaded')
    
if __name__ == '__main__':
    mktrain_data()