import shutil
import os

# choose data used in test
# when using, need to change path
def mktest_data():
    print('start test_data installation')

    os.makedirs('test_data', exist_ok=True)
    path = '/home/data/LFW/lfw'

    man_list = []
    for manPath in os.listdir(path):
        man_list.append(manPath)

    content = []
    with open('pairsDevTest.txt') as f:
        for line in f:
            content.append(line.split())

    for i in range(1,int(content[0][0])*2+1):
        if(content[i][0] in man_list):
            for charPath in os.listdir(os.path.join(path,content[i][0])):
                os.makedirs('test_data/'+content[i][0], exist_ok=True)
                shutil.copy(os.path.join(path,content[i][0],charPath), 'test_data/'+content[i][0])
        if(content[i][2] in man_list):
            for charPath in os.listdir(os.path.join(path,content[i][2])):
                os.makedirs('test_data/'+content[i][2], exist_ok=True)
                shutil.copy(os.path.join(path,content[i][2],charPath), 'test_data/'+content[i][2])

    print('test_data loaded')

if __name__ == '__main__':
    mktest_data()