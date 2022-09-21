import shutil
import os

# homeから必要なテスト用データを引っ張ってくる
def mktest_data():
    print('start test_data installation')

    os.makedirs('test_data', exist_ok=True)
    path = '/home/data/LFW/lfw'

    # すべての画像データのクラス(人の名前)を配列でまとめる
    man_list = []
    for manPath in os.listdir(path):
        man_list.append(manPath)

    # txtファイルの中身を1行ずつ配列(二次元)でまとめる
    content = []
    with open('pairsDevTest.txt') as f:
        for line in f:
            content.append(line.split())

    # txtにあった名前をすべて引っ張ってくる
    # 同クラスなら1人だが他クラスは2人の名前が上がるため，どちらも引っ張れるようにつくる
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