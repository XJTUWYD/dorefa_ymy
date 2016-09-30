import os

if __name__ == '__main__':
    for i in range(0, 5):
        f = open('./benchmark/fold_' + str(i) + '_data.txt', 'r')
        f2 = open('./benchmark/data' + str(i) + '.txt', 'w')

        line = f.readline() # 1行目は飛ばす
        line = f.readline()
        while line:
            splitted = line[:-1].split('\t') # 改行コードを取り除いてタブでスプリット
            # [0]:ディレクトリ名、[1]:ファイル名、[3]:年齢、[4]:性別
            dirName = splitted[0]
            fileName = splitted[1]
            age = str(splitted[3])
            new_line = dirName + '\t' + fileName + '\t' + age + '\n'
            f2.write(new_line)
            line = f.readline()
