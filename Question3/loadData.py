"""
Define a function: (Sample, Class, Feature, Matrix) = fLoadDataMatrix(FileName)
 
"""
from csv import reader
import numpy as np

def fLoadDataMatrix(FileName, separation):
    with open(FileName, encoding='utf-8') as matrix:
        readers = reader(matrix,delimiter = separation) #假定是txt文件，以制表符分隔的数据
        x = list(readers)
        data = np.array(x)
    row_num=data.shape[0]                               #数据row的值
    col_num=data.shape[1]                               #数据col的值
    
    Sample = data[1 : row_num, 0]                       #样本名字  1..row_num-1, 0
    Class = data[1 : row_num, col_num - 1]              #类别     1..row_num-1, -1
    Feature = data[0, 1 : col_num - 1]                  #特征名字  0, 1..col_num-2  
    Matrix = data[1 : row_num,1 : col_num - 1]          #数据矩阵
    return Sample, Class, Feature, Matrix

if __name__ == '__main__':
    Sample, Class, Feature, Matrix = fLoadDataMatrix('data.txt', '\t')
    print('filename : data.txt')
    print('Sample name :', Sample)
    print('Class label :', Class)
    print('Feature name :', Feature)
    print('Matrix data :\n', Matrix)
    Sample, Class, Feature, Matrix = fLoadDataMatrix('data1.csv', ',')
    print('filename : data1.csv')
    print('Sample name :', Sample)
    print('Class label :', Class)
    print('Feature name :', Feature)
    print('Matrix data :\n', Matrix)
