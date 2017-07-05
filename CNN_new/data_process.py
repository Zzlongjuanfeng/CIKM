import numpy as np
import re
import random
import os
import linereader
import csv
import math
import itertools
from collections import Counter


def process_line(line):
    """
    :param line: str
    :return: list
    """
    data_line = re.split(r'[;,\s]\s*', line.strip())  #Regular expression
    index = data_line.pop(0)
    y = float(data_line.pop(0))
    # print(type(data_line))
    x = [float(i) for i in data_line]
    return [x, y, index]

def get_offset(datafile):
    line_offset = []
    offset = 0
    count = 0
    with open(datafile, 'r') as file:
        for line in file:
            line_offset.append(offset)
            offset += len(line)
            count+=1
            print(count)
    outName = os.path.splitext(datafile)[0] + '_offset.txt'
    # print(outName)
    with open(outName, 'w') as out_f:
        for i in line_offset:
            out_f.write(str(i)+'\n')

def process_line_hx(line):
    """
    :param line: str
    :return: list
    """
    data_line = re.split(r'[;,\s]\s*', line.strip())  #Regular expression
    index = data_line.pop(0)
    y = data_line.pop(0)

    #choose data of h1
    #by zhu on 2017.5.25
    selected_data = []
    for t in range(0,15):
        h=1
        selected_data.extend(data_line[t*101*101*4+h*101*101 : t*101*101*4+h*101*101+101*101])
    data_line=selected_data

    # print(type(data_line))
    return [data_line, y, index]

def get_varifacation(data_file, veri_file, train_file, ratio = 0.9):
    with open(veri_file, 'w') as veri_f:
        with open(train_file, 'w') as train_f:
            # build a varification set with the ratio
            with open(data_file) as file_object:
                for line in file_object:

                    # print(type(line))
                    rand = random.random()
                    if rand > ratio:
                        veri_f.writelines(line)
                    else:
                        train_f.writelines(line)

def shuffle_file(data_file):
    data_dir = os.path.split(data_file)[0]
    data_file1 = os.path.join(data_dir, "data_1.txt")
    data_file2 = os.path.join(data_dir, "data_2.txt")
    data_file3 = os.path.join(data_dir, "data_result.txt")

    get_varifacation(data_file, data_file1, data_file2, ratio = 0.5)

    with open(data_file1, 'r') as data_f1:
        with open(data_file2, 'r') as data_f2:
            # build a varification set with the ratio
            # ratio = 0.95
            with open(data_file3, 'w') as file_object:
                for line in data_f1:
                    file_object.writelines(line)
                for line in data_f2:
                    file_object.writelines(line)


def batch_iter1(data_file, batch_size, shuffle = False):
    """
    Generates a batch iterator for a dataset.
    """
    line_offset = []
    offset = 0
    with open(data_file, 'r') as file:
        for line in file:
            line_offset.append(offset)
            offset += len(line)
    length = len(line_offset)

    count = 0
    x_data = []
    y_data = []
    with open(data_file, 'r') as read_file:
        for i in range(length):
            if shuffle is True:
                index = random.randint(0, len(line_offset) -1)
                offset = line_offset.pop(index)
                read_file.seek(offset)

            line = read_file.readline()
            x, y, index = process_line(line)
            x_data.append(x)
            y_data.append(y)
            count += 1

            if count % batch_size == 0:
                yield [np.array(x_data), np.array(y_data)]
                x_data = []
                y_data = []



if __name__ == "__main__":
    batch_size = 50
    data_size = 9500   # the number of training data
    data_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train.txt"

    test_batchSize = 100
    test_size = 2000
    test_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_testA/testA.txt"

    sample_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/data_sample.txt"
    train_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train_file.txt"
    veri_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/veri_file.txt"

    #Output directory for models and summaries
    # out_dir = os.path.abspath(os.path.join(os.path.curdir, "CIKM_data", "CIKM2017_train"))
    # veri_file = os.path.join(out_dir, "veri_file.txt")
    # train_file = os.path.join(out_dir, "train_file.txt")
    # print("Writing to {}".format(out_dir))
    # print(veri_file)
    # print(train_file)

    # get_offset(veri_file)
    # get_varifacation(data_file = data_file, veri_file = veri_file, train_file = train_file)
    #shuffle_file(train_file)


    batches = batch_iter1(sample_file, batch_size = 1, shuffle=True)
    for i, batch in enumerate(batches):
        print((i+1) * 10)
        # print(batch[0].shape)
        print(batch[1])
    #     data_sum += reduce(add, batch[1])
    #     data_ave = data_sum/(i+1)/batch_size
    #     e_sum = 0
    #     y_all.extend(batch[1])
    #     y_m = [(x - data_ave)*(x - data_ave) for x in y_all]
    #     e_sum += reduce(add, y_m)
    #     rmse = math.sqrt(e_sum / (i+1) /batch_size)
    #     print (data_ave, rmse)



