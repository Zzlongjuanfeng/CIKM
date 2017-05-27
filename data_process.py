import numpy as np
import re
import random
import os
import itertools
from collections import Counter


def process_line(line):
    """
    :param line: str
    :return: list
    """
    data_line = re.split(r'[;,\s]\s*', line.strip())  #Regular expression
    index = data_line.pop(0)
    y = data_line.pop(0)
    # print(type(data_line))
    return [data_line, y, index]

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

def get_varifacation(data_file, veri_file, train_file):

    with open(veri_file, 'w') as veri_f:
        with open(train_file, 'w') as train_f:
            # build a varification set with the ratio
            ratio = 0.9
            with open(data_file) as file_object:
                for line in file_object:

                    # print(type(line))
                    rand = random.random()
                    if rand > ratio:
                        veri_f.writelines(line)
                    else:
                        train_f.writelines(line)


def batch_iter1(data_file, batch_size):
    """
    Generates a batch iterator for a dataset.
    """
    count = 0
    x_data = []
    y_data = []
    with open(data_file) as file_object:
        for line in file_object:

            x, y, index = process_line(line)
            x_data.append(x)
            y_data.append(y)
            count += 1

            if count % batch_size == 0:
                yield [x_data, y_data]
                x_data = []
                y_data = []



if __name__ == "__main__":
    batch_size = 50
    data_size = 10000   # the number of training data
    data_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train.txt"

    test_batchSize = 100
    test_size = 2000
    test_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_testA/testA.txt"

    sample_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/data_sample.txt"

    #Output directory for models and summaries
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "CIKM_data", "CIKM2017_train"))
    veri_file = os.path.join(out_dir, "tmp", "veri_file.txt")
    train_file = os.path.join(out_dir, "tmp", "train_file.txt")
    print("Writing to {}".format(out_dir))
    print(veri_file)
    print(train_file)

    get_varifacation(data_file = data_file, veri_file = veri_file, train_file = train_file)

    # i = 0
    # batchs = batch_iter1(veri_file, batch_size)
    # for batch in batchs:
    #     i += 1
    #     print(i)
    #     print(len(batch[0]))
    #     # print(len(batch[1]))




