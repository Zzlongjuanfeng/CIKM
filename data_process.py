import numpy as np
import re
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



def batch_iter1(data_file, batch_size):
    """
    Generates a batch iterator for a dataset.
    """
    count = 0
    x_data = []
    y_data = []
    with open(data_file) as file_object:
        for line in file_object:
            x, y, index= process_line(line)
            x_data.append(x)
            y_data.append(y)
            count += 1
            if count % batch_size == 0:
                yield [x_data, y_data, count]
                x_data = []
                y_data = []
                # print(index)
            # print(type(x_data))


if __name__ == "__main__":
    batch_size = 50
    data_size = 10000   # the number of training data
    data_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train.txt"

    test_batchSize = 200
    test_size = 2000
    test_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_testA/testA.txt"

    batchs = batch_iter1(data_file, batch_size)
    for batch in batchs:
        print(type(batch[0]))
        print(type(batch[1]))

