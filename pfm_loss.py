from PIL import Image
import numpy as np
import os
import re
import sys

def load_pfm(name):
    with open(name, "rb") as file:
        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            raise Exception("expecting non color PFM")
        elif header != 'Pf':
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode('utf-8').rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')

    nans = np.count_nonzero(np.isnan(data))
    if nans != 0:
        print("load_pfm: warning {} nans encountered".format(nan))

    shape = (height, width, 1)
    data = np.reshape(data, shape) * scale
    data = np.flipud(data)

    return data

def pfm_loss(f1, f2):
    if f1.endswith(".txt"):
        loss = 0.0
        count = 0
        f1_list = open(f1, "r")
        f2_list = open(f2, "r")
        f1_line = f1_list.readline.decode('utf-8')
        f2_line = f2_list.readline.decode('utf-8')
        while f1 != None and f2 != None:
            f1 = load_pfm(f1_line)
            f2 = load_pfm(f2_line)
            loss_f1_f2 = np.mean(np.fabs(f1-f2))
            print("{} {} loss {}".format(f1_line, f2_line, loss_f1_f2))
            loss += loss_f1_f2;
            count += 1
            f1_line = f1_list.readline.decode('utf-8')
            f2_line = f2_list.readline.decode('utf-8')
        print("total loss {}".format(loss / count))
    else:
        f1 = load_pfm(f1)
        f2 = load_pfm(f2)
        print("loss is {}".format(np.mean(np.fabs(f1-f2))))

if __name__ == '__main__':
    pfm_loss(sys.argv[1], sys.argv[2])
