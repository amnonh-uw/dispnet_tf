from PIL import Image
import numpy as np
from train import load_pfm


def means(image_path):
    im = Image.open(image_path)
    im = np.asarray(im)

    r = np.mean(im[:, :, 0])
    g = np.mean(im[:, :, 1])
    b = np.mean(im[:, :, 2])

    return np.array([r,g,b])

def mean_disp(disp_path):
    d = load_pfm(disp_path)
    return np.mean(d), d.max(), d.min()

if __name__ == '__main__':
    samples = "FlyingThings3D_release_TRAIN.list"
    # samples = "FlyingThings3D_release_TEST.list"
    total = np.array([0.,0.,0.])
    total_disp = 0.0
    max_disp = 0.0
    min_disp = 999999.0
    count = 0
    with open(samples) as f:
        line = f.readline()

        while line:
            count += 1
            e = line.split()
            total += means(e[0])
            total += means(e[1])
            o_mean, o_max, o_min = mean_disp(e[2])
            total_disp += o_mean
            if o_max > max_disp:
                max_disp = o_max
            if o_min < min_disp:
                min_disp = o_min
            
    total /= (count * 2)
    total_disp /= count
    print("images mean: ", total)
    print("disp mean: ", total_disp)
    print("disp max: ", max_disp)
    print("disp min: ", min_disp)

