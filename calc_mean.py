from PIL import Image
import numpy as np


def means(image_path):
    im = Image.open(image_path)
    im = np.asarray(im)

    r = im.mean(im[::0])
    g = im.mean(im[::1])
    b = im.mean(im[::2])

    return np.array([r,g,b])

if __name__ == '__main__':
    total = np.array([0.,0.,0.])
    count = 0
    with open("FlyingThings3D_release_TRAIN.list") as f:
        line = f.readline()

        while line:
            count += 2
            e = line.split()
            total += means(e[0])
            total += means(e[1])

    total /= count
    print(total)

