import numpy as np
import pickle
from os.path import isfile, isdir, join
from os import listdir
from PIL import Image

idir = join('raw_photos')
odir = ''
files = [f for f in listdir(idir) if isfile(join(idir, f))]

x = []
y = []

for f in files:
    label = int(f.split('_')[0])
    label = -1 if label == 0 else label
    if label not in (-1, 1):
        raise ValueError('unexpected value for label')

    img = Image.open(join(idir, f))

    for rotation in (0, 90, 180, 270):
        img_rot = img.rotate(rotation)
        w, h = img_rot.size
        arr = list(img_rot.convert('LA').resize((28, 28)).getdata(band=0)) + [w, h]

        x.append(np.array(arr))
        y.append(label)

data = {'x': x, 'y': y}
pickle.dump(data, open(join(odir, 'bfw_vectorized_images.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

print('saved {0} data points'.format(len(x)))

