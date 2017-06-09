import numpy as np
import pickle
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt

idir = join('report', 'missed')
f = open(join(idir, 'missed_vectors.pkl'), 'rb')
missed = pickle.load(f)

for i in range(len(missed)):
    x, y = missed[i]
    w, h = x[-2], x[-1]
    x = x[:-2]

    x = np.array(x).reshape((28, 28))
    img = Image.fromarray(np.uint8(x)).convert('LA').resize((w, h))
    img.save(join(idir, '{0}_{1}.png'.format(y, i)))
