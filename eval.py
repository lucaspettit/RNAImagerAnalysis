import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle
import utils
from os.path import join
from sklearn import neural_network
from random import shuffle, randint

A = ['identity', 'logistic', 'tanh', 'relu']
data = pickle.load(open('bfw_vectorized_images.pkl', 'rb'))
data = utils.pack(data['x'], data['y'])
shuffle(data)

train = data[:(int(len(data) * 0.8))]
test = data[(int(len(data) * 0.8)):]

trainx, trainy = utils.unpack(train)
trainx, trainy = np.array(list(trainx)), np.array(list(trainy))
testx, testy = utils.unpack(test)
testx, testy = np.array(list(testx)), np.array(list(testy))
del data

layers = [
    [157, 157, 157],
    [180],
    [140],
    [199, 199, 199]
]
vanilla = [100, 100, 100]

missed = []
vanilla_errors = []
optimal_errors = []

for a, l in zip(A, layers):

    print(a)
    nn = neural_network.MLPClassifier(hidden_layer_sizes=l, activation=a, shuffle=False)
    nn.fit(trainx, trainy)

    error = 0
    for _x, _y in zip(testx, testy):
        res = nn.predict(np.array(_x).reshape((1, -1)))
        if res * _y <= 0:
            error += 1
            missed.append((_x, _y))

    optimal_errors.append(error / len(testx))

    error = 0
    nn = neural_network.MLPClassifier(hidden_layer_sizes=vanilla, activation=a, shuffle=False)
    nn.fit(trainx, trainy)
    for _x, _y in zip(testx, testy):
        res = nn.predict(np.array(_x).reshape((1, -1)))
        if res * _y <= 0:
            error += 1
    vanilla_errors.append(error / len(testx))


outp = utils.mkpath(('report', 'missed'))
f = open(join(outp, 'missed_vectors.pkl'), 'wb')
pickle.dump(missed, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

res = np.array(vanilla_errors) / np.array(optimal_errors)
print('scores : {0}'.format(res))
f = open(join(outp, 'improvements.txt'), 'w')
for a, v, o, r in zip(A, vanilla_errors, optimal_errors, res):
    f.write('{0} : {1} / {2} = {3}\n'.format(a, v, o, r))
f.close()


x = [i for i in range(len(A))]
plt.scatter(x, vanilla_errors, color='b')
plt.scatter(x, optimal_errors, color='y')
plt.show()
