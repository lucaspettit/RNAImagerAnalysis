import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import utils
import pickle
from queue import PriorityQueue
from sys import getsizeof
from sklearn import tree, neighbors, svm, neural_network, ensemble
from threading import Thread, Lock
from timeit import default_timer as timer
from os.path import join
from random import shuffle

# classifying on M/F (omitting "Infant" label)
data_set = 'rna'
print(data_set.upper())

# build constants
S = 0.9, 0.8, 0.7
C = {S[0]: 'r', S[1]: 'g', S[2]: 'b'}
A = ['identity', 'logistic', 'tanh', 'relu']
TREE = 'tree'
KNN = 'k-nn'
SVM = 'svm'
NN = 'neural'
BOOST = 'boost'

TITLE = 'title'
XLABEL = 'x'
FILENAME = 'file'
PHOTONAME = 'photo'
RANGE = 'range'

SVM_KERNEL = 'rbf'
MAX_LAYER = 50
DIR = utils.mkpath(['report', 'classifiers'])

classifiers = {
    TREE: lambda d: tree.DecisionTreeClassifier(max_depth=d),
    KNN: lambda n: neighbors.KNeighborsClassifier(n_neighbors=n),
    SVM: lambda c: svm.SVC(C=c, kernel=SVM_KERNEL),
    NN: lambda i: neural_network.MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation=A[i]),
    BOOST: lambda n: ensemble.AdaBoostClassifier(n_estimators=n)
}

meta = {
    TREE: {
        TITLE: 'Decision Tree Classifier',
        XLABEL: 'depth',
        FILENAME: 'tree_res.txt',
        PHOTONAME: 'tree_graph.png',
        RANGE: [i for i in range(1, 101)]
    },
    KNN: {
        TITLE: 'K Nearest Neighbors Classifier',
        XLABEL: 'neighbors',
        FILENAME: 'k-nn_res.txt',
        PHOTONAME: 'k-nn_graph.png',
        RANGE: [i for i in range(1, 101)]
    },
    SVM: {
        TITLE: 'Support Vector Machine - {0}'.format(SVM_KERNEL),
        XLABEL: 'C',
        FILENAME: 'svm_res_{0}.txt'.format(SVM_KERNEL),
        PHOTONAME: 'svm_graph_{0}.png'.format(SVM_KERNEL),
        RANGE: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5] + [i for i in range(1, 101)]
    },
    NN: {
        TITLE: 'Neural Network',
        XLABEL: '0: {0}    1: {1}    2: {2}    3: {3}'.format(A[0], A[1], A[2], A[3]),
        FILENAME: 'neural_res.txt',
        PHOTONAME: 'neural_graph.png',
        RANGE: [i for i in range(len(A))]
    },
    BOOST: {
        TITLE: 'Ada Boost',
        XLABEL: 'num estimators',
        FILENAME: 'ada_boost_res.txt',
        PHOTONAME: 'ada_boost_graph.png',
        RANGE: [i for i in range(10, 101)]
    }
}

keys = (TREE, KNN, NN, BOOST)

# get data
data = pickle.load(open('bfw_vectorized_images.pkl', 'rb'))
data = utils.pack(data['x'], data['y'])
shuffle(data)


def optimize(data, values, kernel):
    max_split = max(int((len(values) - 1) / 10), 1)
    subsets = utils.split(values, max_split)
    lock = Lock()
    q = PriorityQueue()

    def _optimize(index):
        for i in subsets[index]:
            utils.debug('Thread {0}: arg = {1}'.format(index, i))
            e = (i, utils.kfold(data, i, kernel, k=5))
            lock.acquire()
            q.put(e)
            lock.release()

    start = timer()
    threads = []
    utils.debug('spawning {0} threads'.format(len(subsets)))
    for i in range(min(len(subsets), 10)):
        t = Thread(target=_optimize, args=(i,))
        threads.append(t)
        t.start()
    for i in range(len(threads)):
        threads[i].join()

    lock.acquire()
    if q.empty():
        lock.release()
        raise ValueError('queue empty')
    lock.release()

    end = timer()
    utils.debug('calculation time = {0}'.format(end - start))

    val_errors = []
    while not q.empty():
        val_errors.append(q.get())

    optimal, val_error = sorted(list(val_errors), key=lambda t: t[1])[0]
    val_errors = [ve for i, ve in val_errors]
    optimal = int(optimal)

    # get training error for optimal arg
    x, y = utils.unpack(data)
    x, y = np.array(list(x)), np.array(list(y))
    k = kernel(optimal)
    k.fit(x, y)

    train_error = 0
    start = timer()
    for i in range(len(data)):
        features, label = data[i]
        res = k.predict(np.array(features).reshape((1, -1))) * label
        train_error = utils.running_average(train_error, 0 if res > 0 else 1, i)
    end = timer()
    calc_time = (end - start) / len(data)

    return optimal, val_errors, train_error, val_error, calc_time


print('{0} data points loaded'.format(len(data)))

# get user input
cmd = ''
auto = -1
while cmd != 'quit':
    if auto < 0:
        cmd = input('specify classifier > ').lower()
        if cmd not in classifiers.keys() and cmd != 'auto':
            print('unsupported classifier: {0}'.format(cmd))
            print('supported classifiers: tree, k-nn, svm, neural, or auto')
            continue
        if cmd == 'auto':
            auto = 0
            cmd = keys[auto]
            utils.debug('~~~ {0} ~~~'.format(meta[cmd][TITLE]))
    else:
        auto += 1
        if auto > len(classifiers.keys()):
            break
        cmd = keys[auto]
        utils.debug('~~~ {0} ~~~'.format(meta[cmd][TITLE]))

    # set mutable parameters
    kernel = classifiers[cmd]
    solution_space = meta[cmd][RANGE]
    filename = meta[cmd][FILENAME]
    photoname = meta[cmd][PHOTONAME]
    plt.title(meta[cmd][TITLE])
    plt.xlabel(meta[cmd][XLABEL])
    plt.ylabel('validation error')
    buffer = ''

    for s in S:
        train = np.array(data[:int(len(data) * s)])
        test = np.array(data[int(len(data) * s):])

        optimal, errors, train_error, val_error, calc_time = optimize(train, solution_space, kernel)

        if len(errors) != len(solution_space):
            utils.debug('invalid error size: len(errors) != len(solution_space)\n {0} != {1}'.format(len(errors), len(solution_space)))

        _tx, _ty = utils.unpack(train)
        t = kernel(optimal)
        t.fit(list(_tx), list(_ty))

        e = utils.score(test, lambda a: t.predict(np.array([a]).reshape(1, -1)))

        buffer += str('split       = {0}\n' 
                      'arg*        = {1}\n' 
                      'calc time   = {2} ms\n'
                      'memory      = {3} Kb\n'
                      'val. error  = {4}\n'
                      'train error = {5}\n'
                      'test error  = {6}\n\n'
                      .format(str(s * 100),
                              optimal if cmd.lower() != NN else A[optimal],
                              round(calc_time * 1000, 10),
                              (getsizeof(t) / 1000),
                              round(val_error, 4),
                              round(train_error, 4),
                              round(e, 4)))

        if cmd == NN:
            plt.scatter(solution_space, errors, color=C[s])
        else:
            plt.plot(solution_space, errors, C[s])
            plt.scatter(np.array([optimal]), np.array([val_error]), color='k', zorder=10)

    utils.debug(buffer)
    open(join(DIR, filename), 'w').write(buffer)
    legends = [mpatches.Patch(color=C[S[i]], label='{0} split'.format(int(S[i] * 100))) for i in range(len(S))]
    plt.legend(handles=legends, loc=1)
    plt.grid()
    plt.savefig(join(DIR, photoname))

    if auto < 0:
        plt.show()
    else:
        plt.clf()

