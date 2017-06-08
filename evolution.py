import numpy as np
import utils
import pickle
import sklearn.neural_network as nn
import matplotlib.pyplot as plt
from threading import Thread, RLock, Condition, current_thread
from os.path import isdir, join
from random import shuffle, randint, gauss
from timeit import default_timer as timer
from queue import Queue, PriorityQueue


# update num layers, num neurons
def mklayers_big_step(hl):
    mls = hl[0]
    nl = len(hl)
    new_mls = max(int(gauss(mls, 20)), 1)
    new_nl = max(int(gauss(nl, 1)), 1)
    return [new_mls] * new_nl


# update num neurons for individual layers
def mklayers_little_step(hl):
    new_hl = [0] * len(hl)
    for i in range(len(hl)):
        new_hl[i] = int(round(max(gauss(hl[i], 10), 1)))
    return new_hl


DATASET = 'rna'
ODIR = utils.mkpath(('report', DATASET))
utils.debug('~~~ {0} ~~~\nEvolution Strategy\n'.format(DATASET.upper()))

# build constants
A = ['logistic', 'tanh', 'relu', 'identity']
A = A[2:]
MAX_STEPS = 100
MAX_UNCHANGED = 5
BIG = 'big'
LITTLE = 'little'
N = 5

# thread data
pool_size = min(N, 4)
pool = []
elock = RLock()
tlock = RLock()
ecsv = Condition(elock)
tcsv = Condition(tlock)
eq = PriorityQueue(N)
tq = Queue()
kill = False

# neural net variables
activation = A[0]
init_ls = randint(50, 200)
init_nl = randint(3, 10)

# report tracking variables
unchanged_count = 0
mode = BIG
mklayer = {BIG: mklayers_big_step, LITTLE: mklayers_little_step}
exhausted_space = 50

# load data
data = pickle.load(open('bfw_vectorized_images.pkl', 'rb'))
data = utils.pack(data['x'], data['y'])
shuffle(data)
train = data[:int(len(data) * 0.8)]
test = data[int(len(data) * 0.8):]


# thread method
def _do_work_():
    global count
    name = current_thread().getName()

    while True:
        tcsv.acquire()
        while tq.empty():
            tcsv.wait()

            if kill:
                tcsv.release()
                utils.debug('{0} shutting down'.format(name))
                return

        hl = tq.get()
        tcsv.release()
        utils.debug('{0} evaluating {1}'.format(name, hl))

        def mknn(a):
            # oh no a's not used! i'm gonna kill myself
            return nn.MLPClassifier(hidden_layer_sizes=hl, activation=activation)

        e = utils.kfold(train, None, mknn, k=5)
        utils.debug('{0} done'.format(name))
        ecsv.acquire()
        eq.put((e, hl))
        count -= 1
        if count == 0:
            ecsv.notify()
        ecsv.release()


# generate new layers for all N scatters
def mkseeds(hl, N):
    global mode

    layers = [[]] * N
    n, tick = 0, 0
    while n < N:
        layer = mklayer[mode](hl)
        if str(layer) in visited:
            tick += 1
            if tick >= exhausted_space:
                utils.debug('solution space exhausted for {0}'.format(hl))
                tick = 0
                if mode == LITTLE:
                    return None
                else:
                    utils.debug('switching to {0} mode'.format(LITTLE))
                    mode = LITTLE
                    n = 0
        else:
            visited[str(layer)] = True
            tick = 0
            layers[n] = layer
            n += 1

    return layers


def eval_curr_state(chosen, errors, i, display=False):
    # gather up some other useful information
    hl = chosen
    val_err = errors[-1] if len(errors) > 0 else 'n/a'

    x, y = utils.unpack(train)
    y = list(y)
    x = list(x)
    neural = nn.MLPClassifier(hidden_layer_sizes=hl, activation=activation)
    neural.fit(x, y)
    start = timer()
    train_err = utils.score(train, lambda a: neural.predict(np.array(a).reshape((1, -1))))
    end = timer()
    calc_time = (end - start) / len(train)

    x, y = utils.unpack(test)
    nn.MLPClassifier(hidden_layer_sizes=hl, activation=activation)
    y = list(y)
    x = list(x)

    neural.fit(x, y)
    test_err = utils.score(test, lambda a: neural.predict(np.array(a).reshape((1, -1))))

    m = 'activation    = {0}\n' \
        'split         = {1}\n' \
        'arg*          = {2}\n' \
        'calc time     = {3}\n' \
        'val. error    = {4}\n' \
        'train error   = {5}\n' \
        'test error    = {6}\n' \
        'num updates   = {7}\n' \
        'total evals   = {8}\n' \
        .format(activation, 80,
                hl, calc_time, val_err,
                round(train_err, 4),
                round(test_err, 4),
                i, N * i)

    utils.debug(m)

    path = join(ODIR, 'evolution_{0}.txt'.format(activation))
    f = open(path, 'w')
    f.write(m)
    f.close()

    # display scatter path for BIG phase
    x, y = np.array(history)[:, 1], np.array(history)[:, 0]
    plt.scatter(x, y, color='b', alpha=0.4, zorder=1)
    x, y = np.array(optimal)[:, 1], np.array(optimal)[:, 0]
    plt.plot(x, y, color='r', zorder=10)
    plt.xlabel('# layers')
    plt.ylabel('# of neurons')
    plt.title('Neural Network Search Path - {0}'.format(activation))
    plt.savefig(join(ODIR, 'evolution_graph_bigsteps_{0}.png'.format(activation)))
    if display:
        plt.show()
    else:
        plt.clf()

    # display error rate for duration of training
    steps = [i+1 for i in range(len(errors))]
    plt.plot(steps, errors)
    plt.xlabel('updates')
    plt.ylabel('validation error')
    plt.title('Neural Network - {0}'.format(activation))
    plt.savefig(join(ODIR, 'evolution_graph_error_rate_{0}.png'.format(activation)))
    if display:
        plt.show()

# build thread pool
for n in range(pool_size):
    t = Thread(target=_do_work_, name='Thread {0}'.format(n))
    pool.append(t)
    utils.debug('Thread {0} spinning up'.format(n))
    t.start()

# evo strats
i = 0
for activation in A:
    unchanged_count = 0
    i = 0
    visited = {}
    errors = []
    history = [[init_ls, init_nl]]
    optimal = [[init_ls, init_nl]]
    chosen = [init_ls] * init_nl
    visited[str(chosen)] = True
    chosen_err = 1

    # display useless info
    utils.debug('\nactivation        : {0}\n'
                'scatter count     : {1}\n'
                'pool size         : {2}\n'
                'max updates       : {3}\n'
                'no change cutoff  : {4}\n'
                'hidden layers     : {5}'
                .format(activation, N, pool_size, MAX_STEPS, MAX_UNCHANGED, chosen))

    while True:
        if unchanged_count >= MAX_UNCHANGED:
            if mode == BIG:
                mode = LITTLE
                unchanged_count = 0
                utils.debug('switching to {0} mode'.format(LITTLE))
            else:
                break
        if (MAX_STEPS > 0) and (i >= MAX_STEPS):
            break
        i += 1

        seeds = mkseeds(chosen, N)
        if seeds is None:
            break
        utils.debug('\nupdate {0}'.format(i+1))
        tcsv.acquire()
        for s in seeds:
            tq.put(s)
        count = N
        tcsv.notify_all()
        tcsv.release()

        ecsv.acquire()
        while not eq.full():
            ecsv.wait()

        err, hl = eq.get()
        if mode == BIG:
            history.append([hl[0], len(hl)])
            while not eq.empty():
                _, tmp = eq.get()
                ls, nl = tmp[0], len(tmp)
                history.append([ls, nl])
        else:
            eq.queue.clear()
        ecsv.release()

        if err < chosen_err:
            chosen_err = err
            chosen = hl
            errors.append(err)
            utils.debug('new layer -> {0} : {1}'.format(hl, err))
            unchanged_count = 0
            if mode == BIG:
                optimal.append([hl[0], len(hl)])
        else:
            errors.append(chosen_err)
            unchanged_count += 1
            utils.debug('layer unchanged')

    eval_curr_state(list(chosen), errors, i, display=True)

kill = True
tcsv.acquire()
tcsv.notify_all()
tcsv.release()
for thread in pool:
    thread.join()

print('done')

