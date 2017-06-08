import numpy as np
from threading import Thread, Condition, RLock
from queue import Queue
from sys import stdout
from os import mkdir
from os.path import join, isdir


# Threading

def _debug_do_work():
    global _debug_message_queue
    while True:
        # wait on message
        _debug_message_cond.acquire()
        while _debug_message_queue.empty():
            _debug_message_cond.wait()

        # exit thread
        if _debug_message_thread_exit:
            del _debug_message_queue
            break
        # get message
        msg = _debug_message_queue.get()
        _debug_message_queue.task_done()
        _debug_message_cond.release()

        # write message
        stdout.write(msg)
        stdout.flush()

DEBUG = True
_debug_message_queue = Queue()
_debug_message_lock = RLock()
_debug_message_cond = Condition(_debug_message_lock)
_debug_message_thread_exit = False
_debug_message_thread = Thread(target=_debug_do_work)
_debug_message_thread.start()

# public util functions


def __del__():
    global _debug_message_thread_exit
    _debug_message_cond.acquire()
    _debug_message_thread_exit = True
    _debug_message_cond.notify()
    _debug_message_cond.release()


def debug(msg):
    if DEBUG and not _debug_message_thread_exit:
        _debug_message_cond.acquire()
        _debug_message_queue.put(msg + '\n')
        _debug_message_cond.notify()
        _debug_message_cond.release()


def foldl(f, xs, b):
    b = b if len(xs) == 0 else f(xs[0], b)
    if len(xs) <= 1: return b
    return foldl(f, xs[1:], b)


def purify(s, chars=(' ',)):
    return foldl(lambda a, b: b + ('' if a in chars else a), s, '')


def mkpath(directories):
    path = ''
    for d in directories:
        path = join(path, d)
        if not isdir(path):
            mkdir(path)
    return path


def pack(x, y):
    return np.array([[_x, _y] for _x, _y in zip(x, y)])


def unpack(a):
    a = np.array(a)
    return a[:, 0], a[:, 1]


def split(x, k=2):
    div = round(len(x) / k)
    _x = []
    for i in range(k):
        _x.append(x[div * i:] if i == k - 1 else x[div * i: div * (i + 1)])
    return np.array(_x)


def flatten(x):
    flat = []
    for value in x:
        for v in value:
            flat.append(v)

    return flat


def remove(x, chunk, index):
    return list(x[:index]) + list(x[index + chunk:])


def crop(x, chunk, index):
    return x[index: index + chunk]


def score(data, f):
    s = 0
    for feature, label in data:
        if f(feature) * label < 0:
            s += 1
    return s / len(data)


def running_average(curr, new, n):
    return ((curr * n) + new) / (n + 1)


def list_equal(l1, l2):
    if len(l1) != len(l2):
        return False
    for a, b in zip(l1, l2):
        if a != b:
            return False
    return True


def kfold(data, arg, kernel, k=5):
    X, Y = unpack(data)
    X, Y = split(X, k),  split(Y, k)
    e = 0

    for i in range(k):
        x = flatten(remove(X, 1, i))
        y = flatten(remove(Y, 1, i))
        tx = flatten(crop(X, 1, i))
        ty = flatten(crop(Y, 1, i))

        t = kernel(arg)
        t.fit(x, y)
        _e = score(pack(tx, ty), lambda x: t.predict(np.array([x]).reshape(1, -1)))
        e = running_average(e, _e, i)
    return e
