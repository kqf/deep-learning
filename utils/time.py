import time


def time_function(f, *args):
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic
