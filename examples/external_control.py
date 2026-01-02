import queue

_q = queue.Queue(maxsize=1)

def put(ctrl_tuple):
    try:
        while True:
            _q.get_nowait()
    except queue.Empty:
        pass
    _q.put(ctrl_tuple)

def get():
    return _q.get()
