"""Test project for functional coding"""
import cv2
import numpy as np
from toolz import partial, curry
from pipetools import maybe
from functools import reduce
from fn import _, F


@curry
def video_runner(context: int, f):
    @curry
    def get_frame(c):
        code, frame = c.read()
        if code:
            return frame
        else:
            return None

    cap = cv2.VideoCapture(context)
    f(lambda: get_frame(cap))
    cap.release()


@curry
def image_runner(file, f):
    img = cv2.imread(file)
    f(lambda: img)


def do_processing(get_frame):

    # processing routines
    def show_and_pipe(x):
        cv2.imshow('frame', x)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None
        else:
            return x

    def do_auto_canny(img):
        sigma = 0.33
        v = np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(img, lower, upper)

    # processing loop
    def processing():
        while 1:
            res = (maybe | F(lambda img: cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY))
                         | F(lambda img: cv2.flip(img, flipCode=1))
                         | do_auto_canny
                         | show_and_pipe)(get_frame())
            if res is not None:
                yield res
            else:
                break

    reduce(lambda acc, img: [], processing(), [])
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ir = image_runner('images/test.PNG')
    vr = video_runner(0)
    vr(do_processing)
