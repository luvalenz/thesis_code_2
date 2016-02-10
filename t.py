__author__ = 'lucas'
import numpy as np


class A:

    def __init__(self, attr1):
        self.attr1 = attr1

class B:
    def __init__(self, A):
        self.attr1 = A.attr1

if __name__ == '__main__':
    a = A(np.array([1]))
    b = B(a)
    a.attr1 = 2
    print b.attr1
    print a.attr1