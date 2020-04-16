from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def flatten(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flatten(k)
