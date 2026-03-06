# cython: profile=False
# cython: boundscheck=False, wraparound=False, cdivision=True
cimport cython
cimport numpy as np
import numpy as np

from libc.math cimport sqrt, isnan, NAN
from libcpp.deque cimport deque


cdef class Rolling:
    cdef int window
    cdef deque[double] barv
    cdef int na_count

    def __init__(self, int window):
        self.window = window
        self.na_count = window
        cdef int i
        for i in range(window):
            self.barv.push_back(NAN)

    cdef double update(self, double val):
        return NAN


cdef class Mean(Rolling):
    cdef double vsum

    def __init__(self, int window):
        super(Mean, self).__init__(window)
        self.vsum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        if not isnan(self.barv.front()):
            self.vsum -= self.barv.front()
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
        else:
            self.vsum += val
        return self.vsum / (self.window - self.na_count)


cdef class Slope(Rolling):
    cdef double i_sum
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double xy_sum

    def __init__(self, int window):
        super(Slope, self).__init__(window)
        self.i_sum = 0
        self.x_sum = 0
        self.x2_sum = 0
        self.y_sum = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        cdef double old_val = self.barv.front()
        if not isnan(old_val):
            self.i_sum -= 1
            self.y_sum -= old_val
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
        else:
            self.i_sum += 1
            self.x_sum += self.window
            self.x2_sum += self.window * self.window
            self.y_sum += val
            self.xy_sum += self.window * val
        cdef int count = self.window - self.na_count
        return (count * self.xy_sum - self.x_sum * self.y_sum) / (count * self.x2_sum - self.x_sum * self.x_sum)


cdef class Resi(Rolling):
    cdef double i_sum
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double xy_sum

    def __init__(self, int window):
        super(Resi, self).__init__(window)
        self.i_sum = 0
        self.x_sum = 0
        self.x2_sum = 0
        self.y_sum = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        cdef double old_val = self.barv.front()
        if not isnan(old_val):
            self.i_sum -= 1
            self.y_sum -= old_val
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
        else:
            self.i_sum += 1
            self.x_sum += self.window
            self.x2_sum += self.window * self.window
            self.y_sum += val
            self.xy_sum += self.window * val
        cdef int count = self.window - self.na_count
        cdef double slope = (count * self.xy_sum - self.x_sum * self.y_sum) / (count * self.x2_sum - self.x_sum * self.x_sum)
        cdef double x_mean = self.x_sum / count
        cdef double y_mean = self.y_sum / count
        cdef double intercept = y_mean - slope * x_mean
        return val - (slope * self.window + intercept)


cdef class Rsquare(Rolling):
    cdef double i_sum
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double y2_sum
    cdef double xy_sum

    def __init__(self, int window):
        super(Rsquare, self).__init__(window)
        self.i_sum = 0
        self.x_sum = 0
        self.x2_sum = 0
        self.y_sum = 0
        self.y2_sum = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        cdef double old_val = self.barv.front()
        if not isnan(old_val):
            self.i_sum -= 1
            self.y_sum -= old_val
            self.y2_sum -= old_val * old_val
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
        else:
            self.i_sum += 1
            self.x_sum += self.window
            self.x2_sum += self.window * self.window
            self.y_sum += val
            self.y2_sum += val * val
            self.xy_sum += self.window * val
        cdef int count = self.window - self.na_count
        cdef double rvalue = (count * self.xy_sum - self.x_sum * self.y_sum) / sqrt((count * self.x2_sum - self.x_sum * self.x_sum) * (count * self.y2_sum - self.y_sum * self.y_sum))
        return rvalue * rvalue


cdef np.ndarray[double, ndim=1] rolling(Rolling rolling_obj, np.ndarray a):
    cdef int idx
    cdef int size = len(a)
    cdef np.ndarray[double, ndim=1] result = np.empty(size)
    for idx in range(size):
        result[idx] = rolling_obj.update(a[idx])
    return result


def rolling_mean(np.ndarray a, int window):
    cdef Mean rolling_obj = Mean(window)
    return rolling(rolling_obj, a)


def rolling_slope(np.ndarray a, int window):
    cdef Slope rolling_obj = Slope(window)
    return rolling(rolling_obj, a)


def rolling_rsquare(np.ndarray a, int window):
    cdef Rsquare rolling_obj = Rsquare(window)
    return rolling(rolling_obj, a)


def rolling_resi(np.ndarray a, int window):
    cdef Resi rolling_obj = Resi(window)
    return rolling(rolling_obj, a)
