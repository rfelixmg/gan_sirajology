import numpy as np
from timeit import default_timer as timer
from numbapro import vectorize
from numbapro import cuda
import gc

@vectorize(["float32(float32, float32)"], target='gpu')
def VectorAdd(a, b):
    return a + b

@cuda.jit('void(float32[:], float32[:], float32[:])')
def VectorAddCuda(a, b, result):
    i = cuda.grid(1)
    result[i] = a[i]+b[i]

def main():

    N = 320000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)

    start = timer()
    C = VectorAdd(A, B)
    #VectorAddCuda(A, B, C)
    vectoradd_time = timer() - start

    print ("C[:5] = ", str(C[:5]))
    print ("C[-5:] = ", str(C[-5:]))

    print("VectorAdd took %f seconds", vectoradd_time)

if __name__ == '__main__':
    gc.collect()
    main()

#Old version - not GPU optimized
# import numpy as np
# from timeit import default_timer as timer
#
# def VectorAdd(a, b, c):
#     for i in xrange(a.size):
#         c[i] = a[i] + b[i]
#
# def main():
#
#     N = 320000000
#
#     A = np.ones(N, dtype=np.float32)
#     B = np.ones(N, dtype=np.float32)
#     C = np.zeros(N, dtype=np.float32)
#
#     start = timer()
#     VectorAdd(A, B, C)
#     vectoradd_time = timer() - start
#
#     print ("C[:5] = ", str(C[:5]))
#     print ("C[-5:] = ", str(C[-5:]))
#
#     print("VectorAdd took %f seconds", vectoradd_time)
#
# if __name__ == '__main__':
#     main()