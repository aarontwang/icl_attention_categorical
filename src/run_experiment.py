import os
import jax
import eval
from util import *

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".01"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

if __name__ == '__main__':
    print(jax.default_backend())

    kernels = ["linear", "exp", "rbf", "laplacian", "softmax"]

    for kernel in kernels:
        eval.run_experiment("results_gd_plus/layers_2/c_size_100",
                            5,
                            2,
                            False,
                            "random_grid",
                            kernel,
                            20,
                            False,
                            False,
                            1,
                            1,
                            100,
                            5,
                            5,
                            False,
                            1,
                            1,
                            True,
                            False,
                            True,
                            False,
                            False,
                            False)
