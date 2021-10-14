from typing import Tuple
import torch

def foo(x, y):
    return x + y

@torch.jit.script
def helper(x, y) -> Tuple[torch.Tensor, torch.Tensor]:
    # x, y = args
    # Call `foo` using parallelism:
    # First, we "fork" off a task. This task will run `foo` with argument `x`
    future = torch.jit.fork(foo, x, y)

    # Call `foo` normally
    x_normal = foo(x, y)
    print(x_normal)

    # Second, we "wait" on the task. Since the task may be running in
    # parallel, we have to "wait" for its result to become available.
    # Notice that by having lines of code between the "fork()" and "wait()"
    # call for a given Future, we can overlap computations so that they
    # run in parallel.
    x_parallel = torch.jit.wait(future)

    return x_normal, x_parallel

def example(x, y):
    return helper(x, y)

print(example(torch.tensor(1), torch.tensor(2)))