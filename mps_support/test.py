import random
import timeit

import torch


print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch built with MPS: {torch.backends.mps.is_built()}')
print(f'MPS available: {torch.backends.mps.is_available()}')

x = torch.ones(50000000, device="mps")
mps_seconds = timeit.timeit(lambda: x * random.randint(0, 100), number=100)

print(
    f'On MPS device: {mps_seconds} seconds'
)

x = torch.ones(50000000, device="cpu")
cpu_seconds = timeit.timeit(lambda: x * random.randint(0, 100), number=100)

print(
    f'On CPU device: {cpu_seconds} seconds'
)

print(
    f'MPS is {cpu_seconds / mps_seconds:.2f}x faster than CPU'
)
