# MPS support for PyTorch

With the latest PyTorch nightly build, the toy example here is around 28x faster on the MPS device than on the CPU.

```bash
$ python -m mps_support.test
PyTorch version: 2.1.0.dev20230429
PyTorch built with MPS: True
MPS available: True
On MPS device: 0.12888245799695142 seconds
On CPU device: 3.577935791996424 seconds
MPS is 27.76x faster than CPU
```
