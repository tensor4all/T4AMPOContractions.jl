# T4AMPOContractions

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensor4all.github.io/T4AMPOContractions.jl/dev)
[![CI](https://github.com/tensor4all/T4AMPOContractions.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/tensor4all/T4AMPOContractions.jl/actions/workflows/CI.yml)

The [T4AMPOContractions module](https://github.com/tensor4all/T4AMPOContractions.jl) implements MPO (Matrix Product Operator) contraction algorithms for tensor networks.

## Installation

This module can be installed by typing the following in a Julia REPL:
```julia
using Pkg; Pkg.add("T4AMPOContractions")
```

## Usage

*This section only contains the bare minimum to get you started. An example with more explanation can be found in the [user manual](https://tensor4all.github.io/T4AMPOContractions.jl/dev).*

Given two MPOs (Matrix Product Operators), the function `contract` will compute their contraction. For example:
```julia
import T4AMPOContractions as MPOC
f(v) = 1/(1 + v' * v)
# There are 8 tensor indices, each with values 1...10
localdims = fill(10, 8)
tolerance = 1e-8
tci, ranks, errors = TCI.crossinterpolate2(Float64, f, localdims; tolerance=tolerance)
```
Note:
- `f` is defined as a function that takes a single `Vector` of integers.
- The return type of `f` (`Float64` in this case) must be stated explicitly in the call to `crossinterpolate2`.

The resulting `TensorCI2` object can be further manipulated, see [user manual](https://tensor4all.github.io/T4AMPOContractions.jl/dev).
To evaluate the TCI interpolation, simply call your `TensorCI2` object like you would call the original function:
```julia
originalvalue = f([1, 2, 3, 4, 5, 6, 7, 8])
interpolatedvalue = tci([1, 2, 3, 4, 5, 6, 7, 8])
```
The sum of all function values on the lattice can be obtained very efficiently from a tensor train:
```julia
sumvalue = sum(tci)
```

## Online user manual
An example with more explanation can be found in the [user manual](https://tensor4all.github.io/T4AMPOContractions.jl/dev).

## Related modules

### [TCIITensorConversion.jl](https://github.com/tensor4all/tciitensorconversion.jl)
A small helper module for easy conversion of `TensorCI1`, `TensorCI2` and `TensorTrain` objects into ITensors `MPS` objects. This should be helpful for those integrating TCI into a larger tensor network algorithm.
For this conversion, simply call the `MPS` constructor on the object:
```julia
mps = MPS(tci)
```

### [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl)
A module that implements the *quantics representation* and combines it with TCI for exponentially efficient interpolation of functions with scale separation.

## Contributions

- If you are having have technical trouble, feel free to contact me directly.
- Feature requests and bug reports are always welcome, feel free to open an [issue](https://github.com/tensor4all/T4AMPOContractions.jl/-/issues) for those.
- If you have implemented something that might be useful for others, we'd appreciate a [merge request](https://github.com/tensor4all/T4AMPOContractions.jl/-/merge_requests)!

## Authors

This project is maintained by
- Marc K. Ritter @marc_ritter
- Hiroshi Shinaoka @h.shinaoka

For their contributions to this library's code, we thank
- Satoshi Terasaki @terasakisatoshi

---

## References

- Y. Núñez Fernández, M. Jeannin, P. T. Dumitrescu, T. Kloss, J. Kaye, O. Parcollet, and X. Waintal, *Learning Feynman Diagrams with Tensor Trains*, [Phys. Rev. X 12, 041018 (2022)](https://link.aps.org/doi/10.1103/PhysRevX.12.041018).
(arxiv link: [arXiv:2207.06135](http://arxiv.org/abs/2207.06135))
- I. V. Oseledets, Tensor-Train Decomposition, [SIAM J. Sci. Comput. 33, 2295 (2011)](https://epubs.siam.org/doi/10.1137/090752286).
