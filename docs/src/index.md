```@meta
CurrentModule = T4AMPOContractions
```

# T4AMPOContractions.jl

[T4AMPOContractions.jl](https://github.com/tensor4all/T4AMPOContractions.jl) implements efficient algorithms for contracting Matrix Product Operators (MPOs) represented as tensor trains.

## Installation

```julia
using Pkg
Pkg.add("T4AMPOContractions")
```

## Quick Start

The main function is [`contract`](@ref), which contracts two MPOs:

```julia
using T4AMPOContractions
import T4ATensorCI as TCI

# Create two MPOs (as TensorTrain{4} objects)
A = TCI.TensorTrain{Float64,4}(...)  # Your first MPO
B = TCI.TensorTrain{Float64,4}(...)  # Your second MPO

# Contract them using the naive algorithm
C = contract(A, B; algorithm=:naive, tolerance=1e-12)
```

## Algorithms

The package provides several algorithms for MPO contraction:

- **`:naive`**: Direct tensor contraction followed by SVD compression
- **`:TCI`**: Tensor Cross Interpolation-based contraction
- **`:zipup`**: On-the-fly factorization during contraction
- **`:fit`**: Variational fitting algorithm (requires `SiteTensorTrain` or `InverseTensorTrain`)
- **`:distrfit`**: Distributed variational fitting with MPI support

## Distributed Computing

For large-scale computations, the package supports MPI-based distributed algorithms:

```julia
using T4AMPOContractions

# Initialize MPI
initializempi()

# Use distributed algorithm
C = contract(A, B; algorithm=:distrfit, subcomm=nothing)

# Finalize MPI
finalizempi()
```

## Related Packages

- [T4ATensorCI.jl](https://github.com/tensor4all/T4ATensorCI.jl): Core tensor train and TCI functionality
- [T4ATensorTrain.jl](https://github.com/tensor4all/T4ATensorTrain.jl): Tensor train data structures

## Documentation

- [API Reference](@ref) - Complete function and type documentation
- [Examples](@ref) - Usage examples and tutorials
