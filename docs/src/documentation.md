# Documentation

Documentation of all types and methods in module [T4AMPOContractions](https://github.com/tensor4all/T4AMPOContractions.jl).

## Matrix approximation


### Matrix cross interpolation (MCI)
```@autodocs
Modules = [T4AMPOContractions]
Pages = ["matrixci.jl"]
```

### Adaptive cross approximation (ACA)
```@autodocs
Modules = [T4AMPOContractions]
Pages = ["matrixaca.jl"]
```

### Rank-revealing LU decomposition (rrLU)
```@autodocs
Modules = [T4AMPOContractions]
Pages = ["matrixlu.jl", "matrixluci.jl"]
```

## Tensor trains and tensor cross Interpolation

### Tensor train (TT)
```@autodocs
Modules = [T4AMPOContractions]
Pages = ["abstracttensortrain.jl", "tensortrain.jl", "contraction.jl"]
```

### Tensor cross interpolation (TCI)
Note: In most cases, it is advantageous to use [`T4AMPOContractions.TensorCI2`](@ref) instead.
```@autodocs
Modules = [T4AMPOContractions]
Pages = ["tensorci1.jl", "indexset.jl", "sweepstrategies.jl"]
```

### Tensor cross interpolation 2 (TCI2)
```@autodocs
Modules = [T4AMPOContractions]
Pages = ["tensorci2.jl", "globalpivotfinder.jl"]
```

### Integration
```@autodocs
Modules = [T4AMPOContractions]
Pages = ["integration.jl"]
```

## Helpers and utility methods
```@autodocs
Modules = [T4AMPOContractions]
Pages = ["cachedfunction.jl", "batcheval.jl", "util.jl", "globalsearch.jl"]
```

## Parallel utility
```@autodocs
Modules = [T4AMPOContractions]
Pages = ["mpi.jl"]
```