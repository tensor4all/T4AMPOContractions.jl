module T4AMPOContractions

using LinearAlgebra
using EllipsisNotation
using BitIntegers
using MPI
using Base.Threads

import QuadGK
# To add a method for rank(tci)
import LinearAlgebra: rank, diag
import LinearAlgebra as LA
# To define equality of IndexSet, and TT addition
import Base: ==, +
# To define iterators and element access for MCI, TCI and TT objects
import Base: isempty, iterate, getindex, lastindex, broadcastable
import Base: length, size, sum
import Random

# Define constants before importing T4ATensorCI to avoid conflicts
const LocalIndex = Int
const MultiIndex = Vector{LocalIndex}

# Import T4ATensorCI and re-export its public API
using T4ATensorCI
# Import internal types that are needed by included files (but not types defined in abstracttensortrain.jl)
import T4ATensorCI: IndexSet, MatrixACA, MatrixCI, MatrixLUCI, BatchEvaluator
# Re-export all public functions and types from T4ATensorCI
export crossinterpolate1, crossinterpolate2, optfirstpivot
export tensortrain, TensorTrain, sitedims, evaluate
# Note: contract is extended below in contraction.jl

# Files with MPI additions - these will import from T4ATensorCI and add MPI support
include("sweepstrategies.jl")
include("matrixlu.jl")
include("abstracttensortrain.jl")
include("tensortrain.jl")
include("tensorci1.jl")
include("tensorci2.jl")
include("integration.jl")
include("contraction.jl")

# Unique to T4AMPOContractions
include("mpi.jl")

end
