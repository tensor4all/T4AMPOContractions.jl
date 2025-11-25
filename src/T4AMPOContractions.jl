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
#const LocalIndex = Int
#const MultiIndex = Vector{LocalIndex}

using T4ATensorCI
import T4ATensorCI: IndexSet, MatrixACA, MatrixCI, MatrixLUCI, BatchEvaluator, forwardsweep, AbstractTensorTrain, TensorCI1, TensorCI2, sitetensors, compress!, replacenothing, LocalIndex, MultiIndex

# Import T4ATensorCI and re-export its public API
# Import internal types that are needed by included files (but not types defined in abstracttensortrain.jl)
# Re-export all public functions and types from T4ATensorCI
# export crossinterpolate1, crossinterpolate2, optfirstpivot
# export tensortrain, TensorTrain, sitedims, evaluate
# Note: contract is extended below in contraction.jl

# Files with MPI additions - these will import from T4ATensorCI and add MPI support
include("abstracttensortrain.jl")
include("tensortrain.jl")
include("contraction.jl")

# Unique to T4AMPOContractions
include("mpi.jl")
include("paralleltensortrain.jl")

end
