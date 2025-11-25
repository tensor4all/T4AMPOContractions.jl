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

using T4ATensorCI
import T4ATensorCI: IndexSet, MatrixACA, MatrixCI, MatrixLUCI, BatchEvaluator, forwardsweep, AbstractTensorTrain, TensorCI1, TensorCI2, sitetensors, compress!, replacenothing, LocalIndex, MultiIndex

# Files with MPI additions - these will import from T4ATensorCI and add MPI support
include("abstracttensortrain.jl")
include("factorize.jl")
include("contraction.jl")

# Unique to T4AMPOContractions
include("mpi.jl")
include("paralleltensortrain.jl")

end
