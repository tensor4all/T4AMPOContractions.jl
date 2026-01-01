import T4AMPOContractions as TCI
using Test
using LinearAlgebra

# Temporarily skip Aqua and JET tests if they are not available
try
    include("test_with_aqua.jl")
catch e
    @warn "Skipping Aqua tests: $e"
end
try
    include("test_with_jet.jl")
catch e
    @warn "Skipping JET tests: $e"
end
include("test_util.jl")
include("test_sweepstrategies.jl")
include("test_indexset.jl")
include("test_cachedfunction.jl")
include("test_matrixci.jl")
include("test_matrixaca.jl")
include("test_matrixlu.jl")
include("test_matrixluci.jl")
include("test_batcheval.jl")
include("test_cachedtensortrain.jl")
include("test_tensorci1.jl")
include("test_tensorci2.jl")
include("test_mpi.jl")
include("test_tensortrain.jl")
include("test_contraction.jl")
include("test_conversion.jl")
include("test_integration.jl")
include("test_globalsearch.jl")

# TCIITensorConversion extension
include("test_TCIITensorConversion.jl")
