import T4ATensorCI as TCI
import T4AMPOContractions as MPO
using Test
using LinearAlgebra
using Random

# Run Aqua and JET tests only on Julia 1.11+ to avoid version dependency issues
if VERSION >= v"1.11"
    using Pkg
    Pkg.add("Aqua")
    Pkg.add("JET")
    include("test_with_aqua.jl")
    include("test_with_jet.jl")
end

include("test_mpi.jl")
include("test_contraction.jl")
include("test_factorize.jl")