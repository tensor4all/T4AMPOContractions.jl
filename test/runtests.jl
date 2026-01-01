import T4ATensorCI as TCI
import T4AMPOContractions as MPO
using Test
using LinearAlgebra
using Random

# Run Aqua and JET tests when not explicitly skipped
if !haskey(ENV, "SKIP_AQUA_JET")
    using Pkg
    Pkg.add("Aqua")
    Pkg.add("JET")
    include("test_with_aqua.jl")
    include("test_with_jet.jl")
end

include("test_mpi.jl")
include("test_contraction.jl")
include("test_factorize.jl")