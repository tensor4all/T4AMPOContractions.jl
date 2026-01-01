import T4ATensorCI as TCI
import T4AMPOContractions as MPO
using Test
using LinearAlgebra
using Random

# Skip Aqua and JET tests in CI to avoid Julia version dependency issues
if get(ENV, "CI", "false") == "false"
    include("test_with_aqua.jl")
    include("test_with_jet.jl")
end

include("test_mpi.jl")
include("test_contraction.jl")
include("test_factorize.jl")