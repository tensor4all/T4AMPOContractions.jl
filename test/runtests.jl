import T4AMPOContractions as TCI
using Test
using LinearAlgebra

include("test_with_aqua.jl")
include("test_with_jet.jl")
include("test_mpi.jl")
include("test_contraction.jl")
include("test_factorize.jl")