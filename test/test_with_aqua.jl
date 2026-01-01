using Aqua
import T4AMPOContractions as MPO

@testset "Aqua" begin
    Aqua.test_all(MPO; ambiguities = false, unbound_args = false, deps_compat = false)
end
