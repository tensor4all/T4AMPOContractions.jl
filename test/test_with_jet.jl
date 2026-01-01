using JET
import T4AMPOContractions as MPO

@testset "JET" begin
    if VERSION ≥ v"1.10"
        JET.test_package(MPO; target_defined_modules=true)
    end
end
