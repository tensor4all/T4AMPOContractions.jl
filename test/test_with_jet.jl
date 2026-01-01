using JET
import T4AMPOContractions as MPO

@testset "JET" begin
    if VERSION ≥ v"1.10"
        # Use target_modules instead of deprecated target_defined_modules
        JET.test_package(MPO; target_modules=(MPO,))
    end
end
