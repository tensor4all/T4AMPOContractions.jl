# Tensor-train representations are implemented in T4ATensorTrain.jl.
# This package keeps the names in its own namespace for backwards compatibility
# (tests and internal code refer to e.g. `MPO.SiteTensorTrain`).

import T4ATensorTrain

# Types
import T4ATensorTrain: VidalTensorTrain, InverseTensorTrain, SiteTensorTrain

# Core API
import T4ATensorTrain: center, partition
import T4ATensorTrain: setcenter!, setpartition!
import T4ATensorTrain: setsitetensor!, settwositetensors!
import T4ATensorTrain: movecenterleft!, movecenterright!, movecenterto!
import T4ATensorTrain: movecenterleft, movecenterright
import T4ATensorTrain: centercanonicalize, centercanonicalize!
import T4ATensorTrain: add, subtract

# Orthogonality helpers
import T4ATensorTrain: isleftorthogonal, isrightorthogonal

# Vidal / inverse accessors
import T4ATensorTrain: singularvalues, singularvalue
import T4ATensorTrain: inversesingularvalues, inversesingularvalue

# Note: T4AMPOContractions defines reshapephysicalleft/right in src/factorize.jl,
# so we intentionally do not import them here to avoid name conflicts.
