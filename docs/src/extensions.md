# Examples

## Basic Contraction

```julia
using T4AMPOContractions
import T4ATensorCI as TCI

# Create two simple MPOs
A = TCI.TensorTrain{Float64,4}([...])  # Your MPO tensors
B = TCI.TensorTrain{Float64,4}([...])

# Contract using naive algorithm
C = contract(A, B; algorithm=:naive, tolerance=1e-12, maxbonddim=100)
```

## Using Different Algorithms

```julia
# TCI-based contraction
C_tci = contract(A, B; algorithm=:TCI, tolerance=1e-10)

# Zip-up algorithm with SVD
C_zipup = contract(A, B; algorithm=:zipup, method=:SVD, tolerance=1e-12)

# Variational fitting (requires SiteTensorTrain)
using T4ATensorTrain
A_site = SiteTensorTrain{4}(A)
B_site = SiteTensorTrain{4}(B)
C_fit = contract(A_site, B_site; algorithm=:fit, nsweeps=4, tolerance=1e-12)
```

## Distributed Contraction with MPI

```julia
using T4AMPOContractions
import T4ATensorCI as TCI

# Initialize MPI
initializempi()

# Create MPOs
A = TCI.TensorTrain{Float64,4}(...)
B = TCI.TensorTrain{Float64,4}(...)

# Convert to InverseTensorTrain for distributed algorithms
using T4ATensorTrain
A_inv = InverseTensorTrain{4}(A)
B_inv = InverseTensorTrain{4}(B)

# Distributed contraction
C = contract(A_inv, B_inv; 
    algorithm=:distrfit,
    nsweeps=8,
    tolerance=1e-12,
    synchedinput=false,
    synchedoutput=true
)

# Finalize MPI
finalizempi()
```

## Function Application During Contraction

The `:TCI` algorithm supports applying a function elementwise to the contraction result:

```julia
# Apply exponential function during contraction
C = contract(A, B; algorithm=:TCI, f=exp, tolerance=1e-10)
```

## MPI Utilities

For distributed algorithms, you need to initialize and finalize MPI:

```julia
using T4AMPOContractions

# Initialize MPI (mutes non-root processes by default)
T4AMPOContractions.initializempi()

# ... your distributed computation ...

# Finalize MPI
T4AMPOContractions.finalizempi()
```

