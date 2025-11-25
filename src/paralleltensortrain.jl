"""
    struct ParallelTensorTrain{ValueType}

Represents a tensor train shared among different processors, also known as MPS. This tensor doesn't have to be a legal tensor train (i.e. matching bonds), it is a mere collection of sites.
"""
mutable struct ParallelTensorTrain{ValueType,N} <: AbstractTensorTrain{ValueType}
    sitetensors::Vector{Array{ValueType,N}}

    function ParallelTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}) where {ValueType,N}
        new{ValueType,N}(sitetensors)
    end
end

function Base.show(io::IO, obj::ParallelTensorTrain{V,N}) where {V,N}
    print(
        io,
        "$(typeof(obj)) of rank $(maximum(linkdims(obj)))"
    )
end

function ParallelTensorTrain{V2,N}(tt::TensorTrain{V})::TensorTrain{V2,N} where {V,V2,N}
    return TensorTrain{V2,N}(Array{V2}.(sitetensors(tt)))
end

"""
    function ParallelTensorTrain(sitetensors::Vector{Array{V, 3}}) where {V}

Create a tensor train out of a vector of tensors. Each tensor should have links to the
previous and next tensor as dimension 1 and 3, respectively; the local index ("physical
index" for MPS in physics) is dimension 2.
"""
function ParallelTensorTrain(sitetensors::AbstractVector{<:AbstractArray{V,N}}) where {V,N}
    return ParallelTensorTrain{V,N}(sitetensors)
end

"""
    function ParallelTensorTrain(tci::AbstractTensorTrain{V}) where {V}

Convert a tensor-train-like object into a tensor train. This includes TCI1 and TCI2 objects.

See also: [`TensorCI1`](@ref), [`TensorCI2`](@ref).
"""
function ParallelTensorTrain(tci::AbstractTensorTrain{V})::ParallelTensorTrain{V,3} where {V}
    return ParallelTensorTrain{V,3}(sitetensors(tci))
end

"""
    function ParallelTensorTrain{V2,N}(tci::AbstractTensorTrain{V}) where {V,V2,N}

Convert a tensor-train-like object into a tensor train.

Arguments:
- `tt::AbstractTensorTrain{V}`: a tensor-train-like object.
- `localdims`: a vector of local dimensions for each tensor in the tensor train. A each element
  of `localdims` should be an array-like object of `N-2` integers.
"""
function ParallelTensorTrain{V2,N}(tt::AbstractTensorTrain{V}, localdims)::ParallelTensorTrain{V2,N} where {V,V2,N}
    for d in localdims
        length(d) == N - 2 || error("Each element of localdims be a list of N-2 integers.")
    end
    for n in 1:length(tt)
        prod(size(tt[n])[2:end-1]) == prod(localdims[n]) || error("The local dimensions at n=$n must match the tensor sizes.")
    end
    return ParallelTensorTrain{V2,N}(
        [reshape(Array{V2}(t), size(t, 1), localdims[n]..., size(t)[end]) for (n, t) in enumerate(sitetensors(tt))])
end

function ParallelTensorTrain{N}(tt::AbstractTensorTrain{V}, localdims)::ParallelTensorTrain{V,N} where {V,N}
    return ParallelTensorTrain{V,N}(tt, localdims)
end

function paralleltensortrain(tci)
    return ParallelTensorTrain(tci)
end
