"""
    function average(tt::TensorTrain{V}) where {V}

Evaluates the average of the tensor train approximation over all lattice sites in an efficient
factorized manner.
"""
function average(tt::TCI.AbstractTensorTrain{V}) where {V}
    v = transpose(sum(tt[1], dims=(1, 2))[1, 1, :]) / length(tt[1][1, :, 1])
    for T in tt[2:end]
        v *= sum(T, dims=2)[:, 1, :] / length(T[1, :, 1])
    end
    return only(v)
end

"""
    function weightedsum(tt::TensorTrain{V}, w::Vector{V}) where {V}

Evaluates the weighted sum of the tensor train approximation over all lattice sites in an efficient
factorized manner, where w is the vector of vector of weights which has the same length and the same sizes as tt.
"""
function weightedsum(tt::TCI.AbstractTensorTrain{V}, w::Vector{Vector{V}}) where {V}
    length(tt) == length(w) || throw(DimensionMismatch("The length of the Tensor Train is different from the one of the weight vector ($(length(tt)) and $(length(w)))."))
    size(tt[1])[2] == length(w[1]) || throw(DimensionMismatch("The dimension at site 1 of the Tensor Train is different from the one of the weight vector ($(size(tt[1])[2]) and $(length(w[1])))."))
    v = transpose(sum(tt[1].*w[1]', dims=(1, 2))[1, 1, :])
    for i in 2:length(tt)
        size(tt[i])[2] == length(w[i]) || throw(DimensionMismatch("The dimension at site $(i) of the Tensor Train is different from the one of the weight vector ($(size(tt[i])[2]) and $(length(w[i])))."))
        v *= sum(tt[i].*w[i]', dims=2)[:, 1, :]
    end
    return only(v)
end

function _addtttensor(
    A::Array{V}, B::Array{V};
    factorA=one(V), factorB=one(V),
    lefttensor=false, righttensor=false
) where {V}
    if ndims(A) != ndims(B)
        throw(DimensionMismatch("Elementwise addition only works if both tensors have the same indices, but A and B have different numbers ($(ndims(A)) and $(ndims(B))) of indices."))
    end
    nd = ndims(A)
    offset1 = lefttensor ? 0 : size(A, 1)
    offset3 = righttensor ? 0 : size(A, nd)
    localindices = fill(Colon(), nd - 2)
    C = zeros(V, offset1 + size(B, 1), size(A)[2:nd-1]..., offset3 + size(B, nd))
    C[1:size(A, 1), localindices..., 1:size(A, nd)] = factorA * A
    C[offset1+1:end, localindices..., offset3+1:end] = factorB * B
    return C
end

@doc raw"""
    function add(
        lhs::TCI.AbstractTensorTrain{V}, rhs::TCI.AbstractTensorTrain{V};
        factorlhs=one(V), factorrhs=one(V),
        tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
    ) where {V}

Addition of two tensor trains. If `C = add(A, B)`, then `C(v) ≈ A(v) + B(v)` at each index set `v`. Note that this function increases the bond dimension, i.e. ``\chi_{\text{result}} = \chi_1 + \chi_2`` if the original tensor trains had bond dimensions ``\chi_1`` and ``\chi_2``.

Arguments:
- `lhs`, `rhs`: Tensor trains to be added.
- `factorlhs`, `factorrhs`: Factors to multiply each tensor train by before addition.
- `tolerance`, `maxbonddim`: Parameters to be used for the recompression step.

Returns:
A new `TensorTrain` representing the function `factorlhs * lhs(v) + factorrhs * rhs(v)`.

See also: [`+`](@ref)
"""
function add(
    lhs::TCI.AbstractTensorTrain{V}, rhs::TCI.AbstractTensorTrain{V};
    factorlhs=one(V), factorrhs=one(V),
    tolerance::Float64=0.0, maxbonddim::Int=typemax(Int), normalizeerror::Bool=true
) where {V}
    if length(lhs) != length(rhs)
        throw(DimensionMismatch("Two tensor trains with different length ($(length(lhs)) and $(length(rhs))) cannot be added elementwise."))
    end
    L = length(lhs)
    tt = tensortrain(
        [
            _addtttensor(
                lhs[ell], rhs[ell];
                factorA=((ell == L) ? factorlhs : one(V)),
                factorB=((ell == L) ? factorrhs : one(V)),
                lefttensor=(ell==1),
                righttensor=(ell==L)
            )
            for ell in 1:L
        ]
    )
    TCI.compress!(tt, :SVD; tolerance, maxbonddim, normalizeerror)
    return tt
end

@doc raw"""
    function subtract(
        lhs::TCI.AbstractTensorTrain{V}, rhs::TCI.AbstractTensorTrain{V};
        tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
    )

Subtract two tensor trains `lhs` and `rhs`. See [`add`](@ref).
"""
function subtract(
    lhs::TCI.AbstractTensorTrain{V}, rhs::TCI.AbstractTensorTrain{V};
    tolerance::Float64=0.0, maxbonddim::Int=typemax(Int), normalizeerror::Bool=true
) where {V}
    return add(lhs, rhs; factorrhs=-1 * one(V), tolerance, maxbonddim, normalizeerror)
end

function checkorthogonality(tt::Vector{Array{ValueType, N}}) where {ValueType, N}
    ort = Vector{Symbol}(undef, length(tt))
    for i in 1:length(tt)
        W = tt[i]
        left_check = _contract(permutedims(W, (4,2,3,1,)), W, (2,3,4,),(2,3,1))
        right_check = _contract(W, permutedims(W, (4,2,3,1,)), (2,3,4,),(2,3,1))
        is_left = isapprox(left_check, I, atol=1e-7)
        is_right = isapprox(right_check, I, atol=1e-7)
        ort[i] = if is_left && is_right
            :O # Orthogonal
        elseif is_left
            :L # Left orthogonal
        elseif is_right
            :R # Right orthogonal
        else
            :N # Non orthogonal
        end
    end
    return ort
end
