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
