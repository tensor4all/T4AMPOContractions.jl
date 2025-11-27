# VIDAL

mutable struct VidalTensorTrain{ValueType,N} <: TCI.AbstractTensorTrain{ValueType}
    sitetensors::Vector{Array{ValueType,N}}
    singularvalues::Vector{Matrix{ValueType}}
    partition::UnitRange{Int}

    function VidalTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, singularvalues::AbstractVector{<:AbstractMatrix{ValueType}}, partition::AbstractRange{Integer}) where {ValueType,N}
        n = length(sitetensors)
        step(partition) == 1 || throw(ArgumentError("partition must be a contiguous range (step 1)"))
        first(partition) >= 1 && last(partition) <= n || throw(ArgumentError("All partition indices must be between 1 and $n"))
        
        for i in first(partition):last(partition)-1
            if (last(size(sitetensors[i])) != size(sitetensors[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        for i in first(partition)+1:last(partition)-1
            if !isrightorthogonal(_contract(sitetensors[i], singularvalues[i], (4,), (1,)))
                throw(ArgumentError(
                    "Error: contracting the tensor at $i with the singular value at $i does not lead to a right-orthogonal tensor."
                ))
            end
            if !isleftorthogonal(_contract(singularvalues[i-1], sitetensors[i], (2,), (1,)))
                throw(ArgumentError(
                    "Error: contracting the singular value at $(i-1) with the tensor at $i does not lead to a left-orthogonal tensor."
                ))
            end
        end
        new{ValueType,N}(sitetensors, singularvalues, partition)
    end
end

function Base.show(io::IO, obj::VidalTensorTrain{ValueType,N}) where {ValueType,N}
    print(
        io,
        "$(typeof(obj)) of rank $(maximum(linkdims(obj)))"
    )
end

# In construction from sitetensors, VidalTensor needs the whole tensor, not partition
function VidalTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, partition::AbstractRange{<:Integer})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    n = length(tt)
    singularvalues = Vector{Matrix{ValueType}}(undef, n-1)

    for i in 1:n-1
        Q, R = qr(reshapephysicalleft(TCI.sitetensor(tt,i)))
        sitetensors[i] = reshape(Matrix(Q), size(TCI.sitetensor(tt,i))...)
        sitetensors[i+1] = _contract(Matrix(R), TCI.sitetensor(tt,i+1), (2,), (1,))
    end

    for i in n:-1:2
        left, diamond, right, _, _ = _factorize(
            reshapephysicalright(sitetensors[i]),
            :SVD; tolerance=0.0, maxbonddim=first(size(sitetensors[i])), diamond=true
        )

        singularvalues[i-1] = Diagonal(diamond)

        sitetensors[i] = reshape(right, size(sitetensors[i])...)
        sitetensors[i-1] = _contract(sitetensors[i-1], left*singularvalues[i-1], (4,), (1,))
    end

    for i in 1:n-1
        sitetensors[i] = _contract(sitetensors[i], Diagonal(diag(singularvalues[i]).^-1), (4,), (1,))
    end
    return VidalTensorTrain{ValueType, N}(sitetensors, singularvalues, partition)
end

function singularvalues(tt::VidalTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.singularvalues
end

function partition(tt::VidalTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.partition
end


function VidalTensorTrain{ValueType,N}(tt::TCI.AbstractTensorTrain{ValueType}, partition::AbstractRange{<:Integer})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    return VidalTensorTrain{ValueType,N}(TCI.sitetensors(tt), partition)
end

function VidalTensorTrain{ValueType,N}(tt::TCI.AbstractTensorTrain{ValueType})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    return VidalTensorTrain{ValueType, N}(TCI.sitetensors(tt), 1:length(TCI.sitetensors(tt)))
end

function VidalTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    return VidalTensorTrain{ValueType, N}(sitetensors, 1:length(sitetensors))
end

function VidalTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, singularvalues::AbstractVector{<:AbstractMatrix{ValueType}})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    return VidalTensorTrain{ValueType,N}(sitetensors,singularvalues)
end

function VidalTensorTrain{ValueType2,N}(tt::VidalTensorTrain{ValueType1,N})::VidalTensorTrain{ValueType2,N} where {ValueType1,ValueType2,N}
    return VidalTensorTrain{ValueType2,N}(Array{ValueType2}.(TCI.sitetensors(tt)), Array{ValueType2}.(TCI.singularvalues(tt)))
end

function VidalTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}) where {ValueType,N}
    return VidalTensorTrain{ValueType,N}(sitetensors)
end

function VidalTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return VidalTensorTrain{ValueType,N}(sitetensors, partition)
end

function VidalTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, singularvalues::AbstractArray{<:AbstractMatrix{ValueType}}) where {ValueType,N}
    return VidalTensorTrain{ValueType,N}(sitetensors, singularvalues)
end

function VidalTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, singularvalues::AbstractArray{<:AbstractMatrix{ValueType}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return VidalTensorTrain{ValueType,N}(sitetensors, singularvalues, partition)
end

function VidalTensorTrain{ValueType2,N}(tt::TCI.AbstractTensorTrain{ValueType1}, localdims)::VidalTensorTrain{ValueType2,N} where {ValueType1,ValueType2,N}
    for d in localdims
        length(d) == N - 2 || error("Each element of localdims be a list of N-2 integers.")
    end
    for n in 1:length(tt)
        prod(size(tt[n])[2:end-1]) == prod(localdims[n]) || error("The local dimensions at n=$n must match the tensor sizes.")
    end
    return VidalTensorTrain{ValueType2,N}(
        [reshape(Array{ValueType2}(t), size(t, 1), localdims[n]..., size(t)[end]) for (n, t) in enumerate(sitetensors(tt))])
end
function VidalTensorTrain{N}(tt::TCI.AbstractTensorTrain{ValueType}, localdims)::VidalTensorTrain{ValueType,N} where {ValueType,N}
    return VidalTensorTrain{ValueType,N}(tt, localdims)
end

function vidaltensortrain(a)
    return VidalTensorTrain(a)
end

function vidaltensortrain(a, b)
    return VidalTensorTrain(a, b)
end

function vidaltensortrain(a,b,c)
    return VidalTensorTrain(a,b,c)
end

# INVERSE

mutable struct InverseTensorTrain{ValueType,N} <: TCI.AbstractTensorTrain{ValueType}
    sitetensors::Vector{Array{ValueType,N}}
    inversesingularvalues::Vector{Matrix{ValueType}}
    partition::UnitRange{Int}

    function InverseTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, inversesingularvalues::AbstractVector{<:AbstractMatrix{ValueType}}, partition::AbstractRange{Integer}) where {ValueType,N}
        n = length(sitetensors)
        step(partition) == 1 || throw(ArgumentError("partition must be a contiguous range (step 1)"))
        first(partition) >= 1 && last(partition) <= n || throw(ArgumentError("All partition indices must be between 1 and $n"))

        for i in first(partition):last(partition)-1
            if (last(size(sitetensors[i])) != size(sitetensors[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        for i in first(partition):last(partition)-1
            if !isleftorthogonal(_contract(sitetensors[i], inversesingularvalues[i], (4,), (1,)))
                throw(ArgumentError(
                    "Error: contracting the tensor at $i with the singular value at $i does not lead to a left-orthogonal tensor."
                ))
            end
            if !isrightorthogonal(_contract(inversesingularvalues[i], sitetensors[i+1], (2,), (1,)))
                throw(ArgumentError(
                    "Error: contracting the singular value at $i with the tensor at $(i+1) does not lead to a right-orthogonal tensor."
                ))
            end
        end

        new{ValueType,N}(sitetensors, inversesingularvalues, partition)
    end

end

function Base.show(io::IO, obj::InverseTensorTrain{ValueType,N}) where {ValueType,N}
    print(
        io,
        "$(typeof(obj)) of rank $(maximum(linkdims(obj)))"
    )
end

function InverseTensorTrain{ValueType,N}(tt::TCI.AbstractTensorTrain{ValueType}, partition::AbstractRange{<:Integer})::InverseTensorTrain{ValueType,N} where {ValueType, N}
    if !isa(tt, VidalTensorTrain{ValueType,N})
        tt = VidalTensorTrain{ValueType,N}(tt, partition) # Convert with partition
    end
    n = length(tt)
    sitetensors = Vector{Array{ValueType, 4}}(undef, n)
    inversesingularvalues = Vector{Matrix{ValueType}}(undef, n-1)

    if first(partition) == 1
        sitetensors[1] = _contract(tt.sitetensors[first(partition)], tt.singularvalues[first(partition)], (4,), (1,))
    end
    for i in max(2, first(partition)):min(n-1, last(partition))
        sitetensors[i] = _contract(tt.singularvalues[i-1], tt.sitetensors[i], (2,), (1,))
        sitetensors[i] = _contract(sitetensors[i], tt.singularvalues[i], (4,), (1,))
    end
    if last(partition) == n
        sitetensors[n] = _contract(tt.singularvalues[n-1], tt.sitetensors[n], (2,), (1,))
    end

    for i in first(partition):last(partition)-1
        inversesingularvalues[i] = Diagonal(diag(tt.singularvalues[i]).^-1)
    end
    return InverseTensorTrain{ValueType,N}(sitetensors, inversesingularvalues, partition)
end

function inversesingularvalues(tt::InverseTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.inversesingularvalues
end

function partition(tt::InverseTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.partition
end

function InverseTensorTrain{ValueType,N}(tt::TCI.AbstractTensorTrain{ValueType})::InverseTensorTrain{ValueType,N} where {ValueType, N}
    n = length(tt)
    return InverseTensorTrain{ValueType, N}(tt, 1:n)
end

function InverseTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}})::InverseTensorTrain{ValueType,N} where {ValueType, N}
    n = length(sitetensors)
    return InverseTensorTrain{ValueType, N}(sitetensors, 1:n)
end

function InverseTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, inversesingularvalues::AbstractVector{<:AbstractMatrix{ValueType}})::InverseTensorTrain{ValueType,N} where {ValueType, N}
    n = length(sitetensors)
    return InverseTensorTrain{ValueType,N}(sitetensors, inversesingularvalues, 1:n)
end

function InverseTensorTrain{ValueType2,N}(tt::InverseTensorTrain{ValueType1,N})::InverseTensorTrain{ValueType2,N} where {ValueType1,ValueType2,N}
   return InverseTensorTrain{ValueType2,N}(Array{ValueType2}.(TCI.sitetensors(tt)), Array{ValueType2}.(TCI.inversesingularvalues(tt)), tt.partition)
end

function InverseTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}) where {ValueType,N}
    return InverseTensorTrain{ValueType,N}(sitetensors)
end

function InverseTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return InverseTensorTrain{ValueType,N}(sitetensors, partition)
end

function InverseTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, inversesingularvalues::AbstractArray{<:AbstractMatrix{ValueType}}) where {ValueType,N}
    return InverseTensorTrain{ValueType,N}(sitetensors, inversesingularvalues)
end

function InverseTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, inversesingularvalues::AbstractArray{<:AbstractMatrix{ValueType}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return InverseTensorTrain{ValueType,N}(sitetensors, inversesingularvalues, partition)
end


function InverseTensorTrain{ValueType2,N}(tt::TCI.AbstractTensorTrain{ValueType1}, localdims)::InverseTensorTrain{ValueType2,N} where {ValueType1,ValueType2,N}
    for d in localdims
        length(d) == N - 2 || error("Each element of localdims be a list of N-2 integers.")
    end
    for n in 1:length(tt)
        prod(size(tt[n])[2:end-1]) == prod(localdims[n]) || error("The local dimensions at n=$n must match the tensor sizes.")
    end
    return InverseTensorTrain{ValueType2,N}(
        [reshape(Array{ValueType2}(t), size(t, 1), localdims[n]..., size(t)[end]) for (n, t) in enumerate(sitetensors(tt))])
end


function InverseTensorTrain{N}(tt::TCI.AbstractTensorTrain{ValueType}, localdims)::InverseTensorTrain{ValueType,N} where {ValueType,N}
    return InverseTensorTrain{ValueType,N}(tt, localdims)
end

function inversetensortrain(a)
    return InverseTensorTrain(a)
end

function inversetensortrain(a, b)
    return InverseTensorTrain(a, b)
end

function inversetensortrain(a, b, c)
    return InverseTensorTrain(a, b, c)
end


# SITE
mutable struct SiteTensorTrain{ValueType,N} <: TCI.AbstractTensorTrain{ValueType}
    sitetensors::Vector{Array{ValueType,N}}
    center::Int
    partition::UnitRange{Int}

    function SiteTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, center::Int, partition::AbstractRange{Integer}) where {ValueType,N}
        n = length(sitetensors)
        step(partition) == 1 || throw(ArgumentError("partition must be a contiguous range (step 1)"))
        first(partition) >= 1 && last(partition) <= n || throw(ArgumentError("All partition indices must be between 1 and $n"))
        center >= first(partition) && center <= last(partition) || throw(ArgumentError("center ($center) must lie within partition $partition"))

        for i in first(partition):last(partition)-1
            if (last(size(sitetensors[i])) != size(sitetensors[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        all_left_orth = true
        all_right_orth = true
        
        for i in first(partition):center-1
            if !isleftorthogonal(sitetensors[i])
                all_left_orth = false
            end
        end
        
        for i in center+1:last(partition)
            if !isrightorthogonal(sitetensors[i])
                all_right_orth = false
            end
        end
        
        if !(all_left_orth && all_right_orth)
            sitetensors = centercanonicalize(sitetensors, center)
        end

        new{ValueType,N}(sitetensors, center, partition)
    end
end

function SiteTensorTrain{ValueType,N}(tt::TCI.AbstractTensorTrain{ValueType}, center::Int, partition::AbstractRange{<:Integer})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    return SiteTensorTrain{ValueType,N}(TCI.sitetensors(tt), center, partition)
end

function Base.show(io::IO, obj::SiteTensorTrain{ValueType,N}) where {ValueType,N}
    print(io, "$(typeof(obj)) of rank $(maximum(linkdims(obj))) centered at $(obj.center)")
end

function center(tt::SiteTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.center
end

function partition(tt::SiteTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.partition
end

function SiteTensorTrain{ValueType,N}(tt::TCI.AbstractTensorTrain{ValueType}, center::Int)::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(tt)
    return SiteTensorTrain{ValueType,N}(tt, center, 1:n)
end

function SiteTensorTrain{ValueType,N}(tt::TCI.AbstractTensorTrain{ValueType})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(tt)
    return SiteTensorTrain{ValueType,N}(tt, 1, 1:n)
end

function SiteTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, center::Int)::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(sitetensors)
    return SiteTensorTrain{ValueType,N}(sitetensors, center, 1:n)
end

function SiteTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(sitetensors)
    return SiteTensorTrain{ValueType,N}(sitetensors, 1, 1:n)
end

# Default constructor: center at 1
function SiteTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}) where {ValueType,N}
    return SiteTensorTrain{ValueType,N}(sitetensors)
end

function SiteTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, center::Int) where {ValueType,N}
    return SiteTensorTrain{ValueType,N}(sitetensors, center)
end

function SiteTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, center::Int, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return SiteTensorTrain{ValueType,N}(sitetensors, center, partition)
end

# Construct from an AbstractTensorTrain, default center = 1
function SiteTensorTrain(tt::TCI.AbstractTensorTrain{ValueType}) where {ValueType}
    return SiteTensorTrain{ValueType, ndims(TCI.sitetensors(tt)[1])}(tt)
end

function SiteTensorTrain(tt::TCI.AbstractTensorTrain{ValueType}, center::Int) where {ValueType}
    return SiteTensorTrain{ValueType, ndims(TCI.sitetensors(tt)[1])}(tt, center)
end

function SiteTensorTrain(tt::TCI.AbstractTensorTrain{ValueType}, center::Int, partition::AbstractRange{<:Integer}) where {ValueType}
    return SiteTensorTrain{ValueType, ndims(TCI.sitetensors(tt)[1])}(tt, center, partition)
end

# Type conversion: change element type of a SiteTensorTrain
function SiteTensorTrain{ValueType2,N}(tt::SiteTensorTrain{ValueType1,N})::SiteTensorTrain{ValueType2,N} where {ValueType1,ValueType2,N}
    return SiteTensorTrain{ValueType2,N}(Array{ValueType2}.(TCI.sitetensors(tt)), center(tt), partition(tt))
end

# Construct from an AbstractTensorTrain and reshape according to localdims
function SiteTensorTrain{ValueType2,N}(tt::TCI.AbstractTensorTrain{ValueType1}, localdims)::SiteTensorTrain{ValueType2,N} where {ValueType1,ValueType2,N}
    for d in localdims
        length(d) == N - 2 || error("Each element of localdims be a list of N-2 integers.")
    end
    for n in 1:length(tt)
        prod(size(tt[n])[2:end-1]) == prod(localdims[n]) || error("The local dimensions at n=$n must match the tensor sizes.")
    end
    A = [reshape(Array{ValueType2}(t), size(t, 1), localdims[n]..., size(t)[end]) for (n, t) in enumerate(TCI.sitetensors(tt))]
    return SiteTensorTrain{ValueType2,N}(A, 1)
end

# Generic wrapper for specifying localdims without explicit type parameter
function SiteTensorTrain{N}(tt::TCI.AbstractTensorTrain{ValueType}, localdims)::SiteTensorTrain{ValueType,N} where {ValueType,N}
    return SiteTensorTrain{ValueType,N}(tt, localdims)
end

# Convenience wrapper names
function sitetensortrain(a)
    return SiteTensorTrain(a)
end

function sitetensortrain(a, b)
    return SiteTensorTrain(a, b)
end

function sitetensortrain(a, b, c)
    return SiteTensorTrain(a, b, c)
end

function move_center_right!(tt::SiteTensorTrain{ValueType,N}) where {ValueType,N}
    c = center(tt)
    if c >= last(tt.partition)
        throw(ArgumentError("Cannot move center right: already at the rightmost position of partition"))
    end
    
    # QR decomposition of current center tensor
    T = TCI.sitetensor(tt, c)
    Q, R = qr(reshape(T, prod(size(T)[1:end-1]), size(T)[end]))
    Q = Matrix(Q)
    
    # Update current center tensor with Q
    tt.sitetensors[c] = reshape(Q, size(T)[1:end-1]..., size(Q, 2))
    
    # Contract R into the next tensor
    tmptt = reshape(TCI.sitetensor(tt, c+1), size(R, 2), :)
    tmptt = Matrix(R) * tmptt
    tt.sitetensors[c+1] = reshape(tmptt, size(TCI.sitetensor(tt, c+1))...)
    
    # Move center to the right
    tt.center = c + 1
end

function move_center_left!(tt::SiteTensorTrain{ValueType,N}) where {ValueType,N}
    c = center(tt)
    if c <= first(tt.partition)
        throw(ArgumentError("Cannot move center left: already at the leftmost position of partition"))
    end
    
    # LQ decomposition of current center tensor
    T = TCI.sitetensor(tt, c)
    bonddim_left, d1, d2, bonddim_right = size(T)
    T_mat = reshape(T, bonddim_left, d1*d2*bonddim_right)
    
    L, Q = lq(T_mat)
    Q = Matrix(Q)
    
    # Update current center tensor with Q
    tt.sitetensors[c] = reshape(Q, size(Q, 1), d1, d2, bonddim_right)
    
    # Contract L into the previous tensor
    tmptt = reshape(TCI.sitetensor(tt, c-1), :, size(L, 1))
    tmptt = tmptt * Matrix(L)
    tt.sitetensors[c-1] = reshape(tmptt, size(TCI.sitetensor(tt, c-1), 1), d1, d2, size(tmptt, 2))
    
    # Move center to the left
    tt.center = c - 1
end

function centercanonicalize(sitetensors::Vector{Array{ValueType, N}}, center::Int) where {ValueType, N}
    sitetensors_copy = copy(sitetensors)
    centercanonicalize!(sitetensors_copy, center)
    return sitetensors_copy
end

function centercanonicalize!(sitetensors::Vector{Array{ValueType, N}}, center::Int) where {ValueType, N}
    # LEFT
    for i in 1:center-1
        Q, R = qr(reshape(sitetensors[i], prod(size(sitetensors[i])[1:end-1]), size(sitetensors[i])[end]))
        Q = Matrix(Q)

        sitetensors[i] = reshape(Q, size(sitetensors[i])[1:end-1]..., size(Q, 2))

        tmptt = reshape(sitetensors[i+1], size(R, 2), :)
        tmptt = Matrix(R) * tmptt
        sitetensors[i+1] = reshape(tmptt, size(sitetensors[i+1])...)
    end
    # RIGHT
    for i in length(sitetensors):-1:center+1
        W = sitetensors[i]
        bonddim_left, d1, d2, bonddim_right = size(W)
        W_mat = reshape(W, bonddim_left, d1*d2*bonddim_right)

        L, Q = lq(W_mat)
        Q = Matrix(Q)
        
        sitetensors[i] = reshape(Q, size(Q, 1), d1, d2, bonddim_right)

        tmptt = reshape(sitetensors[i-1], :, size(L, 1))
        tmptt = tmptt * Matrix(L)
        sitetensors[i-1] = reshape(tmptt, size(sitetensors[i-1], 1), d1, d2, size(tmptt, 2))
    end
end


function isleftorthogonal(T::AbstractArray{ValueType,N}; atol::Float64=1e-7)::Bool where {ValueType, N}
    return isapprox(_contract(permutedims(T, (4,2,3,1,)), T, (2,3,4,),(2,3,1)), I, atol=atol)
end

function isrightorthogonal(T::AbstractArray{ValueType,N}; atol::Float64=1e-7)::Bool where {ValueType, N}
    return isapprox(_contract(T, permutedims(T, (4,2,3,1,)), (2,3,4,),(2,3,1)), I, atol=atol)
end

