"""
Contraction of two TTOs
Optionally, the contraction can be done with a function applied to the result.
"""
struct Contraction{ValueType} <: TCI.BatchEvaluator{ValueType}
    mpo::NTuple{2,TCI.TensorTrain{ValueType,4}}
    leftcache::Dict{Vector{Tuple{Int,Int}},Matrix{ValueType}}
    rightcache::Dict{Vector{Tuple{Int,Int}},Matrix{ValueType}}
    f::Union{Nothing,Function}
    sitedims::Vector{Vector{Int}}
end

Base.length(obj::Contraction) = length(obj.mpo[1])

function sitedims(obj::Contraction{ValueType})::Vector{Vector{Int}} where {ValueType}
    return obj.sitedims
end

function Base.lastindex(obj::Contraction{ValueType}) where {ValueType}
    return lastindex(obj.mpo[1])
end

function Base.getindex(obj::Contraction{ValueType}, i) where {ValueType}
    return getindex(obj.mpo[1], i)
end

function Base.show(io::IO, obj::Contraction{ValueType}) where {ValueType}
    print(
        io,
        "$(typeof(obj)) of tensor trains with ranks $(rank(obj.mpo[1])) and $(rank(obj.mpo[2]))",
    )
end

function Contraction(
    a::TCI.TensorTrain{ValueType,4},
    b::TCI.TensorTrain{ValueType,4};
    f::Union{Nothing,Function}=nothing,
) where {ValueType}
    mpo = a, b
    if length(unique(length.(mpo))) > 1
        throw(ArgumentError("Tensor trains must have the same length."))
    end
    for n = 1:length(mpo[1])
        if size(mpo[1][n], 3) != size(mpo[2][n], 2)
            error("Tensor trains must share the identical index at n=$(n)!")
        end
    end

    localdims1 = [size(mpo[1][n], 2) for n = 1:length(mpo[1])]
    localdims3 = [size(mpo[2][n], 3) for n = 1:length(mpo[2])]

    sitedims = [[x, y] for (x, y) in zip(localdims1, localdims3)]

    return Contraction(
        mpo,
        Dict{Vector{Tuple{Int,Int}},Matrix{ValueType}}(),
        Dict{Vector{Tuple{Int,Int}},Matrix{ValueType}}(),
        f,
        sitedims
    )
end

_localdims(obj::TCI.TensorTrain{<:Any,4}, n::Int)::Tuple{Int,Int} =
    (size(obj[n], 2), size(obj[n], 3))
_localdims(obj::Contraction{<:Any}, n::Int)::Tuple{Int,Int} =
    (size(obj.mpo[1][n], 2), size(obj.mpo[2][n], 3))

_getindex(x, indices) = ntuple(i -> x[indices[i]], length(indices))

function _contract(
    a::AbstractArray{T1,N1},
    b::AbstractArray{T2,N2},
    idx_a::NTuple{n1,Int},
    idx_b::NTuple{n2,Int}
) where {T1,T2,N1,N2,n1,n2}
    length(idx_a) == length(idx_b) || error("length(idx_a) != length(idx_b)")
    # check if idx_a contains only unique elements
    length(unique(idx_a)) == length(idx_a) || error("idx_a contains duplicate elements")
    # check if idx_b contains only unique elements
    length(unique(idx_b)) == length(idx_b) || error("idx_b contains duplicate elements")
    # check if idx_a and idx_b are subsets of 1:N1 and 1:N2
    all(1 <= idx <= N1 for idx in idx_a) || error("idx_a contains elements out of range")
    all(1 <= idx <= N2 for idx in idx_b) || error("idx_b contains elements out of range")

    rest_idx_a = setdiff(1:N1, idx_a)
    rest_idx_b = setdiff(1:N2, idx_b)

    amat = reshape(permutedims(a, (rest_idx_a..., idx_a...)), prod(_getindex(size(a), rest_idx_a)), prod(_getindex(size(a), idx_a)))
    bmat = reshape(permutedims(b, (idx_b..., rest_idx_b...)), prod(_getindex(size(b), idx_b)), prod(_getindex(size(b), rest_idx_b)))

    return reshape(amat * bmat, _getindex(size(a), rest_idx_a)..., _getindex(size(b), rest_idx_b)...)
end

function _unfuse_idx(obj::Contraction{ValueType}, n::Int, idx::Int)::Tuple{Int,Int} where {ValueType}
    return reverse(divrem(idx - 1, _localdims(obj, n)[1]) .+ 1)
end

function _fuse_idx(obj::Contraction{ValueType}, n::Int, idx::Tuple{Int,Int})::Int where {ValueType}
    return idx[1] + _localdims(obj, n)[1] * (idx[2] - 1)
end

function _extend_cache(oldcache::Matrix{ValueType}, a_ell::Array{ValueType,4}, b_ell::Array{ValueType,4}, i::Int, j::Int) where {ValueType}
    # (link_a, link_b) * (link_a, s, link_a') => (link_b, s, link_a')
    tmp1 = _contract(oldcache, a_ell[:, i, :, :], (1,), (1,))

    # (link_b, s, link_a') * (link_b, s, link_b') => (link_a', link_b')
    return _contract(tmp1, b_ell[:, :, j, :], (1, 2), (1, 2))
end

function svd_project_right(A::Array{ValueType,4}, max_r::Int; p::Int=2) where {ValueType}
    A_ = reshape(A, prod(size(A)[1:3]), size(A)[4])
    _, A_proj, _, disc = _factorize(A_, :SVD; maxbonddim=max_r+p, tolerance=0.0, leftorthogonal=false)
    # println("Discarded proj_right: $disc")

    return A_proj'
end

function svd_project_left(A::Array{ValueType,4}, max_l::Int; p::Int=2) where {ValueType}
    A_ = reshape(A, size(A)[1], prod(size(A)[2:4]))
    A_proj, _, _, disc = _factorize(A_, :SVD, maxbonddim=max_l+p, tolerance=0.0, leftorthogonal=true)
    # println("Discarded proj_left: $disc")

    return A_proj'
end

function random_project_right(A::Array{ValueType,4}, max_r::Int; p::Int=2) where {ValueType}
    A_ = reshape(A, prod(size(A)[1:3]), size(A)[4])
    A_proj = Matrix(qr(A_'*randn(size(A_)[1], max_r+p)).Q)

    # println("From $(size(A_proj)[1]) to $(size(A_proj)[2])")

    return A_proj
end

function random_project_left(A::Array{ValueType,4}, max_l::Int; p::Int=2) where {ValueType}
    A_ = reshape(A, size(A)[1], prod(size(A)[2:4]))
    A_proj = Matrix(qr(A_*randn(size(A_)[2], max_l+p)).Q)'

    # println("From $(size(A_proj)[2]) to $(size(A_proj)[1])")

    return A_proj
end

# Compute left environment
function evaluateleft(
    obj::Contraction{ValueType},
    indexset::AbstractVector{Tuple{Int,Int}},
)::Matrix{ValueType} where {ValueType}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    a, b = obj.mpo

    if length(indexset) == 0
        return ones(ValueType, 1, 1)
    end

    ell = length(indexset)
    if ell == 1
        i, j = indexset[1]
        return transpose(a[1][1, i, :, :]) * b[1][1, :, j, :]
    end

    key = collect(indexset)
    if !(key in keys(obj.leftcache))
        i, j = indexset[end]
        obj.leftcache[key] = _extend_cache(evaluateleft(obj, indexset[1:ell-1]), a[ell], b[ell], i, j)
    end

    return obj.leftcache[key]
end



# Compute right environment
function evaluateright(
    obj::Contraction{ValueType},
    indexset::AbstractVector{Tuple{Int,Int}},
)::Matrix{ValueType} where {ValueType}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    a, b = obj.mpo

    N = length(obj)

    if length(indexset) == 0
        return ones(ValueType, 1, 1)
    elseif length(indexset) == 1
        i, j = indexset[1]
        return a[end][:, i, :, 1] * transpose(b[end][:, :, j, 1])
    end

    ell = N - length(indexset) + 1

    key = collect(indexset)
    if !(key in keys(obj.rightcache))
        i, j = indexset[1]
        obj.rightcache[key] = _extend_cache(
            evaluateright(obj, indexset[2:end]),
            permutedims(a[ell], (4, 2, 3, 1)),
            permutedims(b[ell], (4, 2, 3, 1)),
            i, j)
    end

    return obj.rightcache[key]
end


function evaluate(obj::Contraction{ValueType}, indexset::AbstractVector{Int})::ValueType where {ValueType}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    indexset_unfused = [_unfuse_idx(obj, n, indexset[n]) for n = 1:length(obj)]
    return evaluate(obj, indexset_unfused)
end

function evaluate(
    obj::Contraction{ValueType},
    indexset::AbstractVector{Tuple{Int,Int}},
)::ValueType where {ValueType}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    midpoint = div(length(obj), 2)
    res = sum(
        evaluateleft(obj, indexset[1:midpoint]) .*
        evaluateright(obj, indexset[midpoint+1:end]),
    )

    if obj.f isa Function
        return obj.f(res)
    else
        return res
    end
end

function _lineari(dims, mi)::Integer
    ci = CartesianIndex(Tuple(mi))
    li = LinearIndices(Tuple(dims))
    return li[ci]
end

function lineari(sitedims::Vector{Vector{Int}}, indexset::Vector{TCI.MultiIndex})::Vector{Int}
    return [_lineari(sitedims[l], indexset[l]) for l in 1:length(indexset)]
end


function (obj::Contraction{ValueType})(indexset::AbstractVector{Int})::ValueType where {ValueType}
    return evaluate(obj, indexset)
end

function (obj::Contraction{ValueType})(indexset::AbstractVector{<:AbstractVector{Int}})::ValueType where {ValueType}
    return evaluate(obj, lineari(obj.sitedims, indexset))
end

function (obj::Contraction{ValueType})(
    leftindexset::AbstractVector{TCI.MultiIndex},
    rightindexset::AbstractVector{TCI.MultiIndex},
    ::Val{M},
)::Array{ValueType,M + 2} where {ValueType,M}
    return batchevaluate(obj, leftindexset, rightindexset, Val(M))
end

function batchevaluate(obj::Contraction{ValueType},
    leftindexset::AbstractVector{TCI.MultiIndex},
    rightindexset::AbstractVector{TCI.MultiIndex},
    ::Val{M},
    projector::Union{Nothing,AbstractVector{<:AbstractVector{<:Integer}}}=nothing)::Array{ValueType,M + 2} where {ValueType,M}
    N = length(obj)
    Nr = length(rightindexset[1])
    s_ = length(leftindexset[1]) + 1
    e_ = N - length(rightindexset[1])
    a, b = obj.mpo

    if projector === nothing
        projector = [fill(0, length(obj.sitedims[n])) for n in s_:e_]
    end
    length(projector) == M || error("Length mismatch: length of projector (=$(length(projector))) must be $(M)")
    for n in s_:e_
        length(projector[n - s_ + 1]) == 2 || error("Invalid projector at $n: $(projector[n - s_ + 1]), the length must be 2")
        all(0 .<= projector[n - s_ + 1] .<= obj.sitedims[n]) || error("Invalid projector: $(projector[n - s_ + 1])")
    end


    # Unfused index
    leftindexset_unfused = [
        [_unfuse_idx(obj, n, idx) for (n, idx) in enumerate(idxs)] for idxs in leftindexset
    ]
    rightindexset_unfused = [
        [_unfuse_idx(obj, N - Nr + n, idx) for (n, idx) in enumerate(idxs)] for
        idxs in rightindexset
    ]

    t1 = time_ns()
    linkdims_a = vcat(1, TCI.linkdims(a), 1)
    linkdims_b = vcat(1, TCI.linkdims(b), 1)

    left_ = Array{ValueType,3}(undef, length(leftindexset), linkdims_a[s_], linkdims_b[s_])
    for (i, idx) in enumerate(leftindexset_unfused)
        left_[i, :, :] .= evaluateleft(obj, idx)
    end
    t2 = time_ns()

    right_ = Array{ValueType,3}(
        undef,
        linkdims_a[e_+1],
        linkdims_b[e_+1],
        length(rightindexset),
    )
    for (i, idx) in enumerate(rightindexset_unfused)
        right_[:, :, i] .= evaluateright(obj, idx)
    end
    t3 = time_ns()

    # (left_index, link_a, link_b, site[s_] * site'[s_] *  ... * site[e_] * site'[e_])
    leftobj::Array{ValueType,4} = reshape(left_, size(left_)..., 1)
    return_size_siteinds = Int[]
    for n = s_:e_
        slice_ab, shape_ab = TCI.projector_to_slice(projector[n - s_ + 1])
        a_n = begin
            a_n_org = obj.mpo[1][n]
            tmp = a_n_org[:, slice_ab[1], :, :]
            reshape(tmp, size(a_n_org, 1), shape_ab[1], size(a_n_org)[3:4]...)
        end
        b_n = begin
            b_n_org = obj.mpo[2][n]
            tmp = b_n_org[:, :, slice_ab[2], :]
            reshape(tmp, size(b_n_org, 1), size(b_n_org, 2), shape_ab[2], size(b_n_org, 4))
        end
        push!(return_size_siteinds, size(a_n, 2) * size(b_n, 3))

        #(left_index, link_a, link_b, S) * (link_a, site[n], shared, link_a')
        #  => (left_index, link_b, S, site[n], shared, link_a')
        tmp1 = _contract(leftobj, a_n, (2,), (1,))

        # (left_index, link_b, S, site[n], shared, link_a') * (link_b, shared, site'[n], link_b')
        #  => (left_index, S, site[n], link_a', site'[n], link_b')
        tmp2 = _contract(tmp1, b_n, (2, 5), (1, 2))

        # (left_index, S, site[n], link_a', site'[n], link_b')
        #  => (left_index, link_a', link_b', S, site[n], site'[n])
        tmp3 = permutedims(tmp2, (1, 4, 6, 2, 3, 5))

        leftobj = reshape(tmp3, size(tmp3)[1:3]..., :)
    end

    return_size = (
        length(leftindexset),
        return_size_siteinds...,
        length(rightindexset),
    )
    t5 = time_ns()

    # (left_index, link_a, link_b, S) * (link_a, link_b, right_index)
    #   => (left_index, S, right_index)
    res = _contract(leftobj, right_, (2, 3), (1, 2))

    if obj.f isa Function
        res .= obj.f.(res)
    end

    return reshape(res, return_size)
end


function _contractsitetensors(a::Array{ValueType,4}, b::Array{ValueType,4})::Array{ValueType,4} where {ValueType}
    # indices: (link_a, s1, s2, link_a') * (link_b, s2, s3, link_b')
    ab::Array{ValueType,6} = _contract(a, b, (3,), (2,))
    # => indices: (link_a, s1, link_a', link_b, s3, link_b')
    abpermuted = permutedims(ab, (1, 4, 2, 5, 3, 6))
    # => indices: (link_a, link_b, s1, s3, link_a', link_b')
    return reshape(abpermuted,
        size(a, 1) * size(b, 1),  # link_a * link_b
        size(a, 2), size(b, 3),  # s1, s3
        size(a, 4) * size(b, 4)   # link_a' * link_b'
    )
end

function contract_naive(
    a::TCI.TensorTrain{ValueType,4}, b::TCI.TensorTrain{ValueType,4};
    tolerance=0.0, maxbonddim=typemax(Int)
)::TCI.TensorTrain{ValueType,4} where {ValueType}
    return contract_naive(Contraction(a, b); tolerance, maxbonddim)
end

function contract_naive(
    obj::Contraction{ValueType};
    tolerance=0.0, maxbonddim=typemax(Int)
)::TCI.TensorTrain{ValueType,4} where {ValueType}
    if obj.f isa Function
        error("Naive contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
    end

    a, b = obj.mpo
    tt = TCI.TensorTrain{ValueType,4}(_contractsitetensors.(TCI.sitetensors(a), TCI.sitetensors(b)))
    if tolerance > 0 || maxbonddim < typemax(Int)
        TCI.compress!(tt, :SVD; tolerance, maxbonddim)
    end
    return tt
end

function _reshape_fusesites(t::AbstractArray{ValueType}) where {ValueType}
    shape = size(t)
    return reshape(t, shape[1], prod(shape[2:end-1]), shape[end]), shape[2:end-1]
end

function _reshape_splitsites(
    t::AbstractArray{ValueType},
    legdims::Union{AbstractVector{Int},Tuple},
) where {ValueType}
    return reshape(t, size(t, 1), legdims..., size(t, ndims(t)))
end

function _findinitialpivots(f, localdims, nmaxpivots)::Vector{TCI.MultiIndex}
    pivots = TCI.MultiIndex[]
    for _ in 1:nmaxpivots
        pivot_ = [rand(1:d) for d in localdims]
        pivot_ = optfirstpivot(f, localdims, pivot_)
        if abs(f(pivot_)) == 0.0
            continue
        end
        push!(pivots, pivot_)
    end
    return pivots
end

function contract_TCI(
    A::TCI.TensorTrain{ValueType,4},
    B::TCI.TensorTrain{ValueType,4};
    initialpivots::Union{Int,Vector{TCI.MultiIndex}}=10,
    f::Union{Nothing,Function}=nothing,
    kwargs...
)::TCI.TensorTrain{ValueType,4} where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end
    if !all([sitedim(A, i)[2] == sitedim(B, i)[1] for i = 1:length(A)])
        throw(
            ArgumentError(
                "Cannot contract tensor trains with non-matching site dimensions.",
            ),
        )
    end
    matrixproduct = Contraction(A, B; f=f)
    localdims = prod.(matrixproduct.sitedims)
    if initialpivots isa Int
        initialpivots = _findinitialpivots(matrixproduct, localdims, initialpivots)
        if isempty(initialpivots)
            error("No initial pivots found.")
        end
    end

    tci, ranks, errors = crossinterpolate2(
        ValueType,
        matrixproduct,
        localdims,
        initialpivots;
        kwargs...,
    )
    legdims = [_localdims(matrixproduct, i) for i = 1:length(tci)]
    return TCI.TensorTrain{ValueType,4}(
        [_reshape_splitsites(t, d) for (t, d) in zip(tci, legdims)]
    )
end

"""
See SVD version:
https://tensornetwork.org/mps/algorithms/zip_up_mpo/
"""
function contract_zipup(
    A::TCI.TensorTrain{ValueType,4},
    B::TCI.TensorTrain{ValueType,4};
    tolerance::Float64=1e-12,
    method::Symbol=:SVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    kwargs...
)::TCI.TensorTrain{ValueType,4} where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end
    R::Array{ValueType,3} = ones(ValueType, 1, 1, 1)

    sitetensors = Vector{Array{ValueType,4}}(undef, length(A))
    for n in 1:length(A)
        # R:     (link_ab, link_an, link_bn)
        # A[n]:  (link_an, s_n, s_n', link_anp1)
        #println("Time RA:")
        RA = _contract(R, A[n], (2,), (1,))

        # RA[n]: (link_ab, link_bn, s_n, s_n' link_anp1)
        # B[n]:  (link_bn, s_n', s_n'', link_bnp1)
        # C:     (link_ab, s_n, link_anp1, s_n'', link_bnp1)
        #  =>    (link_ab, s_n, s_n'', link_anp1, link_bnp1)
        #println("Time C:")
        C = permutedims(_contract(RA, B[n], (2, 4), (1, 2)), (1, 2, 4, 3, 5))
        #println(size(C))
        if n == length(A)
            sitetensors[n] = reshape(C, size(C)[1:3]..., 1)
            break
        end

        # Cmat:  (link_ab * s_n * s_n'', link_anp1 * link_bnp1)

        #lu = rrlu(Cmat; reltol, abstol, leftorthogonal=true)
        left, right, newbonddim = _factorize(
            reshape(C, prod(size(C)[1:3]), prod(size(C)[4:5])),
            method; tolerance, maxbonddim
        )

        # U:     (link_ab, s_n, s_n'', link_ab_new)
        sitetensors[n] = reshape(left, size(C)[1:3]..., newbonddim)

        # R:     (link_ab_new, link_an, link_bn)
        R = reshape(right, newbonddim, size(C)[4:5]...)
    end

    return TCI.TensorTrain{ValueType,4}(sitetensors)
end

# If Ri = i, then R is the right environment until site i+1, which has size (size(A[i])[end], size(B[i])[end], size(X[i]])[end])
# If Li = i, then L is the left environment until site i-1, which has size (size(A[i])[1], size(B[i])[1], size(X[i]])[1])
function updatecore!(A::SiteTensorTrain{ValueType,4}, B::SiteTensorTrain{ValueType,4}, C::SiteTensorTrain{ValueType,4}, i::Int,
    L::Array{ValueType,3}, R::Array{ValueType,3};
    method::Symbol=:SVD, tolerance::Float64=1e-8, maxbonddim::Int=typemax(Int), direction::Symbol=:forward, random_update::Bool=false, p::Int=0
    )::Float64 where {ValueType}
    
    Ai = TCI.sitetensor(A, i)
    Bi = TCI.sitetensor(B, i)
    Aip1 = TCI.sitetensor(A, i+1)
    Bip1 = TCI.sitetensor(B, i+1)

    # Compute left and right effective environments
    Le = if random_update
        A_proj = random_project_right(Ai, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        B_proj = random_project_right(Bi, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        Ai = _contract(Ai, A_proj, (4,), (1,))
        Bi = _contract(Bi, B_proj, (4,), (1,))
        _contract(_contract(L, Ai, (1,), (1,)), Bi, (1, 4), (1, 2))
    else
        _contract(_contract(L, Ai, (1,), (1,)), Bi, (1, 4), (1, 2))
    end
    
    Re = if random_update
        A_proj = random_project_right(Aip1, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        B_proj = random_project_right(Bip1, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        Aip1 = _contract(A_proj, Aip1, (1,), (1,))
        Bip1 = _contract(B_proj, Bip1, (1,), (1,))
        _contract(Bip1, _contract(Aip1, R, (4,), (1,)), (2, 4), (3, 4))
    else
        _contract(Bip1, _contract(Aip1, R, (4,), (1,)), (2, 4), (3, 4))
    end
    
    # Contract environments and factorize
    Ce = _contract(Le, Re, (3, 5), (3, 1)) |>
         Ce -> permutedims(Ce, (1, 2, 3, 5, 4, 6)) |>
         Ce -> reshape(Ce, prod(size(Ce)[1:3]), prod(size(Ce)[4:6]))
    
    left, right, newbonddim, disc = _factorize(
        Ce, method; 
        tolerance, maxbonddim, 
        leftorthogonal=(direction == :forward)
    )
    
    # Update cores
    C_i = reshape(left, :, size(C[i], 2), size(C[i], 3), newbonddim)
    C_ip1 = reshape(right, newbonddim, size(C[i+1], 2), size(C[i+1], 3), :)
    
    
    if direction == :forward
        movecenterright!(A)
        movecenterright!(B)
    else
        movecenterleft!(A)
        movecenterleft!(B)
    end
    settwositetensors!(C, i, C_i, C_ip1)
    return disc
end

function leftenvironment!(
    Ls::Vector{Array{ValueType,3}},
    A::SiteTensorTrain{ValueType, 4},
    B::SiteTensorTrain{ValueType, 4},
    C::SiteTensorTrain{ValueType, 4},
    i::Int;
    random_env::Bool = false, p::Int=0
)::Array{ValueType, 3} where {ValueType}
    Ai = TCI.sitetensor(A, i)
    Bi = TCI.sitetensor(B, i)
    Ci = TCI.sitetensor(C, i)

    if random_env
        A_proj_left = random_project_left(Ai, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        B_proj_left = random_project_left(Bi, Int(ceil(sqrt(maximum(TCI.linkdims(B))))); p)
        
        L = _contract(L, A_proj_left', (1,), (1,))
        L = _contract(L, B_proj_left', (1,), (1,))
        L = permutedims(L, (2,3,1,))

        Ai = _contract(A_proj_left, Ai, (2,), (1,))
        Bi = _contract(B_proj_left, Bi, (2,), (1,))
    end

    Ls[i] = _contract(get(Ls, i-1, ones(ValueType, 1, 1, 1)), Ai, (1,), (1,))
    Ls[i] = _contract(Ls[i], Bi, (1,4,), (1,2,))
    Ls[i] = _contract(Ls[i], conj(Ci), (1,2,4,), (1,2,3,))
end

function rightenvironment!(
    Rs::Vector{Array{ValueType,3}},
    A::SiteTensorTrain{ValueType, 4},
    B::SiteTensorTrain{ValueType, 4},
    C::SiteTensorTrain{ValueType, 4},
    i::Int;
    random_env::Bool = false, p::Int=0
)::Array{ValueType, 3} where {ValueType}
    Ai = TCI.sitetensor(A, i)
    Bi = TCI.sitetensor(B, i)
    Ci = TCI.sitetensor(C, i)

    if random_env
        A_proj_right = random_project_right(Ai, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        B_proj_right = random_project_right(Bi, Int(ceil(sqrt(maximum(TCI.linkdims(B))))); p)
        
        R = _contract(B_proj_right', Rs[i+1], (2,), (2,))
        R = _contract(A_proj_right', R, (2,), (2,))
        R = permutedims(R, (2,3,1,))

        Ai = _contract(Ai, A_proj_right, (4,), (1,))
        Bi = _contract(Bi, B_proj_right, (4,), (1,))
    end

    Rs[i] = _contract(conj(Ci), get(Rs, i-1, ones(ValueType, 1, 1, 1)), (4,), (3,))
    Rs[i] = _contract(Bi, Rs[i], (3,4,), (3,5,))
    Rs[i] = _contract(Ai, Rs[i], (2,3,4), (4,2,5,))
end

function updatecore!(A::InverseTensorTrain{ValueType,4}, B::InverseTensorTrain{ValueType,4}, C::InverseTensorTrain{ValueType,4}, i::Int,
    L::Array{ValueType,3}, R::Array{ValueType,3};
    method::Symbol=:SVD, tolerance::Float64=1e-8, maxbonddim::Int=typemax(Int), random_update::Bool=false, p::Int=0
    )::Float64 where {ValueType}
    
    # Compute left effective environment
    Ai = TCI.sitetensor(A, i)
    Bi = TCI.sitetensor(B, i)
    Yai = inversesingularvalue(A, i)
    Ybi = inversesingularvalue(B, i)
    Aip1 = TCI.sitetensor(A, i+1)
    Bip1 = TCI.sitetensor(B, i+1)

    Le = if random_update
        A_proj = random_project_right(Ai, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        B_proj = random_project_right(Bi, Int(ceil(sqrt(maximum(TCI.linkdims(B))))); p)
        Ai_projected = _contract(Ai, A_proj, (4,), (1,))
        Bi_projected = _contract(Bi, B_proj, (4,), (1,))
        _contract(_contract(L, Ai_projected, (1,), (1,)), Bi_projected, (1, 4), (1, 2))
    else
        Ai = _contract(Ai, Yai, (4,), (1,))
        Bi = _contract(Bi, Ybi, (4,), (1,))
        _contract(_contract(L, Ai, (1,), (1,)), Bi, (1, 4), (1, 2))
    end
    
    # Compute right effective environment
    Re = if random_update
        Aip1 = _contract(Yai, Aip1, (2,), (1,))
        Bip1 = _contract(Ybi, Bip1, (2,), (1,))
        A_proj = random_project_right(Aip1, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        B_proj = random_project_right(Bip1, Int(ceil(sqrt(maximum(TCI.linkdims(B))))); p)
        Aip1_projected = _contract(A_proj', Aip1, (2,), (1,))
        Bip1_projected = _contract(B_proj', Bip1, (2,), (1,))
        _contract(Bip1_projected, _contract(Aip1_projected, R, (4,), (1,)), (2, 4), (3, 4))
    else
        _contract(Bip1, _contract(Aip1, R, (4,), (1,)), (2, 4), (3, 4))
    end
    
    # Contract environments and factorize with diamond
    Ce = _contract(Le, Re, (3, 5), (3, 1)) |>
         C -> permutedims(C, (1, 2, 3, 5, 4, 6)) |>
         C -> reshape(C, prod(size(C)[1:3]), prod(size(C)[4:6]))
    
    left, diamond, right, newbonddim, disc = _factorize(
         Ce, method; 
        tolerance, maxbonddim, 
        diamond=true
    )
    
    # Update cores with singular values
    C_i = reshape(left * Diagonal(diamond), :, size(Psic[i])[2:3]..., newbonddim)
    C_ip1 = reshape(Diagonal(diamond) * right, newbonddim, size(Psic[i+1])[2:3]..., :)
    Vc_i = Diagonal(diamond.^-1)
    
    settwositetensors!(C, i, C_i, Vc_i, C_ip1)

    return disc   
end


# Compute left environment
function leftenvironment!(
    Ls::Vector{Array{ValueType,3}},
    A::InverseTensorTrain{ValueType, 4},
    B::InverseTensorTrain{ValueType, 4},
    C::InverseTensorTrain{ValueType, 4},
    i::Int;
    random_env::Bool = false, p::Int=0
    )::Array{ValueType, 3} where {ValueType}

    Ai = TCI.sitetensor(A, i)
    Bi = TCI.sitetensor(B, i)
    Ci = TCI.sitetensor(C, i)
    Yai = inversesingularvalue(A, i)
    Ybi = inversesingularvalue(B, i)
    Yci = inversesingularvalue(C, i)

    if random_env
        A_proj_left = random_project_left(Ai, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        B_proj_left = random_project_left(Bi, Int(ceil(sqrt(maximum(TCI.linkdims(B))))); p)
        
        Ls[i] = _contract(Ls[i], A_proj_left', (1,), (1,))
        Ls[i] = _contract(Ls[i], B_proj_left', (1,), (1,))
        Ls[i] = permutedims(Ls[i], (2,3,1,))

        Ai = _contract(A_proj_left, Ai, (2,), (1,))
        Bi = _contract(B_proj_left, Bi, (2,), (1,))
    end

    Ai = _contract(Ai, Yai, (4,), (1,))
    Bi = _contract(Bi, Ybi, (4,), (1,))
    Ci = _contract(Ci, Yci, (4,), (1,))
    
    Ls[i] = _contract(get(Ls, i-1, ones(ValueType, 1, 1, 1)), Ai, (1,), (1,))
    Ls[i] = _contract(Ls[i], Bi, (1,4,), (1,2,))
    Ls[i] = _contract(Ls[i], conj(Psic_), (1,2,4,), (1,2,3,))
end

# Compute right environment
function rightenvironment!(
    Rs::Vector{Array{ValueType,3}},
    A::InverseTensorTrain{ValueType, 4},
    B::InverseTensorTrain{ValueType, 4},
    C::InverseTensorTrain{ValueType, 4},
    i::Int;
    random_env::Bool = false, p::Int=0
    )::Array{ValueType, 3} where {ValueType}

    Ai = TCI.sitetensor(A, i)
    Bi = TCI.sitetensor(B, i)
    Ci = TCI.sitetensor(C, i)
    Yaim1 = inversesingularvalue(A, i-1)
    Ybim1 = inversesingularvalue(B, i-1)
    Ycim1 = inversesingularvalue(C, i-1)

    if random_env
        A_proj_right = random_project_right(Ai, Int(ceil(sqrt(maximum(TCI.linkdims(A))))); p)
        B_proj_right = random_project_right(Bi, Int(ceil(sqrt(maximum(TCI.linkdims(B))))); p)

        Rs[i] = _contract(B_proj_right', Rs[i], (2,), (2,))
        Rs[i] = _contract(A_proj_right', Rs[i], (2,), (2,))

        Ai = _contract(Ai, A_proj_right, (4,), (1,))
        Bi = _contract(Bi, B_proj_right, (4,), (1,))

    end
    Ai = _contract(Yaim1, Ai, (2,), (1,))
    Bi = _contract(Ybim1, Bi, (2,), (1,))
    Ci = _contract(Ycim1, Ci, (2,), (1,))

    Rs[i] = _contract(conj(Ci), get(Rs, i+1, ones(ValueType, 1, 1, 1)), (4,), (3,))
    Rs[i] = _contract(Bi, Rs[i], (3,4,), (3,5,))
    Rs[i] =  _contract(Ai, Rs[i], (2,3,4), (4,2,5,))
end

"""
    function contract_fit(A::TCI.TensorTrain{ValueType,4}, B::TCI.TensorTrain{ValueType,4})

Conctractos tensor trains A and B using the fit algorithm.

# Keyword Arguments

- `nsweeps::Int`: Number of sweeps to perform during the algorithm.
- `initial_guess`: Optional initial guess for the tensor train A*B. If not provided, a tensor train of rank one is used. This must have coherent bond dimension (i.e. start and finish with less than [d,d^2,d^3,...,d^3,d^2,d]).
- `tolerance::Float64`: Convergence tolerance for the iterative algorithm.
- `method::Symbol`: Algorithm or method to use for the computation :SVD, :RSVD, :LU, :CI.
- `maxbonddim::Int`: Maximum bond dimension allowed during the decomposition.
"""
function contract_fit(
    A::SiteTensorTrain{ValueType,4},
    B::SiteTensorTrain{ValueType,4};
    C::SiteTensorTrain{ValueType,4}=deepcopy(A), #TODO try without deepcopy
    debug::SiteTensorTrain{ValueType,4}=deepcopy(A),
    nsweeps::Int=2,
    tolerance::Float64=1e-12,
    method::Symbol=:SVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    random_update::Bool=false,
    random_env::Bool=false,
    p::Int=0,
    kwargs...)::TCI.TensorTrain{ValueType,4} where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    n = length(A)
    
    Rs = Vector{Array{ValueType, 3}}(undef, n)
    Ls = Vector{Array{ValueType, 3}}(undef, n)

    setcenter!(A, 1)
    setcenter!(B, 1)
    setcenter!(C, 1)

    # Precompute right environments
    for i in n:-1:3
        rightenvironment!(Rs, A, B, C, i; random_env, p)
    end
    
    # It doesn't matter if we repeat update in 1 or n-1, those are negligible
    direction = :forward
    for sweep in 1:nsweeps
        tot_disc = 0.0
        if direction == :forward
            for i in 1:n-1

                i > 1 && leftenvironment!(Ls, A, B, C, i-1; random_env, p)

                disc = updatecore!(A, B, C, i, get(Ls, i-1, ones(ValueType, 1, 1, 1)), get(Rs, i+2, ones(ValueType, 1, 1, 1));
                    method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, p)

                tot_disc += disc
            end
            direction = :backward
        elseif direction == :backward
            for i in n-1:-1:1
                i < n-1 && rightenvironment!(Rs, A, B, C, i+2; random_env, p)

                disc = updatecore!(A, B, C, i, get(Ls, i-1, ones(ValueType, 1, 1, 1)), get(Rs, i+2, ones(ValueType, 1, 1, 1));
                    method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, p)

                tot_disc += disc
            end
            direction = :forward    
        end

        if tot_disc < tolerance
            break
        end
    end
    return C
end

# TODO remove this
function mynorm(tt)
    res = permutedims(_contract(tt[1], conj(tt[1]), (2,3,), (2,3,)), (1,3,2,4,))
    for i in 2:length(tt)
        # println("Pre step $i, res is $(size(res)), tt[$i] is $(size(tt[i]))")
        res = _contract(res, permutedims(_contract(tt[i], conj(tt[i]), (2,3,), (2,3,)), (1,3,2,4,)), (3,4,), (1,2,))
    end
    if res[1][1] < 0.
        println("Error, norm^2 is $(res[1][1])")
    end
    return sqrt(abs(res[1][1]))
end

function contract_fit(
    A::InverseTensorTrain{ValueType,4},
    B::InverseTensorTrain{ValueType,4};
    C::InverseTensorTrain{ValueType,4}=deepcopy(A),
    nsweeps::Int=2,
    tolerance::Float64=1e-12,
    method::Symbol=:SVD,
    maxbonddim::Int=typemax(Int),
    random_update::Bool=false,
    random_env::Bool=false,
    p::Int=0,
    kwargs...)::TCI.TensorTrain{ValueType,4} where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    n = length(A)
    
    Rs = Vector{Array{ValueType, 3}}(undef, n)
    Ls = Vector{Array{ValueType, 3}}(undef, n)

    # Precompute right environments
    for i in n:-1:3
        rightenvironment!(Rs, A, B, C, i; random_env, p)
    end
    
    direction = :forward
    for sweep in 1:nsweeps
        tot_disc = 0.0
        if direction == :forward
            for i in 1:n-1
                i > 1 && leftenvironment!(Ls, A, B, C, i-1; random_env, p)

                disc = updatecore!(A, B, C, i, get(Ls, i-1, ones(ValueType, 1, 1, 1)), get(Rs, i+2, ones(ValueType, 1, 1, 1));
                    method, tolerance=tolerance/((n-1)), maxbonddim, random_update, p)

                tot_disc += disc
            end
            direction = :backward
        elseif direction == :backward
            for i in n-1:-1:1
                i < n-1 && rightenvironment!(Rs, A, B, C, i+2; random_env, p)

                disc = updatecore!(A, B, C, i, get(Ls, i-1, ones(ValueType, 1, 1, 1)), get(Rs, i+2, ones(ValueType, 1, 1, 1));
                    method, tolerance=tolerance/((n-1)), maxbonddim, random_update, p)

                tot_disc += disc
            end
            direction = :forward    
        end

        if tot_disc < tolerance
            break
        end
    end
    
    return C
end

function contract_distr_fit(
    A::InverseTensorTrain{ValueType,4},
    B::InverseTensorTrain{ValueType,4};
    C::InverseTensorTrain{ValueType,4}=deepcopy(A),
    nsweeps::Int=8,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    partitions::Union{Nothing,AbstractVector{<:AbstractRange{Int}}}=nothing,
    tolerance::Float64=1e-12,
    method::Symbol=:RSVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    random_update::Bool=false,
    random_env::Bool=false,
    synchedinput::Bool=false,
    synchedoutput::Bool=true,
    p::Int=0,
    kwargs...
)::TCI.TensorTrain{ValueType,4} where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    if !synchedinput
        synchronize_tt!(A)
        synchronize_tt!(B)
        synchronize_tt!(C)
    end

    n = length(A)

    Psic = Vector{Array{ValueType, 4}}(undef, n)
    Vc = Vector{Array{ValueType, 2}}(undef, n)
    for i in 1:n
        if !isnothing(Psi_init)
            Psic[i] = deepcopy(Psi_init[i])
        else
            Psic[i] = deepcopy(Psia[i])
        end
    end
    
    for i in 1:n-1
        if !isnothing(V_init)
            Vc[i] = deepcopy(V_init[i])
        else
            Vc[i] = deepcopy(Va[i])
        end
    end
    

    if MPI.Initialized()
        if subcomm != nothing
            comm = subcomm
        else
            comm = MPI.COMM_WORLD
        end
        mpirank = MPI.Comm_rank(comm)
        juliarank = mpirank + 1
        nprocs = MPI.Comm_size(comm)
        if partitions == nothing
            if n < 4
                println("Warning! The TT is too small to be parallelized.")
                return contract_fit(A, B; C, nsweeps, tolerance, method, maxbonddim, kwargs...)
            end
            if n < 6
                if nprocs > 2
                    println("Warning! The TT is too small to be parallelized with more than 2 nodes. Some nodes will not be used.")
                end
                nprocs = 2
                partitions = [1:2,3:n]
            elseif n == 6
                if nprocs == 2
                    partitions = [1:3,4:6]
                elseif nprocs == 3
                    partitions = [1:2,3:4,5:6]
                else
                    println("Warning! The TT is too small to be parallelized with more than 3 nodes. Some nodes will not be used.")
                    nprocs = 3
                    partitions = [1:2,3:4,5:6]
                end
            else 
                extra1 = 3 # "Hyperparameter"
                extraend = 3
                if nprocs > div(n - extra1 - extraend, 2) # It's just one update per node.
                    println("Warning! A TT of lenght L can use be parallelized with up to (L-$(extra1 + extraend))/2 nodes. Some nodes will not be used.")
                    # Each node has 2 cores. Except first and last who have 2+extra1 and 2+extraend+n%2
                    if n % 2 == 0
                        partitions = vcat([1:extra1+1],[extra1+i+1:extra1+i+2 for i in 1:2:n-extra1-extraend-2],[n-extraend:n])
                    else
                        partitions = vcat([1:extra1+1],[extra1+i+1:extra1+i+2 for i in 1:2:n-extra1-extraend-3],[n-1-extraend:n])
                    end
                    for _ in div(n - extra1 - extraend, 2)+1:nprocs
                        push!(partitions, 1:-1)
                    end
                    nprocs = div(n - extra1 - extraend, 2)
                else
                    number_of_sites, rem = divrem(n-extra1-extraend, nprocs)
                    sites = [number_of_sites for i in 1:nprocs]
                    sites[1] += extra1
                    sites[end] += extraend
                    # Distribute the remainder as evenly as possible
                    for i in 1:rem
                        if i % 2 == 1
                            sites[div(i,2)+2] += 1
                        else
                            sites[end-div(i,2)] += 1
                        end
                    end
                    partitions = [sum(sites[1:i-1])+1:sum(sites[1:i]) for i in 1:nprocs]
                end
            end
        end
    else
        println("Warning! Distributed strategy has been chosen, but MPI is not initialized, please use TCI.initializempi() before contract() and use TCI.finalizempi() afterwards")
        return contract_fit(A, B; C, nsweeps, tolerance, method, maxbonddim, kwargs...)
    end

    setpartition!(A, partitions[juliarank])
    setpartition!(B, partitions[juliarank])
    setpartition!(C, partitions[juliarank])

    first_site = first(partitions[juliarank])
    last_site = last(partitions[juliarank])
    if nprocs == 1
        println("Warning! Distributed strategy has been chosen, but only one process is running")
        return contract_fit(A, B; C, nsweeps, tolerance, method, maxbonddim, kwargs...)
    end

    if juliarank <= nprocs

        Ls = Vector{Array{ValueType, 3}}(undef, n)
        Rs = Vector{Array{ValueType, 3}}(undef, n)

        #Ls are left enviroments and Rs are right environments
        if first_site != 1
            Ls[first_site-1] = rand(ValueType, 
                first(size(sitetensor(A, first_site))), first(size(sitetensor(A, first_site))), first(size(sitetensor(A, first_site))))
            Ls[first_site-1] = Ls[first_site-1] ./ sqrt(sum(Ls[first_site-1].^2)) 
        end
        if last_site != n
            Rs[last_site+1] = rand(ValueType,
                last(size(sitetensor(A, last_site))), last(size(sitetensor(A, last_site))), last(size(sitetensor(A, last_site))))
            Rs[last_site+1] = Rs[last_site+1] ./ sqrt(sum(Rs[last_site+1].^2))
        end

        # TODO make so that you can either precompute all or only local environments
        if juliarank % 2 == 1 # Precompute right environment if going forward
            for i in last_site:-1:first_site+2
                rightenvironment!(Rs, A, B, C, i; random_env, p)
            end
        else # Precompute left environment if going backward
            for i in first_site:last_site-2
                leftenvironment!(Ls, A, B, C, i-1; random_env, p)
            end
        end

        for sweep in 1:nsweeps
            tot_disc = 0.0

            if sweep % 2 == juliarank % 2 # Forward
                for i in first_site:last_site-1
                    i > 1 && leftenvironment!(Ls, A, B, C, i-1; random_env, p)

                    disc = updatecore!(A, B, C, i, get(Ls, i-1, ones(ValueType, 1, 1, 1)), get(Rs, i+2, ones(ValueType, 1, 1, 1));
                        method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, p)

                    tot_disc += disc
                end
            else
                for i in last_site-1:-1:first_site
                    i < n-1 && rightenvironment!(Rs, A, B, C, i+2; random_env, p)

                    disc = updatecore!(A, B, C, i, get(Ls, i-1, ones(ValueType, 1, 1, 1)), get(Rs, i+2, ones(ValueType, 1, 1, 1));
                        method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, p)

                    tot_disc += disc
                end
            end

            if MPI.Initialized()
                if juliarank % 2 == sweep % 2 # Left
                    if juliarank != nprocs

                        leftenvironment!(Ls, A, B, C, last_site-1; random_env, p)

                        sizes = Vector{Int}(undef, 3)
                        reqs = MPI.Isend(collect(size(Ls[last_site-1])), comm; dest=mpirank+1, tag=juliarank)
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank+1, tag=juliarank+1)

                        MPI.Waitall([reqs, reqr])

                        Rs[last_site+2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                        
                        reqs = MPI.Isend(Ls[last_site-1], comm; dest=mpirank+1, tag=juliarank)
                        reqr = MPI.Irecv!(Rs[last_site+2], comm; source=mpirank+1, tag=juliarank+1)

                        MPI.Waitall([reqs, reqr])
                        disc = updatecore!(A, B, C, last_site, Ls[last_site-1], Rs[last_site+2];
                            method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:backward, random_update)

                        Ci = TCI.sitetensor(C, last_site)
                        Yci = inversesingularvalue(C, last_site)
                        Cip1 = TCI.sitetensor(C, last_site+1)

                        reqs = MPI.Isend(collect(size(Ci)), comm; dest=mpirank+1, tag=3*juliarank)
                        reqs1 = MPI.Isend(collect(size(Yci)), comm; dest=mpirank+1, tag=3*juliarank+1)
                        reqs2 = MPI.Isend(collect(size(Cip1)), comm; dest=mpirank+1, tag=3*juliarank+2)

                        MPI.Waitall([reqs, reqs1, reqs2])

                        reqs = MPI.Isend(Ci, comm; dest=mpirank+1, tag=3*juliarank)
                        reqs1 = MPI.Isend(Yci, comm; dest=mpirank+1, tag=3*juliarank+1)
                        reqs2 = MPI.Isend(Cip1, comm; dest=mpirank+1, tag=3*juliarank+2)

                        MPI.Waitall([reqs, reqs1, reqs2])

                        rightenvironment!(Rs, A, B, C, last_site+1; random_env, p)
                    end
                else # Right
                    if juliarank != 1
                        rightenvironment!(Rs, A, B, C, first_site+1; random_env, p)

                        sizes = Vector{Int}(undef, 3)
                        reqs = MPI.Isend(collect(size(Rs[first_site+1])), comm; dest=mpirank-1, tag=juliarank)
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank-1, tag=juliarank-1)
                        
                        MPI.Waitall([reqs, reqr])

                        Ls[first_site-2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                        reqs = MPI.Isend(Rs[first_site+1], comm; dest=mpirank-1, tag=juliarank)
                        reqr = MPI.Irecv!(Ls[first_site-2], comm; source=mpirank-1, tag=juliarank-1)
                        
                        MPI.Waitall([reqs, reqr])

                        sizes = Vector{Int}(undef, 4)
                        sizes1 = Vector{Int}(undef, 2)
                        sizes2 = Vector{Int}(undef, 4)
                        
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank-1, tag=3*(juliarank-1))
                        reqr1 = MPI.Irecv!(sizes1, comm; source=mpirank-1, tag=3*(juliarank-1)+1)
                        reqr2 = MPI.Irecv!(sizes2, comm; source=mpirank-1, tag=3*(juliarank-1)+2)
                        
                        MPI.Waitall([reqr, reqr1, reqr2])
                        
                        Ci = ones(ValueType, sizes[1], sizes[2], sizes[3], sizes[4])
                        Yci = ones(ValueType, sizes1[1], sizes1[2])
                        Cip1 = ones(ValueType, sizes2[1], sizes2[2], sizes2[3], sizes2[4])

                        reqr = MPI.Irecv!(Ci, comm; source=mpirank-1, tag=3*(juliarank-1))
                        reqr1 = MPI.Irecv!(Yci, comm; source=mpirank-1, tag=3*(juliarank-1)+1)
                        reqr2 = MPI.Irecv!(Cip1, comm; source=mpirank-1, tag=3*(juliarank-1)+2)

                        MPI.Waitall([reqr, reqr1, reqr2])

                        settwositetensors!(C, first_site-1, Ci, Yci, Cip1)

                        leftenvironment!(Ls, A, B, C, first_site-1; random_env, p)
                    end
                end
            end

            tot_disc += disc

            converged = tot_disc < tolerance
            global_converged = MPI.Allreduce(converged, MPI.LAND, comm)
            if global_converged && sweep >= nprocs
                break
            end
        end
    end
    

    # Redistribute the tensor train among the processes.
    if synchedoutput
        if juliarank == 1
            received_sitetensors = Vector{Array{ValueType,4}}(undef, n)
            received_inversesingularvalues = Vector{Matrix}(undef, n-1)
            for j in 2:nprocs
                received_sitetensors[partitions[j][1]:partitions[j][end]] = MPI.recv(comm; source=j-1, tag=1)
                received_inversesingularvalues[partitions[j][1]:partitions[j][end]-1] = MPI.recv(comm; source=j-1, tag=2)
            end
        else
            send_sitetensors = sitetensors(C)[partitions[juliarank][1]:partitions[juliarank][end]]
            send_inversesingularvalues = inversesingularvalues(C)[partitions[juliarank][1]:partitions[juliarank][end]-1]
            MPI.send(send_sitetensors, comm; dest=0, tag=1)
            MPI.send(send_inversesingularvalues, comm; dest=0, tag=2)
        end

        nprocs = MPI.Comm_size(comm) # In case not all processes where used to compute
        
        if juliarank == 1
            received_sitetensors[1:partitions[1][end]] = sitetensors(C)[1:partitions[1][end]]
            received_inversesingularvalues[1:partitions[1][end]-1] = inversesingularvalues(C)[1:partitions[1][end]-1]
            for j in 2:nprocs
                MPI.send(received_sitetensors, comm; dest=j-1, tag=1)
                MPI.send(received_inversesingularvalues, comm; dest=j-1, tag=2)
            end
        else
            received_sitetensors = MPI.recv(comm; source=0, tag=1)
            received_inversesingularvalues = MPI.recv(comm; source=0, tag=2)
        end

        return TCI.InverseTensorTrain{ValueType,4}(received_sitetensors, received_inversesingularvalues, 1:n)
    end

    return C
end

"""
    function contract_distr_fit(A::TCI.TensorTrain{ValueType,4}, B::TCI.TensorTrain{ValueType,4})

Conctractos tensor trains A and B using the fit algorithm.

# Keyword Arguments

- `nsweep::Int`: Number of sweeps to perform during the algorithm.
- `initial_guess`: Initial guess of the solution A*B.
- `subcomm::Union{Nothing, MPI.Comm}`: Subcommunicator to use for the distributed computation. If nothing, the global communicator is used.
- `tolerance::Float64`: Convergence tolerance for the iterative algorithm.
- `method::Symbol`: Algorithm or method to use for the computation :SVD, :RSVD, :LU, :CI.
- `maxbonddim::Int`: Maximum bond dimension allowed during the decomposition.
- `synchedinput::Bool`: Whether the input MPOs are synchronized. Use "true" if you know that the input MPOs are exactly the same across different nodes.
- `synchedoutput::Bool`: Whether the output MPOs must be synchronized. Use "true" if you want that all the nodes have the correct MPOs as output.
"""
function contract_distr_fit(
    A::SiteTensorTrain{ValueType,4},
    B::SiteTensorTrain{ValueType,4};
    C::SiteTensorTrain{ValueType,4}=deepcopy(A), #TODO try without deepcopy
    nsweeps::Int=8,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    partitions::Union{Nothing,Vector{UnitRange{Int}}}=nothing,
    tolerance::Float64=1e-12,
    method::Symbol=:RSVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    synchedinput::Bool=false,
    synchedoutput::Bool=true,
    random_update::Bool=false,
    random_env::Bool=false, p::Int=0,
    kwargs...
)::TCI.TensorTrain{ValueType,4} where {ValueType}
    if length(mpoA) != length(mpoB)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    # println("La tolerance usata e': $tolerance")
    if !synchedinput
        synchronize_tt!(A)
        synchronize_tt!(B)
        synchronize_tt!(C)
    end

    n = length(A)

    if MPI.Initialized()
        if subcomm != nothing
            comm = subcomm
        else
            comm = MPI.COMM_WORLD
        end
        mpirank = MPI.Comm_rank(comm)
        juliarank = mpirank + 1
        nprocs = MPI.Comm_size(comm)
        if partitions == nothing
            if n < 4
                println("Warning! The TT is too small to be parallelized.")
                return contract_fit(A, B; C, nsweeps, tolerance, method, maxbonddim, kwargs...)
            end
            if n < 6
                if nprocs > 2
                    println("Warning! The TT is too small to be parallelized with more than 2 nodes. Some nodes will not be used.")
                end
                nprocs = 2
                partitions = [1:2,3:n]
            elseif n == 6
                if nprocs == 2
                    partitions = [1:3,4:6]
                elseif nprocs == 3
                    partitions = [1:2,3:4,5:6]
                else
                    println("Warning! The TT is too small to be parallelized with more than 3 nodes. Some nodes will not be used.")
                    nprocs = 3
                    partitions = [1:2,3:4,5:6]
                end
            else 
                extra1 = 3 # "Hyperparameter"
                extraend = 3
                if nprocs > div(n - extra1 - extraend, 2) # It's just one update per node.
                    println("Warning! A TT of lenght L can use be parallelized with up to (L-$(extra1 + extraend))/2 nodes. Some nodes will not be used.")
                    # Each node has 2 cores. Except first_site and last_site who have 2+extra1 and 2+extraend+n%2
                    if n % 2 == 0
                        partitions = vcat([1:extra1+1],[extra1+i+1:extra1+i+2 for i in 1:2:n-extra1-extraend-2],[n-extraend:n])
                    else
                        partitions = vcat([1:extra1+1],[extra1+i+1:extra1+i+2 for i in 1:2:n-extra1-extraend-3],[n-1-extraend:n])
                    end
                    for _ in div(n - extra1 - extraend, 2)+1:nprocs
                        push!(partitions, 1:-1)
                    end
                    nprocs = div(n - extra1 - extraend, 2)
                else
                    number_of_sites, rem = divrem(n-extra1-extraend, nprocs)
                    sites = [number_of_sites for i in 1:nprocs]
                    sites[1] += extra1
                    sites[end] += extraend
                    # Distribute the remainder as evenly as possible
                    for i in 1:rem
                        if i % 2 == 1
                            sites[div(i,2)+2] += 1
                        else
                            sites[end-div(i,2)] += 1
                        end
                    end
                    partitions = [sum(sites[1:i-1])+1:sum(sites[1:i]) for i in 1:nprocs]
                end
            end
        end
    else
        println("Warning! Distributed strategy has been chosen, but MPI is not initialized, please use TCI.initializempi() before contract() and use TCI.finalizempi() afterwards")
        return contract_fit(A, B; C, nsweeps, tolerance, method, maxbonddim, kwargs...)
    end

    if nprocs == 1
        println("Warning! Distributed strategy has been chosen, but only one process is running")
        return contract_fit(A, B; C, nsweeps, tolerance, method, maxbonddim, kwargs...)
    end

    if juliarank <= nprocs

        setpartition!(A, partitions[juliarank])
        setpartition!(B, partitions[juliarank])
        setpartition!(C, partitions[juliarank])

        Ls = Vector{Array{ValueType, 3}}(undef, n)
        Rs = Vector{Array{ValueType, 3}}(undef, n)

        first_site = partitions[juliarank][1]
        last_site = partitions[juliarank][end]

        #Ls are left enviroments and Rs are right environments
        if first_site != 1
            Ls[first_site-1] = rand(ValueType,
                first(size(sitetensor(A, first_site))), first(size(sitetensor(A, first_site))), first(size(sitetensor(A, first_site))))
            Ls[first_site-1] = Ls[first_site-1] ./ sqrt(sum(Ls[first_site-1].^2))
        end
        if last_site != n
            Rs[last_site+1] = rand(ValueType,
                last(size(sitetensor(A, last_site))), last(size(sitetensor(B, last_site))), last(size(sitetensor(C, last_site))))
            Rs[last_site+1] = Rs[last_site+1] ./ sqrt(sum(Rs[last_site+1].^2))
        end

        if juliarank % 2 == 1 # Precompute right environment if going forward
            setcenter!(A, first_site)
            setcenter!(B, first_site)
            setcenter!(X, first_site)
            for i in last_site:-1:first_site+2
                rightenvironment!(Rs, A, B, C, i; random_env, p)
            end
        else # Precompute left environment if going backward
            setcenter!(A, last_site-1)
            setcenter!(B, last_site-1)
            setcenter!(X, last_site)
            for i in first_site:last_site-2
                leftenvironment!(Ls, A, B, C, i-1; random_env, p)
            end
        end

        for sweep in 1:nsweeps
            tot_disc = 0.0
            time_update = time_ns()

            if sweep % 2 == juliarank % 2
                for i in first_site:last_site-1

                    i > 1 && leftenvironment!(Ls, A, B, C, i-1; random_env, p)

                    disc = updatecore!(A, B, C, i, get(Ls, i-1, ones(ValueType, 1, 1, 1)), get(Rs, i+2, ones(ValueType, 1, 1, 1));
                        method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, p)

                    tot_disc += disc
                end
            else
                for i in last_site-1:-1:first_site
                    i < n-1 && rightenvironment!(Rs, A, B, C, i+2; random_env, p)

                    disc, _, _ = updatecore!(A, B, C, i, get(Ls, i-1, ones(ValueType, 1, 1, 1)), get(Rs, i+2, ones(ValueType, 1, 1, 1));
                        method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, p)

                    tot_disc += disc
                end
            end

            time_communication = time_ns()
            if MPI.Initialized()
                if juliarank % 2 == sweep % 2 # Left
                    if juliarank != nprocs
                        leftenvironment!(Ls, A, B, C, last_site-1; random_env, p)

                        sizes = Vector{Int}(undef, 3)
                        reqs = MPI.Isend(collect(size(Ls[last_site-1])), comm; dest=mpirank+1, tag=juliarank)
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank+1, tag=juliarank+1)

                        MPI.Waitall([reqs, reqr])

                        Rs[last_site+2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                        
                        reqs = MPI.Isend(Ls[last_site-1], comm; dest=mpirank+1, tag=juliarank)
                        reqr = MPI.Irecv!(Rs[last_site+2], comm; source=mpirank+1, tag=juliarank+1)

                        MPI.Waitall([reqs, reqr])
                        disc = updatecore!(A, B, C, last_site, Ls[last_site-1], Rs[last_site+2];
                            method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:backward, random_update)

                        Ci = TCI.sitetensor(C, last_site)
                        Cip1 = TCI.sitetensor(C, last_site+1)

                        reqs = MPI.Isend(collect(size(Ci)), comm; dest=mpirank+1, tag=3*juliarank)
                        reqs1 = MPI.Isend(collect(size(Cip1)), comm; dest=mpirank+1, tag=3*juliarank+1)

                        MPI.Waitall([reqs, reqs1])

                        reqs = MPI.Isend(Ci, comm; dest=mpirank+1, tag=3*juliarank)
                        reqs1 = MPI.Isend(Cip1, comm; dest=mpirank+1, tag=3*juliarank+1)

                        MPI.Waitall([reqs, reqs1])
                        rightenvironment!(Rs, A, B, C, last_site+1; random_env, p)
                    end
                else # Right
                    if juliarank != 1
                        rightenvironment!(Rs, A, B, C, first_site+1; random_env, p)
                        sizes = Vector{Int}(undef, 3)
                        reqs = MPI.Isend(collect(size(Rs[first_site+1])), comm; dest=mpirank-1, tag=juliarank)
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank-1, tag=juliarank-1)
                        
                        MPI.Waitall([reqs, reqr])

                        Ls[first_site-2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                        reqs = MPI.Isend(Rs[first_site+1], comm; dest=mpirank-1, tag=juliarank)
                        reqr = MPI.Irecv!(Ls[first_site-2], comm; source=mpirank-1, tag=juliarank-1)
                        
                        MPI.Waitall([reqs, reqr])

                        
                        sizes = Vector{Int}(undef, 4)
                        sizes1 = Vector{Int}(undef, 2)
                        sizes2 = Vector{Int}(undef, 4)
                        
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank-1, tag=3*(juliarank-1))
                        reqr1 = MPI.Irecv!(sizes1, comm; source=mpirank-1, tag=3*(juliarank-1)+1)
                        
                        MPI.Waitall([reqr, reqr1])
                        
                        Ci = ones(ValueType, sizes[1], sizes[2], sizes[3], sizes[4])
                        Cip1 = ones(ValueType, sizes2[1], sizes2[2], sizes2[3], sizes2[4])

                        reqr = MPI.Irecv!(Ci, comm; source=mpirank-1, tag=3*(juliarank-1))
                        reqr1 = MPI.Irecv!(Cip1, comm; source=mpirank-1, tag=3*(juliarank-1)+1)
                        
                        MPI.Waitall([reqr, reqr1])

                        settwositetensors!(C, first_site-1, Ci, Cip1)

                        Ls[first_site-1] = leftenvironment(Ls, A, B, C, first_site-1; random_env, p)
                    end
                end
            end
            tot_disc += disc

            converged = tot_disc < tolerance
            global_converged = MPI.Allreduce(converged, MPI.LAND, comm)
            if global_converged
                break
            end
        end

        # Synchronize results across processes using tree-based communication
        rounds = ceil(Int, log2(nprocs))

        # TODO need to test this.
        for s in 1:rounds
            delta = 2^(s-1)
            
            if ((juliarank - 1) % (2*delta)) == 0 # I'm a receiver
            juliapartner = juliarank + delta
            
            if juliapartner <= nprocs
                my_first = partitions[juliarank][1]
                my_last = partitions[juliarank][end]
                partner_first = partitions[juliapartner][1]
                partner_last = partitions[min(juliapartner + delta - 1, nprocs)][end]
                
                # Receive partner's portion of C
                sizes = Vector{Int}(undef, partner_last - partner_first + 1)
                for i in partner_first:partner_last
                    sizes_i = MPI.recv(comm; source=juliapartner-1, tag=100*s + i)
                    
                    C_i = ones(ValueType, sizes_i[1], sizes_i[2], sizes_i[3], sizes_i[4])
                    MPI.recv!(C_i, comm; source=juliapartner-1, tag=200*s + i)
                    
                    TCI.setsitetensor!(C, i, C_i)
                end
                
                # Update partitions to include partner's range
                setpartition!(C, my_first:partner_last)
            end
            
            elseif ((juliarank - 1) % (2*delta)) == delta # I'm a sender
            juliapartner = juliarank - delta
            
            if juliapartner >= 1
                my_first = partitions[juliarank][1]
                my_last = partitions[min(juliarank + delta - 1, nprocs)][end]
                
                # Send my portion of C to partner
                for i in my_first:my_last
                C_i = TCI.sitetensor(C, i)
                MPI.send(collect(size(C_i)), comm; dest=juliapartner-1, tag=100*s + i)
                MPI.send(C_i, comm; dest=juliapartner-1, tag=200*s + i)
                end
            end
            end
            
            MPI.Barrier(comm)
        end

    end
    nprocs = MPI.Comm_size(comm) # In case not all processes where used to compute

    # Redistribute the tensor train among the processes.
    if synchedoutput
        if juliarank == 1
            received_sitetensors = Vector{Array{ValueType,4}}(undef, n)
            for j in 2:nprocs
                received_sitetensors[partitions[j][1]:partitions[j][end]] = MPI.recv(comm; source=j-1, tag=1)
            end
        else
            send_sitetensors = sitetensors(C)[partitions[juliarank][1]:partitions[juliarank][end]]
            MPI.send(send_sitetensors, comm; dest=0, tag=1)
        end

        nprocs = MPI.Comm_size(comm) # In case not all processes where used to compute
        
        if juliarank == 1
            received_sitetensors[1:partitions[1][end]] = sitetensors(C)[1:partitions[1][end]]
            for j in 2:nprocs
                MPI.send(received_sitetensors, comm; dest=j-1, tag=1)
            end
        else
            received_sitetensors = MPI.recv(comm; source=0, tag=1)
        end

        return TCI.SiteTensorTrain{ValueType,4}(received_sitetensors, 1:n)
    end

    return C
end

"""
    function contract(
        A::TCI.TensorTrain{ValueType1,4},
        B::TCI.TensorTrain{ValueType2,4};
        algorithm::Symbol=:TCI,
        tolerance::Float64=1e-12,
        maxbonddim::Int=typemax(Int),
        f::Union{Nothing,Function}=nothing,
        subcomm::Union{Nothing, MPI.Comm}=nothing,
        kwargs...
    ) where {ValueType1,ValueType2}

Contract two tensor trains `A` and `B`.

Currently, two implementations are available:
 1. `algorithm=:TCI` constructs a new TCI that fits the contraction of `A` and `B`.
 2. `algorithm=:naive` uses a naive tensor contraction and subsequent SVD recompression of the tensor train.
 3. `algorithm=:zipup` uses a naive tensor contraction with on-the-fly LU decomposition.
 4. `algorithm=:distrzipup` uses a naive tensor contraction with on-the-fly LU decomposition, distributed on multiple nodes.
 5. `algorithm=:fit` uses a variational approach to minimize the difference between `A`*`B` and the solution.
 6. `algorithm=:distrfit` uses a variational approach to minimize the difference between `A`*`B` and the solution, distributed on multiple nodes.


Arguments:
- `A` and `B` are the tensor trains to be contracted.
- `algorithm` chooses the algorithm used to evaluate the contraction.
- `tolerance` is the tolerance of the TCI or SVD recompression.
- `maxbonddim` sets the maximum bond dimension of the resulting tensor train.
- `f` is a function to be applied elementwise to the result. This option is only available with `algorithm=:TCI`.
- `method` chooses the method used for the factorization in the `algorithm=:zipup` case (`:SVD` or `:LU`).
- `subcomm` is an optional MPI communicator for distributed algorithms. If not provided, the default communicator is used.
- `kwargs...` are forwarded to [`crossinterpolate2`](@ref) if `algorithm=:TCI` or to [`contract_fit`](@ref) if `algorithm=:fit` or `algorithm=:distrfit`.
"""
function contract(
    A::TCI.TensorTrain{ValueType1,4},
    B::TCI.TensorTrain{ValueType2,4};
    algorithm::Symbol=:naive,
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    method::Symbol=:SVD,
    f::Union{Nothing,Function}=nothing,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    kwargs...
)::TCI.TensorTrain{promote_type(ValueType1,ValueType2),4} where {ValueType1,ValueType2}
    Vres = promote_type(ValueType1, ValueType2)
    if algorithm === :TCI
        mpo = contract_TCI(A, B; tolerance=tolerance, maxbonddim=maxbonddim, f=f, kwargs...)
    elseif algorithm === :naive
        if f !== nothing
            error("Naive contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        mpo = contract_naive(A, B; tolerance=tolerance, maxbonddim=maxbonddim)
    elseif algorithm === :zipup
        if f !== nothing
            error("Zipup contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        mpo = contract_zipup(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    elseif algorithm === :fit || algorithm === :distrfit
        error("algorithm=:$algorithm requires SiteTensorTrain or InverseTensorTrain types. Convert your TensorTrain first or use algorithm=:TCI, :naive, or :zipup.")
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))
    end
    return mpo
end

# Contract for SiteTensorTrain: supports :fit and :distrfit
function contract(
    A::SiteTensorTrain{ValueType1,4},
    B::SiteTensorTrain{ValueType2,4};
    algorithm::Symbol=:fit,
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    method::Symbol=:SVD,
    f::Union{Nothing,Function}=nothing,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    kwargs...
)::TCI.TensorTrain{promote_type(ValueType1,ValueType2),4} where {ValueType1,ValueType2}
    if f !== nothing
        error("Fit/distributed fit contraction cannot use a function. Use algorithm=:TCI with TCI.TensorTrain instead.")
    end
    
    if algorithm === :fit
        mpo = contract_fit(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    elseif algorithm === :distrfit
        mpo = contract_distr_fit(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method, subcomm=subcomm, kwargs...)
    elseif algorithm === :TCI || algorithm === :naive || algorithm === :zipup
        # Convert to TensorTrain for these algorithms
        mpo = contract(TCI.TensorTrain(TCI.sitetensors(A)), TCI.TensorTrain(TCI.sitetensors(B)); 
                      algorithm=algorithm, tolerance=tolerance, maxbonddim=maxbonddim, method=method, f=f, kwargs...)
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))
    end
    return mpo
end

# Contract for InverseTensorTrain: supports :fit and :distrfit
function contract(
    A::InverseTensorTrain{ValueType1,4},
    B::InverseTensorTrain{ValueType2,4};
    algorithm::Symbol=:fit,
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    method::Symbol=:SVD,
    f::Union{Nothing,Function}=nothing,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    kwargs...
)::TCI.TensorTrain{promote_type(ValueType1,ValueType2),4} where {ValueType1,ValueType2}
    if f !== nothing
        error("Fit/distributed fit contraction cannot use a function. Use algorithm=:TCI with TCI.TensorTrain instead.")
    end
    
    if algorithm === :fit
        mpo = contract_fit(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    elseif algorithm === :distrfit
        mpo = contract_distr_fit(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method, subcomm=subcomm, kwargs...)
    elseif algorithm === :TCI || algorithm === :naive || algorithm === :zipup
        # Convert to TensorTrain for these algorithms
        mpo = contract(TCI.TensorTrain(A), TCI.TensorTrain(B); # TODO, these needs to be created 
                      algorithm=algorithm, tolerance=tolerance, maxbonddim=maxbonddim, method=method, f=f, kwargs...)
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))
    end
    return mpo
end

function contract(
    A::Union{TCI.TensorCI1{ValueType1},TCI.TensorCI2{ValueType1},TCI.TensorTrain{ValueType1,3}},
    B::TCI.TensorTrain{ValueType2,4};
    kwargs...
)::TCI.TensorTrain{promote_type(ValueType1,ValueType2),3} where {ValueType1,ValueType2}
    tt = contract(TCI.TensorTrain{4}(A, [(1, s...) for s in sitedims(A)]), B; kwargs...)
    return TCI.TensorTrain{3}(tt, prod.(sitedims(tt)))
end

function contract(
    A::TCI.TensorTrain{ValueType1,4},
    B::Union{TCI.TensorCI1{ValueType2},TCI.TensorCI2{ValueType2},TCI.TensorTrain{ValueType2,3}};
    kwargs...
)::TCI.TensorTrain{promote_type(ValueType1,ValueType2),3} where {ValueType1,ValueType2}
    tt = contract(A, TCI.TensorTrain{4}(B, [(s..., 1) for s in sitedims(B)]); kwargs...)
    return TCI.TensorTrain{3}(tt, prod.(sitedims(tt)))
end

# TODO SiteTensorTrain and TensorTrain, who wins? Right now not possible to mix
function contract(
    A::Union{TCI.TensorCI1{ValueType1},TCI.TensorCI2{ValueType1},SiteTensorTrain{ValueType1,3}},
    B::SiteTensorTrain{ValueType2,4};
    kwargs...
)::SiteTensorTrain{promote_type(ValueType1,ValueType2),3} where {ValueType1,ValueType2}
    tt = contract(SiteTensorTrain{4}(A, [(1, s...) for s in sitedims(A)]), B; kwargs...)
    return SiteTensorTrain{3}(tt, prod.(sitedims(tt)))
end

function contract(
    A::SiteTensorTrain{ValueType1,4},
    B::Union{TCI.TensorCI1{ValueType2},TCI.TensorCI2{ValueType2},SiteTensorTrain{ValueType2,3}};
    kwargs...
)::SiteTensorTrain{promote_type(ValueType1,ValueType2),3} where {ValueType1,ValueType2}
    tt = contract(A, SiteTensorTrain{4}(B, [(s..., 1) for s in sitedims(B)]); kwargs...)
    return SiteTensorTrain{3}(tt, prod.(sitedims(tt)))
end

# It's a scalar
function contract(
    A::Union{TCI.TensorCI1{ValueType1},TCI.TensorCI2{ValueType1},TCI.TensorTrain{ValueType1,3},SiteTensorTrain{ValueType1,3}},
    B::Union{TCI.TensorCI1{ValueType2},TCI.TensorCI2{ValueType2},TCI.TensorTrain{ValueType2,3},SiteTensorTrain{ValueType2,3}};
    kwargs...
)::promote_type(ValueType1,ValueType2) where {ValueType1,ValueType2}
    tt = contract(TCI.TensorTrain{4}(A, [(1, s...) for s in sitedims(A)]), TCI.TensorTrain{4}(B, [(s..., 1) for s in sitedims(B)]); kwargs...)
    return prod(prod.(tt.sitetensors))
end

# InverseTensorTrain with 3D tensors (upgrade to 4D)
function contract(
    A::InverseTensorTrain{ValueType1,3},
    B::InverseTensorTrain{ValueType2,3};
    kwargs...
)::promote_type(ValueType1,ValueType2) where {ValueType1,ValueType2}
    # Upgrade to 4D
    A4 = InverseTensorTrain{ValueType1,4}([reshape(t, 1, size(t)...) for t in TCI.sitetensors(A)], inversesingularvalues(A), partition(A))
    B4 = InverseTensorTrain{ValueType2,4}([reshape(t, size(t)..., 1) for t in TCI.sitetensors(B)], inversesingularvalues(B), partition(B))
    tt = TCI.TensorTrain{4}(contract(A4, B4; kwargs...)) # TODO actually create (JET will throw an error)
    return prod(prod.(tt.sitetensors))
end

function contract(
    A::InverseTensorTrain{ValueType1,4},
    B::InverseTensorTrain{ValueType2,3};
    kwargs...
)::TCI.TensorTrain{promote_type(ValueType1,ValueType2),3} where {ValueType1,ValueType2}
    # Upgrade B to 4D
    B4 = InverseTensorTrain{ValueType2,4}([reshape(t, size(t)..., 1) for t in TCI.sitetensors(B)], inversesingularvalues(B), partition(B))
    tt = contract(A, B4; kwargs...)
    return InverseTensorTrain{3}(tt, prod.(sitedims(tt)))
end

function contract(
    A::InverseTensorTrain{ValueType1,3},
    B::InverseTensorTrain{ValueType2,4};
    kwargs...
)::TCI.TensorTrain{promote_type(ValueType1,ValueType2),3} where {ValueType1,ValueType2}
    # Upgrade A to 4D
    A4 = InverseTensorTrain{ValueType1,4}([reshape(t, 1, size(t)...) for t in TCI.sitetensors(A)], inversesingularvalues(A), partition(A))
    tt = contract(A4, B; kwargs...)
    return InverseTensorTrain{3}(tt, prod.(sitedims(tt)))
end