function _factorize(
    A::AbstractMatrix{V}, method::Symbol; tolerance::Float64, maxbonddim::Int, leftorthogonal::Bool=false, diamond::Bool=false, normalizeerror=true, q::Int=0, p::Int=16
)::Union{Tuple{Matrix{V},Matrix{V},Int,Float64},Tuple{Matrix{V},Vector{Float64},Matrix{V},Int,Float64},Tuple{Matrix{V},Vector{Float64},Matrix{V},Int},Tuple{Matrix{V},Matrix{V},Int}} where {V}
    reltol = 1e-14
    abstol = 0.0
    if normalizeerror
        reltol = tolerance
    else
        abstol = tolerance
    end
    if method === :LU
        factorization = TCI.rrlu(A, abstol=abstol, reltol=reltol, maxrank=maxbonddim, leftorthogonal=leftorthogonal)
        if TCI.npivots(factorization) == maxbonddim # TODO is this conceptually right?
            disc = Inf
        else
            disc = 0.0
        end
        return TCI.left(factorization), TCI.right(factorization), TCI.npivots(factorization), disc
    elseif method === :CI
        factorization = TCI.MatrixLUCI(A, abstol=abstol, reltol=reltol, maxrank=maxbonddim, leftorthogonal=leftorthogonal)
        if TCI.npivots(factorization) == maxbonddim # TODO is this conceptually right?
            disc = Inf
        else
            disc = 0.0
        end
        return TCI.left(factorization), TCI.right(factorization), TCI.npivots(factorization), disc
    elseif method === :SVD
        factorization = LinearAlgebra.svd(A)
        err = [sum(factorization.S[n+1:end] .^ 2) for n in 1:length(factorization.S)]
        normalized_err = err ./ sum(factorization.S .^ 2)

        trunci = min(
            TCI.replacenothing(findfirst(<(abstol^2), err), length(err)),
            TCI.replacenothing(findfirst(<(reltol^2), normalized_err), length(normalized_err)),
            maxbonddim
        )

        if length(factorization.S) > trunci
            if normalizeerror
                disc = sqrt(normalized_err[trunci])
            else
                disc = sqrt(err[trunci])
            end
        else
            disc = 0.0
        end

        if diamond
            return (
                    Matrix(factorization.U[:, 1:trunci]),
                    factorization.S[1:trunci],
                    Matrix(factorization.Vt[1:trunci, :]),
                    trunci,
                    disc
                )
        else
            if leftorthogonal
                return (
                    Matrix(factorization.U[:, 1:trunci]),
                    Matrix(Diagonal(factorization.S[1:trunci]) * factorization.Vt[1:trunci, :]),
                    trunci,
                    disc
                )
            else
                return (
                    Matrix(factorization.U[:, 1:trunci] * Diagonal(factorization.S[1:trunci])),
                    Matrix(factorization.Vt[1:trunci, :]),
                    trunci,
                    disc
                )
            end
        end
    elseif method === :RSVD
        invert = false
        if size(A)[1] < size(A)[2]
            A = A'
            invert = true
        end
        if maxbonddim == typemax(Int) || maxbonddim + p > size(A)[1] || maxbonddim + p > size(A)[2]
            factorization = LinearAlgebra.svd(A)
        else
            m, n = size(A)
            G = randn(n, maxbonddim + p)
            Y = A*G 
            Q = Matrix(LinearAlgebra.qr!(Y).Q) # THEORETICAL BOTTLENECK
            for _ in 1:q # q=0 for best performance
                Y = A'*Q
                Q = Matrix(LinearAlgebra.qr!(Y).Q)
                Y = A*Q
                Q = Matrix(LinearAlgebra.qr!(Y).Q)
            end
            B = Q' * A
            factorization = LinearAlgebra.svd(B)
            factorization = SVD((Q*factorization.U)[:,1:maxbonddim], factorization.S[1:maxbonddim], factorization.Vt[1:maxbonddim,:])
        end
        err = [sum(factorization.S[n+1:end] .^ 2) for n in 1:length(factorization.S)]
        normalized_err = err ./ sum(factorization.S .^ 2)

        trunci = min(
            TCI.replacenothing(findfirst(<(abstol^2), err), length(err)),
            TCI.replacenothing(findfirst(<(reltol^2), normalized_err), length(normalized_err)),
            maxbonddim
        )

        if length(factorization.S) > trunci
            if normalizeerror
                disc = sqrt(normalized_err[trunci])
            else
                disc = sqrt(err[trunci])
            end
        else
            disc = 0.0
        end


        if diamond
            if invert
                return (
                    Matrix(factorization.Vt[1:trunci, :]'),
                    factorization.S[1:trunci],
                    Matrix(factorization.U[:, 1:trunci]'),
                    trunci, disc
                )
            else
                return (
                    Matrix(factorization.U[:, 1:trunci]),
                    factorization.S[1:trunci],
                    Matrix(factorization.Vt[1:trunci, :]),
                    trunci, disc
                )
            end
        else
            if leftorthogonal
                if invert
                    return (
                        Matrix(factorization.Vt[1:trunci, :]' * Diagonal(factorization.S[1:trunci])),
                        Matrix(factorization.U[:, 1:trunci]'),
                        trunci, disc
                    )
                else
                    return (
                        Matrix(factorization.U[:, 1:trunci]),
                        Matrix(Diagonal(factorization.S[1:trunci]) * factorization.Vt[1:trunci, :]),
                        trunci, disc
                    )
                end
            else
                if invert
                    return (
                        Matrix(factorization.Vt[1:trunci, :]' * Diagonal(factorization.S[1:trunci])),
                        Matrix(factorization.U[:, 1:trunci]'),
                        trunci, disc
                    )
                else
                    return (
                        Matrix(factorization.U[:, 1:trunci] * Diagonal(factorization.S[1:trunci])),
                        Matrix(factorization.Vt[1:trunci, :]),
                        trunci, disc
                    )
                end
            end
        end
    else
        error("Not implemented yet.")
    end
end

function IMPO(R::Int)
    return TCI.TensorTrain{Float64, 4}([reshape([1.,0.,0.,1.], 1,2,2,1) for _ in 1:R])
end

function reshapephysicalright(T::AbstractArray{ValueType, N}) where {ValueType, N}
    return reshape(T, first(size(T)), :)
end

function reshapephysicalleft(T::AbstractArray{ValueType, N}) where {ValueType, N}
    return reshape(T, :, last(size(T)))
end