# Import base functionality from T4ATensorCI
import T4ATensorCI: submatrixargmax, rrLU, swaprow!, swapcol!, addpivot!, _optimizerrlu!, rrlu!, rrlu, cols2Lmatrix!, rows2Umatrix!, size, left, right, diag, rowindices, colindices, npivots, pivoterrors, lastpivoterror, solve, pushrandomsubset!
import T4ATensorCI: arrlu as arrlu_base

# MPI-specific function for distributed computation
function distribute(_batchf, full, fragmented, teamsize, teamrank, stripes, subcomm)
    chunksize, remainder = divrem(length(fragmented), teamsize)
    if stripes == :vertical
        col_chunks = [chunksize + (i <= remainder ? 1 : 0) for i in 1:teamsize]
        row_chunks = [length(full) for _ in 1:teamsize]
        ranges = [(vcat([0],cumsum(col_chunks))[i] + 1):(cumsum(col_chunks)[i]) for i in 1:teamsize]
    else # horizontal
        row_chunks = [chunksize + (i <= remainder ? 1 : 0) for i in 1:teamsize]
        col_chunks = [length(full) for _ in 1:teamsize]
        ranges = [(vcat([0],cumsum(row_chunks))[i] + 1):(cumsum(row_chunks)[i]) for i in 1:teamsize]
    end
    fragment = fragmented[ranges[teamrank]]
    localsubmatrix = if isempty(fragment)
        zeros(length(full), 0)
    else
        if stripes == :vertical
            _batchf(full, fragment)
        else
            _batchf(fragment, full)
        end
    end
    if teamrank == 1
        submatrix = if stripes == :vertical
            zeros(length(full), length(fragmented))
        else
            zeros(length(fragmented), length(full))
        end
        sizes = vcat(row_chunks', col_chunks')
        counts = vec(prod(sizes, dims=1))
        submatrix_vbuf = MPI.VBuffer(submatrix, counts)
    else
        submatrix_vbuf = MPI.VBuffer(nothing)
    end
    MPI.Gatherv!(localsubmatrix, submatrix_vbuf, 0, subcomm)
    if teamrank != 1
        submatrix = nothing
    end
    submatrix
end

# Extended arrlu with MPI support
function arrlu(
    ::Type{ValueType},
    f,
    matrixsize::Tuple{Int,Int},
    I0::AbstractVector{Int}=Int[],
    J0::AbstractVector{Int}=Int[];
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true,
    numrookiter::Int=5,
    usebatcheval::Bool=false,
    leaders::Vector=Int[],
    leaderslist::Vector=Int[],
    subcomm=nothing # TODO change: bad practice
)::rrLU{ValueType} where {ValueType}
    # If no MPI, fall back to base implementation
    if isempty(leaders)
        return arrlu_base(ValueType, f, matrixsize, I0, J0; 
                         maxrank=maxrank, reltol=reltol, abstol=abstol,
                         leftorthogonal=leftorthogonal, numrookiter=numrookiter,
                         usebatcheval=usebatcheval)
    end
    
    # MPI-enabled version
    lu = rrLU{ValueType}(matrixsize...; leftorthogonal)
    islowrank = false
    maxrank = min(maxrank, matrixsize...)

    _batchf = usebatcheval ? f : ((x, y) -> f.(x, y'))

        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        juliarank = rank + 1
        nprocs = MPI.Comm_size(comm)
        teamrank = MPI.Comm_rank(subcomm)
        teamjuliarank = teamrank + 1
        teamsize = MPI.Comm_size(subcomm)

    while true
        if leftorthogonal
            pushrandomsubset!(J0, 1:matrixsize[2], max(1, length(J0)))
        else
            pushrandomsubset!(I0, 1:matrixsize[1], max(1, length(I0)))
        end
        
        for rookiter in 1:numrookiter
            colmove = (iseven(rookiter) == leftorthogonal)
                if teamsize > 1 # Parallel and leadership
                    submatrix = if colmove
                        distribute(_batchf, lu.colpermutation, I0, teamsize, teamjuliarank, :horizontal, subcomm)
                    else
                        distribute(_batchf, lu.rowpermutation, J0, teamsize, teamjuliarank, :vertical, subcomm)
                    end
                    MPI.Barrier(subcomm)
                    if teamjuliarank == 1
                        lu.npivot = 0
                        _optimizerrlu!(lu, submatrix; maxrank, reltol, abstol)
                        islowrank |= npivots(lu) < minimum(size(submatrix))
                    end
                else
                    submatrix = if colmove
                        _batchf(I0, lu.colpermutation)
                    else
                        _batchf(lu.rowpermutation, J0)
                    end
                    lu.npivot = 0
                    _optimizerrlu!(lu, submatrix; maxrank, reltol, abstol)
                    islowrank |= npivots(lu) < minimum(size(submatrix))
                end    
            
                tb = time_ns()
                lu = MPI.bcast([lu], subcomm)[1]
                islowrank = MPI.bcast([islowrank], subcomm)[1]
                tb = (time_ns() - tb)*1e-9
                if rowindices(lu) == I0 && colindices(lu) == J0
                    break
                end
                J0 = colindices(lu)
                I0 = rowindices(lu)
        end

        
        if islowrank || length(I0) >= maxrank
            break
        end
    end

    
    if size(lu.L, 1) < matrixsize[1]
        I2 = setdiff(1:matrixsize[1], I0)
        lu.rowpermutation = vcat(I0, I2)
        L2 = distribute(_batchf, J0, I2, teamsize, teamjuliarank, :horizontal, subcomm)
        if teamjuliarank == 1
            cols2Lmatrix!(L2, (@view lu.U[1:lu.npivot, 1:lu.npivot]), leftorthogonal)
            lu.L = vcat((@view lu.L[1:lu.npivot, 1:lu.npivot]), L2)
        end
    end


    if size(lu.U, 2) < matrixsize[2]
        J2 = setdiff(1:matrixsize[2], J0)
        lu.colpermutation = vcat(J0, J2)
        U2 = distribute(_batchf, I0, J2, teamsize, teamjuliarank, :vertical, subcomm)
        if teamjuliarank == 1
            rows2Umatrix!(U2, (@view lu.L[1:lu.npivot, 1:lu.npivot]), leftorthogonal)
            lu.U = hcat((@view lu.U[1:lu.npivot, 1:lu.npivot]), U2)
        end
    end

    return lu
end

# Export the extended arrlu function
export arrlu
