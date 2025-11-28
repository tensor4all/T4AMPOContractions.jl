function _tomat(tto::TCI.TensorTrain{T,4}) where {T}
    sitedims = TCI.sitedims(tto)
    localdims1 = [s[1] for s in sitedims]
    localdims2 = [s[2] for s in sitedims]
    mat = Matrix{T}(undef, prod(localdims1), prod(localdims2))
    for (i, inds1) in enumerate(CartesianIndices(Tuple(localdims1)))
        for (j, inds2) in enumerate(CartesianIndices(Tuple(localdims2)))
            mat[i, j] = TCI.evaluate(tto, collect(zip(Tuple(inds1), Tuple(inds2))))
        end
    end
    return mat
end

function _tovec(tt::TCI.TensorTrain{T,3}) where {T}
    sitedims = TCI.sitedims(tt)
    localdims1 = [s[1] for s in sitedims]
    return TCI.evaluate.(Ref(tt), CartesianIndices(Tuple(localdims1))[:])
end

function _gen_testdata_TTO_TTO()
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [5, 5, 5, 5]

    a = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1])
        for n = 1:N
    ])
    return N, a, b, localdims1, localdims2, localdims3
end

function _gen_testdata_TTO_TTS()
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [3, 3, 3, 3]
    localdims2 = [3, 3, 3, 3]

    a = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TCI.TensorTrain{ComplexF64,3}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], bonddims_b[n+1])
        for n = 1:N
    ])
    return N, a, b, localdims1, localdims2
end


@testset "contractions" begin
    @testset "_contract" begin
        a = rand(2, 3, 4)
        b = rand(2, 5, 4)
        ab = MPO._contract(a, b, (1, 3), (1, 3))
        @test vec(reshape(permutedims(a, (2, 1, 3)), 3, :) * reshape(permutedims(b, (1, 3, 2)), :, 5)) ≈ vec(ab)
    end

    @testset "zipup environment" begin
        a = rand(2, 3, 4)
        b = rand(2, 5, 4)
        ab = MPO._contract(a, b, (1, 3), (1, 3))
        @test vec(reshape(permutedims(a, (2, 1, 3)), 3, :) * reshape(permutedims(b, (1, 3, 2)), :, 5)) ≈ vec(ab)
    end

    @testset "MPO-MPO contraction (TCI, naive)" for f in [nothing, x -> 2 * x], algorithm in [:TCI, :naive]
        N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()

        if f !== nothing && algorithm === :naive
            @test_throws ErrorException MPO.contract(a, b; f=f, algorithm=algorithm)
        else
            ab = MPO.contract(a, b; f=f, algorithm=algorithm)
            @test TCI.sitedims(ab) == [[localdims1[i], localdims3[i]] for i = 1:N]
            if f === nothing
                @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
            else
                @test _tomat(ab) ≈ f.(_tomat(a) * _tomat(b))
            end
        end
    end

    @testset "Contraction, batchevaluate" begin
        N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()

        ab = MPO.Contraction(a, b)
        leftindexset = [[1]]
        rightindexset = [[1]]

        ref = ab(leftindexset, rightindexset, Val(2))
        ref_multiindex = reshape(ref, length(leftindexset), 2, 2, 2, 2, length(rightindexset))

        let
            res = TCI.batchevaluate(ab, leftindexset, rightindexset, Val(2), [[0, 0], [1, 0]])
            @test vec(ref_multiindex[:, :, :, 1, :, :]) ≈ vec(res)
        end

        let
            res = TCI.batchevaluate(ab, leftindexset, rightindexset, Val(2), [[0, 0], [1, 1]])
            @test vec(ref_multiindex[:, :, :, 1, 1, :]) ≈ vec(res)
        end

        let
            res = TCI.batchevaluate(ab, leftindexset, rightindexset, Val(2), [[0, 1], [1, 0]])
            @test vec(ref_multiindex[:, :, 1, 1, :, :]) ≈ vec(res)
        end
    end

    @testset "MPO-MPS contraction (TCI, naive)" for f in [nothing, x -> 2 * x], algorithm in [:TCI, :naive]
        N, a, b, localdims1, localdims2 = _gen_testdata_TTO_TTS()

        if f !== nothing && algorithm === :naive
            @test_throws ErrorException MPO.contract(a, b; f=f, algorithm=algorithm)
            @test_throws ErrorException MPO.contract(b, a; f=f, algorithm=algorithm)
        else
            ab = MPO.contract(a, b; f=f, algorithm=algorithm)
            ba = MPO.contract(b, a; f=f, algorithm=algorithm)
            @test TCI.sitedims(ab) == [[localdims1[i]] for i = 1:N]
            if f === nothing
                @test _tovec(ab) ≈ _tomat(a) * _tovec(b)
                @test transpose(_tovec(ba)) ≈ transpose(_tovec(b)) * _tomat(a)
            else
                @test _tovec(ab) ≈ f.(_tomat(a) * _tovec(b))
                @test transpose(_tovec(ba)) ≈ f.(transpose(_tovec(b)) * _tomat(a))
            end
        end
    end

    @testset "rightenvironment site" begin
        Random.seed!(42)

        Rs = Vector{Array{ComplexF64,3}}(undef, 3)
        As = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)] 
        A = MPO.SiteTensorTrain{ComplexF64, 4}(As)
        Bs = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        B = MPO.SiteTensorTrain{ComplexF64, 4}(Bs)
        Cs = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]
        C = MPO.SiteTensorTrain{ComplexF64, 4}(Cs)

        As = TCI.sitetensors(A)
        Bs = TCI.sitetensors(B)
        Cs = TCI.sitetensors(C)

        R = ones(ComplexF64, 1, 1, 1)
        R3 = MPO._contract(conj(Cs[3]), R, (4,), (3,))
        R3 = MPO._contract(Bs[3], R3, (3,4,), (3,5,))
        R3 = MPO._contract(As[3], R3, (2,3,4), (4,2,5,))

        R2 = MPO._contract(conj(Cs[2]), R3, (4,), (3,))
        R2 = MPO._contract(Bs[2], R2, (3,4,), (3,5,))
        R2 = MPO._contract(As[2], R2, (2,3,4), (4,2,5,))

        MPO.rightenvironment!(Rs, A, B, C, 3)
        MPO.rightenvironment!(Rs, A, B, C, 2)

        @test Rs[3] ≈ R3
        @test Rs[2] ≈ R2
    end

    @testset "leftenvironment site" begin
        Random.seed!(42)

        Ls = Vector{Array{ComplexF64,3}}(undef, 3)
        As = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)]
        A = MPO.SiteTensorTrain{ComplexF64, 4}(As)
        Bs = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        B = MPO.SiteTensorTrain{ComplexF64, 4}(Bs)
        Cs = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]
        C = MPO.SiteTensorTrain{ComplexF64, 4}(Cs)

        As = TCI.sitetensors(A)
        Bs = TCI.sitetensors(B)
        Cs = TCI.sitetensors(C)

        L = ones(ComplexF64, 1, 1, 1)
        L1 = MPO._contract(L, As[1], (1,), (1,))
        L1 = MPO._contract(L1, Bs[1], (1,4,), (1,2,))
        L1 = MPO._contract(L1, conj(Cs[1]), (1,2,4,), (1,2,3,))

        L2 = MPO._contract(L1, As[2], (1,), (1,))
        L2 = MPO._contract(L2, Bs[2], (1,4,), (1,2,))
        L2 = MPO._contract(L2, conj(Cs[2]), (1,2,4,), (1,2,3,))
        
        MPO.leftenvironment!(Ls, A, B, C, 1)
        MPO.leftenvironment!(Ls, A, B, C, 2)

        @test Ls[1] ≈ L1
        @test Ls[2] ≈ L2
    end

    @testset "rightenvironment inverse" begin
        Random.seed!(42)

        Rs = Vector{Array{ComplexF64,3}}(undef, 3)
        As = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)]
        Bs = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        Cs = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]

        A = MPO.InverseTensorTrain{ComplexF64, 4}(TCI.TensorTrain{ComplexF64,4}(As))
        B = MPO.InverseTensorTrain{ComplexF64, 4}(TCI.TensorTrain{ComplexF64,4}(Bs))
        C = MPO.InverseTensorTrain{ComplexF64, 4}(TCI.TensorTrain{ComplexF64,4}(Cs))

        As = TCI.sitetensors(A)
        Bs = TCI.sitetensors(B)
        Cs = TCI.sitetensors(C)
        Yas = MPO.inversesingularvalues(A)
        Ybs = MPO.inversesingularvalues(B)
        Ycs = MPO.inversesingularvalues(C)

        As[3] = MPO._contract(Yas[2], As[3], (2,), (1,))
        Bs[3] = MPO._contract(Ybs[2], Bs[3], (2,), (1,))
        Cs[3] = MPO._contract(Ycs[2], Cs[3], (2,), (1,))
        As[2] = MPO._contract(Yas[1], As[2], (2,), (1,))
        Bs[2] = MPO._contract(Ybs[1], Bs[2], (2,), (1,))
        Cs[2] = MPO._contract(Ycs[1], Cs[2], (2,), (1,))

        R = ones(ComplexF64, 1, 1, 1)
        R3 = MPO._contract(conj(Cs[3]), R, (4,), (3,))
        R3 = MPO._contract(Bs[3], R3, (3,4,), (3,5,))
        R3 = MPO._contract(As[3], R3, (2,3,4), (4,2,5,))

        R2 = MPO._contract(conj(Cs[2]), R3, (4,), (3,))
        R2 = MPO._contract(Bs[2], R2, (3,4,), (3,5,))
        R2 = MPO._contract(As[2], R2, (2,3,4), (4,2,5,))

        MPO.rightenvironment!(Rs, A, B, C, 3)
        MPO.rightenvironment!(Rs, A, B, C, 2)
        
        @test Rs[3] ≈ R3
        @test Rs[2] ≈ R2
    end

    @testset "leftenvironment inverse" begin
        Random.seed!(42)

        Ls = Vector{Array{ComplexF64,3}}(undef, 3)
        As = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)]
        Bs = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        Cs = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]

        A = MPO.InverseTensorTrain{ComplexF64, 4}(TCI.TensorTrain{ComplexF64,4}(As))
        B = MPO.InverseTensorTrain{ComplexF64, 4}(TCI.TensorTrain{ComplexF64,4}(Bs))
        C = MPO.InverseTensorTrain{ComplexF64, 4}(TCI.TensorTrain{ComplexF64,4}(Cs))

        As = TCI.sitetensors(A)
        Bs = TCI.sitetensors(B)
        Cs = TCI.sitetensors(C)
        Yas = MPO.inversesingularvalues(A)
        Ybs = MPO.inversesingularvalues(B)
        Ycs = MPO.inversesingularvalues(C)

        As[1] = MPO._contract(As[1], Yas[1], (4,), (1,))
        Bs[1] = MPO._contract(Bs[1], Ybs[1], (4,), (1,))
        Cs[1] = MPO._contract(Cs[1], Ycs[1], (4,), (1,))
        As[2] = MPO._contract(As[2], Yas[2], (4,), (1,))
        Bs[2] = MPO._contract(Bs[2], Ybs[2], (4,), (1,))
        Cs[2] = MPO._contract(Cs[2], Ycs[2], (4,), (1,))

        L = ones(ComplexF64, 1, 1, 1)
        L1 = MPO._contract(L, As[1], (1,), (1,))
        L1 = MPO._contract(L1, Bs[1], (1,4,), (1,2,))
        L1 = MPO._contract(L1, conj(Cs[1]), (1,2,4,), (1,2,3,))
        L2 = MPO._contract(L1, As[2], (1,), (1,))
        L2 = MPO._contract(L2, Bs[2], (1,4,), (1,2,))
        L2 = MPO._contract(L2, conj(Cs[2]), (1,2,4,), (1,2,3,))

        MPO.leftenvironment!(Ls, A, B, C, 1)
        MPO.leftenvironment!(Ls, A, B, C, 2)

        @test Ls[1] ≈ L1
        @test Ls[2] ≈ L2
    end

    @testset "updatecore! site" for method in [:SVD, :LU], maxbonddim in [typemax(Int), 10], direction in [:forward, :backward]
        Random.seed!(1)
        tolerance = 0.
        
        As = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 12), randn(ComplexF64, 12, 3, 4, 1)]
        Bs = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 13), randn(ComplexF64, 13, 4, 5, 1)]
        Cs = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 14), randn(ComplexF64, 14, 3, 5, 1)]

        A = MPO.SiteTensorTrain{ComplexF64, 4}(As)
        B = MPO.SiteTensorTrain{ComplexF64, 4}(Bs)
        C = MPO.SiteTensorTrain{ComplexF64, 4}(Cs)

        As = TCI.sitetensors(A)
        Bs = TCI.sitetensors(B)
        Cs = TCI.sitetensors(C)

        Rs = Vector{Array{ComplexF64,3}}(undef, 4)
        Ls = Vector{Array{ComplexF64,3}}(undef, 4)
        MPO.rightenvironment!(Rs, A, B, C, 4)
        MPO.leftenvironment!(Ls, A, B, C, 1)

        Le = MPO._contract(MPO._contract(Ls[1], As[2], (1,), (1,)), Bs[2], (1, 4), (1, 2))
        Re = MPO._contract(Bs[3], MPO._contract(As[3], Rs[4], (4,), (1,)), (2, 4), (3, 4))
        Ce = MPO._contract(Le, Re, (3, 5), (3, 1))
        Ce = permutedims(Ce, (1, 2, 3, 5, 4, 6))

        left, right, newbonddim, disc = MPO._factorize(
        reshape(Ce, prod(size(Ce)[1:3]), prod(size(Ce)[4:6])),
        method; tolerance, maxbonddim, leftorthogonal=(direction == :forward ? true : false)
        )
        
        C_2 = reshape(left, size(C[2])[1:3]..., newbonddim)
        C_3 = reshape(right, newbonddim, size(C[3])[2:4]...)

        disc = updatecore!(A, B, C, 2, Ls, Rs; method, maxbonddim, tolerance, direction)

        @test C_2 ≈ TCI.sitetensor(C,2)
        @test C_3 ≈ TCI.sitetensor(C,3)
        if maxbonddim == typemax(Int)
            @test disc == 0.
        else
            @test disc > 0.
        end
    end

    @testset "updatecore! inverse" for method in [:SVD], maxbonddim in [typemax(Int), 10], direction in [:forward, :backward]
        Random.seed!(1)
        tolerance = 0.

        As = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)]
        Bs = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        Cs = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]

        A = MPO.InverseTensorTrain{ComplexF64, 4}(TCI.TensorTrain{ComplexF64,4}(As))
        B = MPO.InverseTensorTrain{ComplexF64, 4}(TCI.TensorTrain{ComplexF64,4}(Bs))
        C = MPO.InverseTensorTrain{ComplexF64, 4}(TCI.TensorTrain{ComplexF64,4}(Cs))

        As = TCI.sitetensors(A)
        Bs = TCI.sitetensors(B)
        Cs = TCI.sitetensors(C)
        Yas = MPO.inversesingularvalues(A)
        Ybs = MPO.inversesingularvalues(B)
        Ycs = MPO.inversesingularvalues(C)

        Rs = Vector{Array{ComplexF64,3}}(undef, 4)
        Ls = Vector{Array{ComplexF64,3}}(undef, 4)
        MPO.rightenvironment!(Rs, A, B, C, 4)
        MPO.leftenvironment!(Ls, A, B, C, 1)

        As[2] = MPO._contract(As[2], Yas[2], (4,), (1,))
        Bs[2] = MPO._contract(Bs[2], Ybs[2], (4,), (1,))

        Le = MPO._contract(MPO._contract(Ls[1], As[2], (1,), (1,)), Bs[2], (1, 4), (1, 2))
        Re = MPO._contract(Bs[3], MPO._contract(As[3], Rs[4], (4,), (1,)), (2, 4), (3, 4))

        Ce = MPO._contract(Le, Re, (3, 5), (3, 1))
        Ce = permutedims(Ce, (1, 2, 3, 5, 4, 6))

        left, diamond, right, newbonddim, disc = MPO._factorize(
            reshape(Ce, prod(size(Ce)[1:3]), prod(size(Ce)[4:6])),
            method; tolerance, maxbonddim, diamond=true
        )
        
        C_2 = reshape(left*Diagonal(diamond), size(C[2])[1:3]..., newbonddim)
        C_3 = reshape(Diagonal(diamond)*right, newbonddim, size(C[3])[2:4]...)
        VC_2 = Diagonal(diamond.^-1)

        disc = MPO.updatecore!(A, B, C, 2, Ls, Rs; method, maxbonddim, tolerance, direction)

        @test C_2 ≈ TCI.sitetensor(C,2)
        @test C_3 ≈ TCI.sitetensor(C,3)
        @test VC_2 ≈ MPO.inversesingularvalue(C,2)
        if maxbonddim == typemax(Int)
            @test disc == 0.
        else
            @test disc > 0.
        end
    end

    @testset "MPO-MPO contraction zipup" for method in [:SVD, :LU, :RSVD], maxbonddim in [typemax(Int), 10]
        Random.seed!(42) # For reproducibility
        N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()

        ab = MPO.contract(a, b; algorithm=:zipup, method, maxbonddim, tolerance=0.0)

        @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
    end

    @testset "MPO-MPO contraction fit inverse" for method in [:SVD, :LU, :RSVD], maxbonddim in [typemax(Int), 10], nsweeps in [9,10]
        Random.seed!(42) # For reproducibility
        N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()

        a = MPO.InverseTensorTrain{ComplexF64, 4}(a)
        b = MPO.InverseTensorTrain{ComplexF64, 4}(b)

        ab = MPO.contract(a, b; algorithm=:fit, method, maxbonddim, nsweeps, tolerance=0.0)

        @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
    end

    @testset "MPO-MPO contraction fit site" for method in [:SVD, :LU, :RSVD], maxbonddim in [typemax(Int), 10], nsweeps in [9,10]
        Random.seed!(42) # For reproducibility
        N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()

        a = MPO.SiteTensorTrain{ComplexF64, 4}(a)
        b = MPO.SiteTensorTrain{ComplexF64, 4}(b)

        ab = MPO.contract(a, b; algorithm=:fit, method, maxbonddim, nsweeps, tolerance=0.0)

        @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
    end


    @testset "MPO-MPS contraction zipup" for method in [:SVD, :LU, :RSVD], maxbonddim in [typemax(Int), 10]
        Random.seed!(42) # For reproducibility
        N, a, b, localdims1, localdims2 = _gen_testdata_TTO_TTS()

        ab = MPO.contract(a, b; algorithm=:zipup, method, maxbonddim, tolerance=0.0)
        @test _tovec(ab) ≈ _tomat(a) * _tovec(b)
    end

    @testset "MPO-MPS contraction fit inverse" for method in [:SVD, :LU, :RSVD], maxbonddim in [typemax(Int), 10], nsweeps in [9,10]
        Random.seed!(42) # For reproducibility
        N, a, b, localdims1, localdims2 = _gen_testdata_TTO_TTS()

        a = MPO.InverseTensorTrain{ComplexF64, 4}(a)
        b = MPO.InverseTensorTrain{ComplexF64, 3}(b)

        ab = MPO.contract(a, b; algorithm=:fit, method, maxbonddim, nsweeps, tolerance=0.0)

        @test _tovec(ab) ≈ _tomat(a) * _tovec(b)
    end

    @testset "MPO-MPS contraction fit site" for method in [:SVD, :LU, :RSVD], maxbonddim in [typemax(Int), 10], nsweeps in [9,10]
        Random.seed!(42) # For reproducibility
        N, a, b, localdims1, localdims2 = _gen_testdata_TTO_TTS()

        a = MPO.SiteTensorTrain{ComplexF64, 4}(a)
        b = MPO.SiteTensorTrain{ComplexF64, 3}(b)

        ab = MPO.contract(a, b; algorithm=:fit, method, maxbonddim, nsweeps, tolerance=0.0)

        @test _tovec(ab) ≈ _tomat(a) * _tovec(b)
    end
end