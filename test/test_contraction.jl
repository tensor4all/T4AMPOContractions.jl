import TensorCrossInterpolation as TCI
using TensorCrossInterpolation
using Random
# TODO remove
using Test
using LinearAlgebra

function _tomat(tto::TensorTrain{T,4}) where {T}
    sitedims = TCI.sitedims(tto)
    localdims1 = [s[1] for s in sitedims]
    localdims2 = [s[2] for s in sitedims]
    mat = Matrix{T}(undef, prod(localdims1), prod(localdims2))
    for (i, inds1) in enumerate(CartesianIndices(Tuple(localdims1)))
        for (j, inds2) in enumerate(CartesianIndices(Tuple(localdims2)))
            mat[i, j] = evaluate(tto, collect(zip(Tuple(inds1), Tuple(inds2))))
        end
    end
    return mat
end

function _tovec(tt::TensorTrain{T,3}) where {T}
    sitedims = TCI.sitedims(tt)
    localdims1 = [s[1] for s in sitedims]
    return evaluate.(Ref(tt), CartesianIndices(Tuple(localdims1))[:])
end

function _gen_testdata_TTO_TTO()
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [5, 5, 5, 5]

    a = TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TensorTrain{ComplexF64,4}([
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

    a = TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TensorTrain{ComplexF64,3}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], bonddims_b[n+1])
        for n = 1:N
    ])
    return N, a, b, localdims1, localdims2
end


@testset "contractions" begin
    @testset "_contract" begin
        a = rand(2, 3, 4)
        b = rand(2, 5, 4)
        ab = TCI._contract(a, b, (1, 3), (1, 3))
        @test vec(reshape(permutedims(a, (2, 1, 3)), 3, :) * reshape(permutedims(b, (1, 3, 2)), :, 5)) ≈ vec(ab)
    end

    @testset "rightenvironment" begin
        a = rand(2, 3, 4)
        b = rand(2, 5, 4)
        ab = TCI._contract(a, b, (1, 3), (1, 3))
        @test vec(reshape(permutedims(a, (2, 1, 3)), 3, :) * reshape(permutedims(b, (1, 3, 2)), :, 5)) ≈ vec(ab)
    end

    @testset "MPO-MPO contraction (TCI, naive)" for f in [nothing, x -> 2 * x], algorithm in [:TCI, :naive]
        #==
        N = 4
        bonddims_a = [1, 2, 3, 2, 1]
        bonddims_b = [1, 2, 3, 2, 1]
        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]
        localdims3 = [2, 2, 2, 2]

        a = TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
            for n = 1:N
        ])
        b = TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1])
            for n = 1:N
        ])
        ==#
        N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()

        if f !== nothing && algorithm === :naive
            @test_throws ErrorException contract(a, b; f=f, algorithm=algorithm)
        else
            ab = contract(a, b; f=f, algorithm=algorithm)
            @test sitedims(ab) == [[localdims1[i], localdims3[i]] for i = 1:N]
            if f === nothing
                @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
            else
                @test _tomat(ab) ≈ f.(_tomat(a) * _tomat(b))
            end
        end
    end

    @testset "Contraction, batchevaluate" begin
        import TensorCrossInterpolation: TensorTrain

        N = 4
        bonddims_a = [1, 2, 3, 2, 1]
        bonddims_b = [1, 2, 3, 2, 1]
        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]
        localdims3 = [2, 2, 2, 2]

        a = TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
            for n = 1:N
        ])
        b = TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1])
            for n = 1:N
        ])

        ab = TCI.Contraction(a, b)
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

        #res = TCI.batchevaluate(ab, leftindexset, rightindexset, Val(2), [[1], [1]])
        #@test vec(ref[:, 1, 1, :]) ≈ vec(res)


    end

    @testset "MPO-MPS contraction (TCI, naive)" for f in [nothing, x -> 2 * x], algorithm in [:TCI, :naive]
        #==
        N = 4
        bonddims_a = [1, 2, 3, 2, 1]
        bonddims_b = [1, 2, 3, 2, 1]
        localdims1 = [3, 3, 3, 3]
        localdims2 = [3, 3, 3, 3]

        a = TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
            for n = 1:N
        ])
        b = TensorTrain{ComplexF64,3}([
            rand(ComplexF64, bonddims_b[n], localdims2[n], bonddims_b[n+1])
            for n = 1:N
        ])
        ==#
        N, a, b, localdims1, localdims2 = _gen_testdata_TTO_TTS()

        if f !== nothing && algorithm === :naive
            @test_throws ErrorException contract(a, b; f=f, algorithm=algorithm)
            @test_throws ErrorException contract(b, a; f=f, algorithm=algorithm)
        else
            ab = contract(a, b; f=f, algorithm=algorithm)
            ba = contract(b, a; f=f, algorithm=algorithm)
            @test sitedims(ab) == [[localdims1[i]] for i = 1:N]
            if f === nothing
                @test _tovec(ab) ≈ _tomat(a) * _tovec(b)
                @test transpose(_tovec(ba)) ≈ transpose(_tovec(b)) * _tomat(a)
            else
                @test _tovec(ab) ≈ f.(_tomat(a) * _tovec(b))
                @test transpose(_tovec(ba)) ≈ f.(transpose(_tovec(b)) * _tomat(a))
            end
        end
    end

    @testset "rightenvironment" begin
        Random.seed!(42)

        R = ones(ComplexF64, 1, 1, 1)
        A = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)]
        B = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        C = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]

        R3 = TCI._contract(conj(C[3]), R, (4,), (3,))
        R3 = TCI._contract(B[3], R3, (3,4,), (3,5,))
        R3 = TCI._contract(A[3], R3, (2,3,4), (4,2,5,))

        R2 = TCI._contract(conj(C[2]), R3, (4,), (3,))
        R2 = TCI._contract(B[2], R2, (3,4,), (3,5,))
        R2 = TCI._contract(A[2], R2, (2,3,4), (4,2,5,))

        right3 = TCI.rightenvironment(A, B, C, 3)
        right2 = TCI.rightenvironment(A, B, C, 2; R=right3, Ri=2)
        # TODO try removing the Ri from the code

        @test right3 ≈ R3
        @test right2 ≈ R2
    end

    @testset "leftenvironment" begin
        Random.seed!(42)

        L = ones(ComplexF64, 1, 1, 1)
        A = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)]
        B = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        C = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]

        L1 = TCI._contract(L, A[1], (1,), (1,))
        L1 = TCI._contract(L1, B[1], (1,4,), (1,2,))
        L1 = TCI._contract(L1, conj(C[1]), (1,2,4,), (1,2,3,))

        L2 = TCI._contract(L1, A[2], (1,), (1,))
        L2 = TCI._contract(L2, B[2], (1,4,), (1,2,))
        L2 = TCI._contract(L2, conj(C[2]), (1,2,4,), (1,2,3,))


        left1 = TCI.leftenvironment(A, B, C, 1)
        left2 = TCI.leftenvironment(A, B, C, 2; L=left1, Li=2)
        # TODO try removing the Li from the code

        @test left1 ≈ L1
        @test left2 ≈ L2
    end

    @testset "rightenvironment inverse" begin
        Random.seed!(42)

        R = ones(ComplexF64, 1, 1, 1)
        A = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)]
        VA = [randn(ComplexF64, 6, 6), randn(ComplexF64, 9, 9)]
        B = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        VB = [randn(ComplexF64, 7, 7), randn(ComplexF64, 10, 10)]
        C = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]
        VC = [randn(ComplexF64, 8, 8), randn(ComplexF64, 11, 11)]

        right3 = TCI.rightenvironment(A, VA, B, VB, C, VC, 3)
        right2 = TCI.rightenvironment(A, VA, B, VB, C, VC, 2; R=right3, Ri=2)
        # TODO try removing the Ri from the code

        A[3] = TCI._contract(VA[2], A[3], (2,), (1,))
        B[3] = TCI._contract(VB[2], B[3], (2,), (1,))
        C[3] = TCI._contract(VC[2], C[3], (2,), (1,))

        A[2] = TCI._contract(VA[1], A[2], (2,), (1,))
        B[2] = TCI._contract(VB[1], B[2], (2,), (1,))
        C[2] = TCI._contract(VC[1], C[2], (2,), (1,))

        R3 = TCI._contract(conj(C[3]), R, (4,), (3,))
        R3 = TCI._contract(B[3], R3, (3,4,), (3,5,))
        R3 = TCI._contract(A[3], R3, (2,3,4), (4,2,5,))

        R2 = TCI._contract(conj(C[2]), R3, (4,), (3,))
        R2 = TCI._contract(B[2], R2, (3,4,), (3,5,))
        R2 = TCI._contract(A[2], R2, (2,3,4), (4,2,5,))

        @test right3 ≈ R3
        @test right2 ≈ R2
    end

    @testset "leftenvironment inverse" begin
        Random.seed!(42)

        L = ones(ComplexF64, 1, 1, 1)
        A = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)]
        VA = [randn(ComplexF64, 6, 6), randn(ComplexF64, 9, 9)]
        B = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        VB = [randn(ComplexF64, 7, 7), randn(ComplexF64, 10, 10)]
        C = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]
        VC = [randn(ComplexF64, 8, 8), randn(ComplexF64, 11, 11)]
        
        left1 = TCI.leftenvironment(A, VA, B, VB, C, VC, 1)
        left2 = TCI.leftenvironment(A, VA, B, VB, C, VC, 2; L=left1, Li=2)
        # TODO try removing the Li from the code

        A[1] = TCI._contract(A[1], VA[1], (4,), (1,))
        B[1] = TCI._contract(B[1], VB[1], (4,), (1,))
        C[1] = TCI._contract(C[1], VC[1], (4,), (1,))

        A[2] = TCI._contract(A[2], VA[2], (4,), (1,))
        B[2] = TCI._contract(B[2], VB[2], (4,), (1,))
        C[2] = TCI._contract(C[2], VC[2], (4,), (1,))

        L1 = TCI._contract(L, A[1], (1,), (1,))
        L1 = TCI._contract(L1, B[1], (1,4,), (1,2,))
        L1 = TCI._contract(L1, conj(C[1]), (1,2,4,), (1,2,3,))

        L2 = TCI._contract(L1, A[2], (1,), (1,))
        L2 = TCI._contract(L2, B[2], (1,4,), (1,2,))
        L2 = TCI._contract(L2, conj(C[2]), (1,2,4,), (1,2,3,))

        @test left1 ≈ L1
        @test left2 ≈ L2
    end

    @testset "updatecore!" for method in [:SVD, :LU], maxbonddim in [typemax(Int), 10], direction in [:forward, :backward]
        Random.seed!(1)
        tolerance = 0.
        
        A = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 12), randn(ComplexF64, 12, 3, 4, 1)]
        B = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 13), randn(ComplexF64, 13, 4, 5, 1)]
        C = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 14), randn(ComplexF64, 14, 3, 5, 1)]

        R = TCI.rightenvironment(A, B, C, 4)
        L = TCI.leftenvironment(A, B, C, 1)

        Le = TCI._contract(TCI._contract(L, A[2], (1,), (1,)), B[2], (1, 4), (1, 2))
        Re = TCI._contract(B[3], TCI._contract(A[3], R, (4,), (1,)), (2, 4), (3, 4))

        Ce = TCI._contract(Le, Re, (3, 5), (3, 1))
        Ce = permutedims(Ce, (1, 2, 3, 5, 4, 6))

        left, right, newbonddim, disc = TCI._factorize(
        reshape(Ce, prod(size(Ce)[1:3]), prod(size(Ce)[4:6])),
        method; tolerance, maxbonddim, leftorthogonal=(direction == :forward ? true : false)
        )
        
        C_2 = reshape(left, :, size(C[2])[2:3]..., newbonddim)
        C_3 = reshape(right, newbonddim, size(C[3])[2:3]..., :)

        disc, left, right = TCI.updatecore!(A, B, C, 2; method, maxbonddim, tolerance, direction)

        @test right ≈ R
        @test left ≈ L
        @test C_2 ≈ C[2]
        @test C_3 ≈ C[3]
        if maxbonddim == typemax(Int)
            @test disc == 0.
        else
            @test disc > 0.
        end
    end

    @testset "updatecore! inverse" for method in [:SVD], maxbonddim in [typemax(Int), 10], direction in [:forward, :backward]
        Random.seed!(1)
        tolerance = 0.

        A = [randn(ComplexF64, 1, 3, 4, 6), randn(ComplexF64, 6, 3, 4, 9), randn(ComplexF64, 9, 3, 4, 1)]
        VA = [randn(ComplexF64, 6, 6), randn(ComplexF64, 9, 9)]
        B = [randn(ComplexF64, 1, 4, 5, 7), randn(ComplexF64, 7, 4, 5, 10), randn(ComplexF64, 10, 4, 5, 1)]
        VB = [randn(ComplexF64, 7, 7), randn(ComplexF64, 10, 10)]
        C = [randn(ComplexF64, 1, 3, 5, 8), randn(ComplexF64, 8, 3, 5, 11), randn(ComplexF64, 11, 3, 5, 1)]
        VC = [randn(ComplexF64, 8, 8), randn(ComplexF64, 11, 11)]

        R = TCI.rightenvironment(A, VA, B, VB, C, VC, 4)
        L = TCI.leftenvironment(A, VA, B, VB, C, VC, 1)

        Ai = TCI._contract(A[2], VA[2], (4,), (1,))
        Bi = TCI._contract(B[2], VB[2], (4,), (1,))

        Le = TCI._contract(TCI._contract(L, Ai, (1,), (1,)), Bi, (1, 4), (1, 2))
        Re = TCI._contract(B[3], TCI._contract(A[3], R, (4,), (1,)), (2, 4), (3, 4))

        Ce = TCI._contract(Le, Re, (3, 5), (3, 1))
        Ce = permutedims(Ce, (1, 2, 3, 5, 4, 6))

        left, diamond, right, newbonddim, disc = TCI._factorize(
            reshape(Ce, prod(size(Ce)[1:3]), prod(size(Ce)[4:6])),
            method; tolerance, maxbonddim, diamond=true
        )
        
        C_2 = reshape(left*Diagonal(diamond), :, size(C[2])[2:3]..., newbonddim)
        C_3 = reshape(Diagonal(diamond)*right, newbonddim, size(C[3])[2:3]..., :)
        VC_2 = Diagonal(diamond.^-1)

        disc, left, right = TCI.updatecore!(A, VA, B, VB, C, VC, 2; method, maxbonddim, tolerance, direction)

        @test right ≈ R
        @test left ≈ L
        @test C_2 ≈ C[2]
        @test C_3 ≈ C[3]
        if maxbonddim == typemax(Int)
            @test disc == 0.
        else
            @test disc > 0.
        end
    end

    @testset "MPO-MPO contraction (zipup, fit)" for algorithm in [:zipup, :fit], method in [:SVD, :LU, :RSVD], maxbonddim in [typemax(Int), 10], nsweeps in [9,10]
        Random.seed!(42) # For reproducibility
        N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()
        #println([size(a[i]) for i in 1:length(a)])
        #println([size(b[i]) for i in 1:length(b)])
        #println("$maxbonddim with $method")
        ab = contract(a, b; algorithm, method, maxbonddim, nsweeps, tolerance=0.0)
        #println([size(a[i]) for i in 1:length(a)])
        #println([size(b[i]) for i in 1:length(b)])
        #println([size(ab[i]) for i in 1:length(ab)])

        # println("$algorithm, $method, $maxbonddim, $nsweeps")
        @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
    end

    @testset "MPO-MPO contraction inverse" for method in [:SVD, :LU, :RSVD], maxbonddim in [typemax(Int), 10], nsweeps in [9,10]
        Random.seed!(42) # For reproducibility
        N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()
        #println([size(a[i]) for i in 1:length(a)])
        #println([size(b[i]) for i in 1:length(b)])
        #println("$maxbonddim with $method")
        # println(algorithm, method, maxbonddim)

        Gamma_a, Lambda_a = TCI.extract_vidal(a.sitetensors)
        Gamma_b, Lambda_b = TCI.extract_vidal(b.sitetensors)
        
        Psia, Va = TCI.vidal_to_inv(Gamma_a, Lambda_a)
        Psib, Vb = TCI.vidal_to_inv(Gamma_b, Lambda_b)

        ab = contract(a, b; algorithm=:fit, method, maxbonddim, nsweeps, tolerance=0.0)
        #println([size(a[i]) for i in 1:length(a)])
        #println([size(b[i]) for i in 1:length(b)])
        #println([size(ab[i]) for i in 1:length(ab)])

        # println("fit, $method, $maxbonddim, $nsweeps")
        @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
    end


    @testset "MPO-MPS contraction (zipup, fit)" for algorithm in [:zipup, :fit], method in [:SVD, :LU, :RSVD], maxbonddim in [typemax(Int), 10], nsweeps in [9,10]
        Random.seed!(42) # For reproducibility
        N, a, b, localdims1, localdims2 = _gen_testdata_TTO_TTS()

        ab = contract(a, b; algorithm, method, maxbonddim, nsweeps, tolerance=0.0)
        # println("$algorithm, $method, $maxbonddim, $nsweeps")
        @test _tovec(ab) ≈ _tomat(a) * _tovec(b)
    end
end