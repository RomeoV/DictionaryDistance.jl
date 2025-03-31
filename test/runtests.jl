using DictionaryDistance
using Distances
using Test
using Aqua
using JET
using SparseArrays
import Random: shuffle

@testset "DictionaryDistance.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        if !haskey(ENV, "JULIA_SKIP_AQUA")
            # we do pirate (on purpose) the `_roc` method.
            # also there's some inherited method ambiguities from `StatsBase`, but they're not from us.
            Aqua.test_all(DictionaryDistance; ambiguities = false, piracies = false)
        else
            @info "Skipping Aqua.jl tests"
        end
    end

    @testset "Code linting (JET.jl)" begin
        if !haskey(ENV, "JULIA_SKIP_JET")
            JET.test_package(DictionaryDistance; target_defined_modules = true)
        else
            @info "Skipping JET.jl tests"
        end
    end

    D = rand(100, 400)
    X = sprand(400, 10_000, 0.01)

    perm = shuffle(axes(D, 2))
    D_perm = D[:, perm]
    (; assignment, cost) = align_dictionaries(D, D_perm)
    @testset "test dictionary alignment" begin
        @test sortperm(perm) == assignment
        @test cost < sqrt(eps(eltype(D)))
    end

    perm = shuffle(axes(X, 1))
    X_perm = X[perm, :]
    (; assignment, cost) = align_dictionaries(X', X_perm', SqEuclidean())
    @testset "test dictionary alignment" begin
        @test sortperm(perm) == assignment
        @test cost < sqrt(eps(eltype(X)))
    end

    X_recover = X_perm[assignment, :]
    r = roc(nonzerovec(X[:]), nonzerovec(X_recover[:]))
    @testset "test metrics" begin
        @test precision(r) ≈ 1
        @test recall(r) ≈ 1
        @test f1score(r) ≈ 1
    end
end
