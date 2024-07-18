using DictionaryDistance
using Test
using Aqua
using JET
using SparseArrays
import Random: shuffle

@testset "DictionaryDistance.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        # we do pirate (on purpose) the `_roc` method.
        # also there's some inherited method ambiguities from `StatsBase`, but they're not from us.
        Aqua.test_all(DictionaryDistance; ambiguities=false, piracies=false)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(DictionaryDistance; target_defined_modules = true)
    end

    D = rand(100, 400);
    X = sprand(400, 10_000, 0.01)

    perm = shuffle(axes(D, 2))
    D_perm = D[:, perm]
    (; assignment) = align_dictionaries(D, D_perm)
    @testset "test dictionary alignment" begin
        @test sortperm(perm) == assignment
    end

    X_perm = X[perm, :];
    X_recover = X_perm[assignment, :];
    r = roc(nonzerovec(X[:]), nonzerovec(X_recover[:]))
    @testset "test metrics" begin
        @test precision(r) ≈ 1
        @test recall(r) ≈ 1
        @test f1score(r) ≈ 1
    end
end
