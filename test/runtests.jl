using DictionaryDistance
using Test
using Aqua
using JET

@testset "DictionaryDistance.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DictionaryDistance)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(DictionaryDistance; target_defined_modules = true)
    end
    # Write your tests here.
end
