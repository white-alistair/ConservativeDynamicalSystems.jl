using ConservativeDynamicalSystems, Random, OrdinaryDiffEq, Test

@testset "ConservativeDynamicalSystems.jl" begin
    include("three_body_problem.jl")
end
