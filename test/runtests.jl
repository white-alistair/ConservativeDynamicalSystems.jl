using ConservativeDynamicalSystems, Random, OrdinaryDiffEq, Test

@testset "ConservativeDynamicalSystems.jl" begin
    include("vector_products.jl")
    include("double_pendulum.jl")
    include("three_body_problem.jl")
end
