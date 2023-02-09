using ConservativeDynamicalSystems, Random, OrdinaryDiffEq, Test

@testset "ConservativeDynamicalSystems.jl" begin
    include("vector_products.jl")
    include("double_pendulum.jl")
    include("henon_heiles.jl")
    include("qbh.jl")
    include("kepler_problem.jl")
    include("three_body_problem.jl")
end
