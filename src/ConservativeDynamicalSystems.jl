module ConservativeDynamicalSystems

export AbstractDynamicalSystem, 
    SimplePendulum,
    DoublePendulum,
    QuadrupoleBosonHamiltonian,
    HenonHeilesSystem,
    ThreeBodyProblem,
    rhs,
    rhs!,
    invariants,
    invariants!,
    invariants_jacobian,
    invariants_jacobian!

abstract type AbstractDynamicalSystem{T<:Real} end

include("constants.jl")
include("vector_products.jl")
include("simple_pendulum.jl")
include("double_pendulum.jl")
include("qbh.jl")
include("henon_heiles.jl")
include("three_body_problem.jl")

end
