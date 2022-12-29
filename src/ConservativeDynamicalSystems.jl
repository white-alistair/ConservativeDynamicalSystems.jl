module ConservativeDynamicalSystems

export SimplePendulum,
    DoublePendulum,
    QuadrupoleBosonHamiltonian,
    HenonHeilesSystem,
    ThreeBodyProblem,
    rhs,
    rhs_jacobian,
    rhs!,
    constraints,
    constraints_hamiltonian,
    constraints!

using ForwardDiff

abstract type AbstractDynamicalSystem{T<:Real} end

include("constants.jl")
include("vector_products.jl")
include("simple_pendulum.jl")
include("double_pendulum.jl")
include("qbh.jl")
include("henon_heiles.jl")
include("three_body_problem.jl")

end
