module ConservativeDynamicalSystems

export SimplePendulum,
    DoublePendulum,
    QuadrupoleBosonHamiltonian,
    HenonHeiles,
    ThreeBodyProblem,
    rhs,
    jacobian_rhs,
    rhs!,
    hamiltonian,
    jacobian_hamiltonian,
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
