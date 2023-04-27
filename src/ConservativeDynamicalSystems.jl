module ConservativeDynamicalSystems

export AbstractDynamicalSystem, 
    SimplePendulum,
    DoublePendulum,
    DoublePendulumHamiltonian,
    QuadrupoleBosonHamiltonian,
    HenonHeilesSystem,
    KeplerProblem,
    ThreeBodyProblem,
    Robertson,
    rhs,
    rhs!,
    # invariants,
    # invariants!,
    # invariants_jacobian,
    # invariants_jacobian!,
    get_default_initial_conditions

abstract type AbstractDynamicalSystem{T<:Real} end

include("constants.jl")
include("vector_products.jl")
include("simple_pendulum.jl")
include("double_pendulum.jl")
include("qbh.jl")
include("henon_heiles.jl")
include("kepler_problem.jl")
include("three_body_problem.jl")
include("robertson.jl")

end
