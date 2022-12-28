module ConservativeDynamicalSystems

export SimplePendulum,
    DoublePendulum,
    QuadrupoleBosonHamiltonian,
    HenonHeiles,
    rhs,
    jacobian_rhs,
    rhs!,
    hamiltonian,
    jacobian_hamiltonian

using ForwardDiff

abstract type AbstractDynamicalSystem{T<:Real} end

include("constants.jl")
include("simple_pendulum.jl")
include("double_pendulum.jl")
include("qbh.jl")
include("henon_heiles.jl")

end
