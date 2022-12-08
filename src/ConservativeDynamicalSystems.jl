module ConservativeDynamicalSystems

export SimplePendulum, DoublePendulum, QuadrupoleBosonHamiltonian, rhs, rhs!, hamiltonian, jacobian

abstract type AbstractDynamicalSystem{T<:AbstractFloat} end

include("constants.jl")
include("simple_pendulum.jl")
include("double_pendulum.jl")
include("qbh.jl")

end
