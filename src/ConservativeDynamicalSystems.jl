module ConservativeDynamicalSystems

abstract type AbstractDynamicalSystem{T<:AbstractFloat} end

include("constants.jl")
include("simple_pendulum.jl")
include("double_pendulum.jl")
include("qbh.jl")

end
