struct SimplePendulum{T} <: AbstractDynamicalSystem{T}
    m::T
    l::T
end

SimplePendulum{T}() where {T} = SimplePendulum{T}(1.0, 1.0)

function get_default_initial_conditions(::SimplePendulum{T}) where {T}
    return T[π / 4, 0.0]
end

function rhs(u::AbstractVector{T}, system::SimplePendulum, t) where {T}
    θ, ω = u
    (; m, l) = system
    return [ω, -g(T) / l * sin(θ)]
end

function rhs_jacobian(u::AbstractVector{T}, system::SimplePendulum{T}, t) where {T}
    return ForwardDiff.jacobian(u -> rhs(u, system, nothing), u)
end

function rhs!(du::AbstractVector{T}, u::AbstractVector{T}, system::SimplePendulum{T}, t) where {T}
    θ, ω = u
    (; m, l) = system

    du[1] = ω
    du[2] = -g(T) / l * sin(θ)

    return nothing
end
