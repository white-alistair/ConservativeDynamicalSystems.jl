struct SimplePendulum{T} <: AbstractDynamicalSystem{T}
    m::T
    l::T
end

SimplePendulum{T}() where {T} = SimplePendulum{T}(1.0, 1.0)

function get_default_initial_conditions(::SimplePendulum{T}) where {T}
    return T[π/4, 0.0]
end

function rhs(u::AbstractVector{T}, system::SimplePendulum, t) where {T}
    θ, ω = u
    (; l) = system
    return [ω, -1 / l * sin(θ)]
end

function (system::SimplePendulum)(u, p, t)
    return rhs(u, system, t)
end

function rhs_jacobian(u::AbstractVector{T}, system::SimplePendulum{T}, t) where {T}
    return ForwardDiff.jacobian(u -> rhs(u, system, nothing), u)
end

function rhs!(
    du::AbstractVector{T},
    u::AbstractVector{T},
    system::SimplePendulum{T},
    t,
) where {T}
    θ, ω = u
    (; l) = system

    du[1] = ω
    du[2] = -1 / l * sin(θ)

    return nothing
end

function (system::SimplePendulum)(du, u, p, t)
    rhs!(du, u, system, t)
    return nothing
end

function energy(u::AbstractVector, system::SimplePendulum{T}, t) where {T}
    θ, ω = u
    (; m, l) = system
    return T(0.5) * m * l^2 * ω^2 - m * l * cos(θ)
end

function invariants(u, system::SimplePendulum, t)
    return [energy(u, system, t)]
end

function invariants_jacobian(u, system::SimplePendulum, t)
    θ, ω = u
    (; m, l) = system 
    return [m * l * sin(θ);; m * l^2 * ω]
end

function cartesian(system::SimplePendulum, u::AbstractVector)
    (; l) = system
    θ, ω = u
    x = l * sin(θ)
    y = -l * cos(θ)
    dx = l * cos(θ) * ω
    dy = l * sin(θ) * ω
    return [x, y, dx, dy]
end

function cartesian(system::SimplePendulum, trajectory::AbstractMatrix)
    return mapslices(u -> cartesian(system, u), trajectory, dims = 1)
end
