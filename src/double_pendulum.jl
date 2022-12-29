struct DoublePendulum{T} <: AbstractDynamicalSystem{T}
    m₁::T
    m₂::T
    l₁::T
    l₂::T
end

DoublePendulum{T}() where {T} = DoublePendulum{T}(1.0, 1.0, 1.0, 1.0)

function get_default_initial_conditions(::DoublePendulum{T}) where {T}
    return T[π/2, π/2, 0.0, 0.0]
end

function get_h₁(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return p₁ * p₂ * sin(θ₁ - θ₂) / (l₁ * l₂ * (m₁ + m₂ * sin(θ₁ - θ₂)^2))
end

function get_h₂(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return (
        m₂ * l₂^2 * p₁^2 + (m₁ + m₂) * l₁^2 * p₂^2 -
        2 * m₂ * l₁ * l₂ * p₁ * p₂ * cos(θ₁ - θ₂)
    ) / (2 * l₁^2 * l₂^2 * (m₁ + m₂ * sin(θ₁ - θ₂)^2)^2)
end

function dθ₁(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return (l₂ * p₁ - l₁ * p₂ * cos(θ₁ - θ₂)) / (l₁^2 * l₂ * (m₁ + m₂ * sin(θ₁ - θ₂)^2))
end

function dθ₂(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return (-m₂ * l₂ * p₁ * cos(θ₁ - θ₂) + (m₁ + m₂) * l₁ * p₂) /
           (m₂ * l₁ * l₂^2 * (m₁ + m₂ * sin(θ₁ - θ₂)^2))
end

function dp₁(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁) = system
    θ₁, θ₂ = u[1:2]
    h₁ = get_h₁(system, u)
    h₂ = get_h₂(system, u)
    return -(m₁ + m₂) * g(T) * l₁ * sin(θ₁) - h₁ + h₂ * sin(2 * (θ₁ - θ₂))
end

function dp₂(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₂, l₂) = system
    θ₁, θ₂, = u[1:2]
    h₁ = get_h₁(system, u)
    h₂ = get_h₂(system, u)
    return -m₂ * g(T) * l₂ * sin(θ₂) + h₁ - h₂ * sin(2 * (θ₁ - θ₂))
end

function rhs(u::AbstractVector{T}, system::DoublePendulum, t) where {T}
    return [dθ₁(system, u), dθ₂(system, u), dp₁(system, u), dp₂(system, u)]
end

function rhs_jacobian(u::AbstractVector{T}, system::DoublePendulum{T}, t) where {T}
    return ForwardDiff.jacobian(u -> rhs(u, system, t), u)
end

function rhs!(
    du::AbstractVector{T},
    u::AbstractVector{T},
    system::DoublePendulum{T},
    t,
) where {T}
    du[1] = dθ₁(system, u)
    du[2] = dθ₂(system, u)
    du[3] = dp₁(system, u)
    du[4] = dp₂(system, u)
    return nothing
end

function hamiltonian(u::AbstractVector{T}, system::DoublePendulum{T}, t::T) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return
    (
        m₂ * l₂^2 * p₁^2 + (m₁ + m₂) * l₁^2 * p₂^2 -
        2 * m₂ * l₁ * l₂ * p₁ * p₂ * cos(θ₁ - θ₂)
    ) / (2 * m₂ * l₁^2 * l₂^2 * (m₁ + m₂ * sin(θ₁ - θ₂)^2)) -
    (m₁ + m₂) * g(T) * l₁ * cos(θ₁) - m₂ * g(T) * l₂ * cos(θ₂)
end

function constraints(u::AbstractVector{T}, system::DoublePendulum{T}, t::T) where {T}
    return [hamiltonian(u, system, t)]
end

function constraints_jacobian(
    u::AbstractVector{T},
    system::DoublePendulum{T},
    t::T,
) where {T}
    return [-dp₁(system, u) -dp₂(system, u) dθ₁(system, u) dθ₂(system, u)]
end
