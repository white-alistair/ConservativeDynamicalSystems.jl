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

@inline function get_C₁(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return p₁ * p₂ * sin(θ₁ - θ₂) / (l₁ * l₂ * (m₁ + m₂ * sin(θ₁ - θ₂)^2))
end

@inline function get_C₂(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return sin(2 * (θ₁ - θ₂)) * (
        m₂ * l₂^2 * p₁^2 + (m₁ + m₂) * l₁^2 * p₂^2 -
        2 * m₂ * l₁ * l₂ * p₁ * p₂ * cos(θ₁ - θ₂)
    ) / (2 * l₁^2 * l₂^2 * (m₁ + m₂ * sin(θ₁ - θ₂)^2)^2)
end

@inline function dθ₁(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return (l₂ * p₁ - l₁ * p₂ * cos(θ₁ - θ₂)) / (l₁^2 * l₂ * (m₁ + m₂ * sin(θ₁ - θ₂)^2))
end

@inline function dθ₂(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return (-m₂ * l₂ * p₁ * cos(θ₁ - θ₂) + (m₁ + m₂) * l₁ * p₂) /
           (m₂ * l₁ * l₂^2 * (m₁ + m₂ * sin(θ₁ - θ₂)^2))
end

@inline function dp₁(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁) = system
    θ₁ = u[1]
    C₁ = get_C₁(system, u)
    C₂ = get_C₂(system, u)
    return -(m₁ + m₂) * g(T) * l₁ * sin(θ₁) - C₁ + C₂
end

@inline function dp₂(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₂, l₂) = system
    θ₂ = u[2]
    C₁ = get_C₁(system, u)
    C₂ = get_C₂(system, u)
    return -m₂ * g(T) * l₂ * sin(θ₂) + C₁ - C₂
end

function rhs(u::AbstractVector{T}, system::DoublePendulum, t) where {T}
    return [dθ₁(system, u), dθ₂(system, u), dp₁(system, u), dp₂(system, u)]
end

function (system::DoublePendulum)(u, p, t)
    return rhs(u, system, t)
end

function rhs!(
    du::AbstractVector{T},
    u::AbstractVector{T},
    system::DoublePendulum,
    t,
) where {T}
    du[1] = dθ₁(system, u)
    du[2] = dθ₂(system, u)
    du[3] = dp₁(system, u)
    du[4] = dp₂(system, u)
    return nothing
end

function (system::DoublePendulum)(
    du,
    u,
    p,
    t,
)
    rhs!(du, u, system, t)
    return nothing
end

function hamiltonian(u, system::DoublePendulum{T}, t) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, p₁, p₂ = u
    return (
        (
            m₂ * l₂^2 * p₁^2 + (m₁ + m₂) * l₁^2 * p₂^2 -
            2 * m₂ * l₁ * l₂ * p₁ * p₂ * cos(θ₁ - θ₂)
        ) / (2 * m₂ * l₁^2 * l₂^2 * (m₁ + m₂ * sin(θ₁ - θ₂)^2)) -
        (m₁ + m₂) * l₁ * g(T) * cos(θ₁) - m₂ * l₂ * g(T) * cos(θ₂)
    )
end

function invariants(u, system::DoublePendulum, t)
    return [hamiltonian(u, system, t)]
end

function invariants!(invariants, u, system::DoublePendulum, t)
    invariants[1] = hamiltonian(u, system, t)
    return nothing
end

function invariants_jacobian(u, system::DoublePendulum, t)
    return [-dp₁(system, u) -dp₂(system, u) dθ₁(system, u) dθ₂(system, u)]
end

function invariants_jacobian!(J, u, system::DoublePendulum, t)
    J[1, 1] = -dp₁(system, u)
    J[1, 2] = -dp₂(system, u)
    J[1, 3] = dθ₁(system, u)
    J[1, 4] = dθ₂(system, u)
    return nothing
end
