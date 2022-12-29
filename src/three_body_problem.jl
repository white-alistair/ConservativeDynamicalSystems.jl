struct ThreeBodyProblem{T} <: AbstractDynamicalSystem{T}
    m₁::T
    m₂::T
    m₃::T
end

ThreeBodyProblem{T}() where {T} = ThreeBodyProblem{T}(1.0, 1.0, 1.0)

@inline function get_positions(u)
    return @views u[1:3], u[4:6], u[7:9]
end

@inline function get_velocities(u)
    return @views u[10:12], u[13:15], u[16:18]
end

@inline function get_separations(u)
    r₁, r₂, r₃ = get_positions(u)
    return get_separations(r₁, r₂, r₃)
end

@inline function get_separations(r₁, r₂, r₃)
    r₁₂ = sqrt(sum(abs2, r₁ - r₂))
    r₁₃ = sqrt(sum(abs2, r₁ - r₃))
    r₂₃ = sqrt(sum(abs2, r₂ - r₃))
    return r₁₂, r₁₃, r₂₃
end

function rhs!(
    du::AbstractVector{T},
    u::AbstractVector{T},
    system::ThreeBodyProblem{T},
    t,
) where {T}
    (; m₁, m₂, m₃) = system
    r₁, r₂, r₃ = get_positions(u)
    r₁₂, r₁₃, r₂₃ = get_separations(u)

    # Velocities
    du[1:9] .= u[10:18]

    # Accelerations
    du[10:12] .= -m₂ * (r₁ - r₂) / r₁₂^3 - m₃ * (r₁ - r₃) / r₁₃^3
    du[13:15] .= -m₃ * (r₂ - r₃) / r₂₃^3 - m₁ * (r₂ - r₁) / r₁₂^3
    du[16:18] .= -m₁ * (r₃ - r₁) / r₁₃^3 - m₂ * (r₃ - r₂) / r₂₃^3

    return nothing
end

@inline function kinetic_energy(
    system::ThreeBodyProblem{T},
    dr₁::AbstractVector{T},
    dr₂::AbstractVector{T},
    dr₃::AbstractVector{T},
) where {T}
    (; m₁, m₂, m₃) = system
    return 0.5 * (m₁ * dr₁ ⋅ dr₁ + m₂ * dr₂ ⋅ dr₂ + m₃ * dr₃ ⋅ dr₃)
end

@inline function potential_energy(
    system::ThreeBodyProblem{T},
    r₁::AbstractVector{T},
    r₂::AbstractVector{T},
    r₃::AbstractVector{T},
) where {T}
    (; m₁, m₂, m₃) = system
    r₁₂, r₁₃, r₂₃ = get_separations(r₁, r₂, r₃)
    return m₁ * m₂ / r₁₂ + m₁ * m₃ / r₁₃ + m₂ * m₃ / r₂₃
end

function constraints!(
    constraints::AbstractVector{T},
    u::AbstractVector{T},
    system::ThreeBodyProblem{T},
    t::T,
) where {T}
    (; m₁, m₂, m₃) = system
    r₁, r₂, r₃ = get_positions(u)
    dr₁, dr₂, dr₃ = get_velocities(u)

    # Integrals of motion
    # Here we follow the treatment and notation of https://arxiv.org/abs/1508.02312
    C₁ = m₁ * dr₁ + m₂ * dr₂ + m₃ * dr₃
    C₂ = m₁ * r₁ + m₂ * r₂ + m₃ * r₃ - C₁ * t
    C₃ = m₁ * r₁ × dr₁ + m₂ * r₂ × dr₂ + m₃ * r₃ × dr₃
    C₄ = potential_energy(system, r₁, r₂, r₃) + kinetic_energy(system, dr₁, dr₂, dr₃)

    constraints[1:3] .= C₁
    constraints[4:6] .= C₂
    constraints[7:9] .= C₃
    constraints[10] = C₄

    return nothing
end
