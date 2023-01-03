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
    return -1 * (m₁ * m₂ / r₁₂ + m₁ * m₃ / r₁₃ + m₂ * m₃ / r₂₃)
end

function constraints(u::AbstractVector{T}, system::ThreeBodyProblem{T}, t) where {T}
    (; m₁, m₂, m₃) = system
    r₁, r₂, r₃ = get_positions(u)
    dr₁, dr₂, dr₃ = get_velocities(u)

    # Integrals of motion
    # Here we follow the notation of https://arxiv.org/abs/1508.02312
    # C₁, C₂ and C₃ are three-element vectors while C₄ is a scalar
    C₁ = m₁ * dr₁ + m₂ * dr₂ + m₃ * dr₃
    C₂ = m₁ * r₁ + m₂ * r₂ + m₃ * r₃ - C₁ * t
    C₃ = m₁ * r₁ × dr₁ + m₂ * r₂ × dr₂ + m₃ * r₃ × dr₃
    C₄ = potential_energy(system, r₁, r₂, r₃) + kinetic_energy(system, dr₁, dr₂, dr₃)

    return [C₁..., C₂..., C₃..., C₄]
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
    # Here we follow the notation of https://arxiv.org/abs/1508.02312
    # C₁, C₂ and C₃ are three-element vectors while C₄ is a scalar
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

function constraints_jacobian(
    u::AbstractVector{T},
    system::ThreeBodyProblem{T},
    t::T,
) where {T}
    (; m₁, m₂, m₃) = system
    r₁₂, r₁₃, r₂₃ = get_separations(u)
    x₁, y₁, z₁, x₂, y₂, z₂, x₃, y₃, z₃, dx₁, dy₁, dz₁, dx₂, dy₂, dz₂, dx₃, dy₃, dz₃ = u
    J = zeros(T, 10, 18)

    #! format: off
    # C₁
    J[1, 10] = m₁
    J[1, 13] = m₂
    J[1, 16] = m₃

    J[2, 11] = m₁
    J[2, 14] = m₂
    J[2, 17] = m₃

    J[3, 12] = m₁
    J[3, 15] = m₂
    J[3, 18] = m₃

    # C₂
    J[4, 1]  = m₁
    J[4, 4]  = m₂
    J[4, 7]  = m₃
    J[4, 10] = -m₁ * t
    J[4, 13] = -m₂ * t
    J[4, 16] = -m₃ * t

    J[5, 2]  = m₁
    J[5, 5]  = m₂
    J[5, 8]  = m₃
    J[5, 11] = -m₁ * t
    J[5, 14] = -m₂ * t
    J[5, 17] = -m₃ * t

    J[6, 3]  = m₁
    J[6, 6]  = m₂
    J[6, 9]  = m₃
    J[6, 12] = -m₁ * t
    J[6, 15] = -m₂ * t
    J[6, 18] = -m₃ * t

    # C₃
    J[7, 2]  =  m₁ * dz₁
    J[7, 3]  = -m₁ * dy₁
    J[7, 5]  =  m₂ * dz₂
    J[7, 6]  = -m₂ * dy₂
    J[7, 8]  =  m₃ * dz₃
    J[7, 9]  = -m₃ * dy₃
    J[7, 11] = -m₁ * z₁
    J[7, 12] =  m₁ * y₁
    J[7, 14] = -m₂ * z₂
    J[7, 15] =  m₂ * y₂
    J[7, 17] = -m₃ * z₃
    J[7, 18] =  m₃ * y₃

    J[8, 1]  = -m₁ * dz₁
    J[8, 3]  =  m₁ * dx₁
    J[8, 4]  = -m₂ * dz₂
    J[8, 6]  =  m₂ * dx₂
    J[8, 7]  = -m₃ * dz₃
    J[8, 9]  =  m₃ * dx₃
    J[8, 10] =  m₁ * z₁
    J[8, 12] = -m₁ * x₁
    J[8, 13] =  m₂ * z₂
    J[8, 15] = -m₂ * x₂
    J[8, 16] =  m₃ * z₃
    J[8, 18] = -m₃ * x₃

    J[9, 1]  =  m₁ * dy₁
    J[9, 2]  = -m₁ * dx₁
    J[9, 4]  =  m₂ * dy₂
    J[9, 5]  = -m₂ * dx₂
    J[9, 7]  =  m₃ * dy₃
    J[9, 8]  = -m₃ * dx₃
    J[9, 10] = -m₁ * y₁
    J[9, 11] =  m₁ * x₁
    J[9, 13] = -m₂ * y₂
    J[9, 14] =  m₂ * x₂
    J[9, 16] = -m₃ * y₃
    J[9, 17] =  m₃ * x₃

    # C₄
    J[10, 1]  = -m₁ * m₂ * (x₁ - x₂) / r₁₂^3 - m₁ * m₃ * (x₁ - x₃) / r₁₃^3
    J[10, 2]  = -m₁ * m₂ * (y₁ - y₂) / r₁₂^3 - m₁ * m₃ * (y₁ - y₃) / r₁₃^3
    J[10, 3]  = -m₁ * m₂ * (z₁ - z₂) / r₁₂^3 - m₁ * m₃ * (z₁ - z₃) / r₁₃^3
    J[10, 4]  =  m₁ * m₂ * (x₁ - x₂) / r₁₂^3 - m₂ * m₃ * (x₂ - x₃) / r₂₃^3
    J[10, 5]  =  m₁ * m₂ * (y₁ - y₂) / r₁₂^3 - m₂ * m₃ * (y₂ - y₃) / r₂₃^3
    J[10, 6]  =  m₁ * m₂ * (z₁ - z₂) / r₁₂^3 - m₂ * m₃ * (z₂ - z₃) / r₂₃^3
    J[10, 7]  =  m₁ * m₃ * (x₁ - x₃) / r₁₃^3 + m₂ * m₃ * (x₂ - x₃) / r₂₃^3
    J[10, 8]  =  m₁ * m₃ * (y₁ - y₃) / r₁₃^3 + m₂ * m₃ * (y₂ - y₃) / r₂₃^3
    J[10, 9]  =  m₁ * m₃ * (z₁ - z₃) / r₁₃^3 + m₂ * m₃ * (z₂ - z₃) / r₂₃^3
    J[10, 10] = m₁ * dx₁
    J[10, 11] = m₁ * dy₁
    J[10, 12] = m₁ * dz₁
    J[10, 13] = m₂ * dx₂
    J[10, 14] = m₂ * dy₂
    J[10, 15] = m₂ * dz₂
    J[10, 16] = m₃ * dx₃
    J[10, 17] = m₃ * dy₃
    J[10, 18] = m₃ * dz₃
    #! format: on

    return J
end

function constraints_jacobian!(
    J::AbstractMatrix{T},
    u::AbstractVector{T},
    system::ThreeBodyProblem{T},
    t::T,
) where {T}
    (; m₁, m₂, m₃) = system
    r₁₂, r₁₃, r₂₃ = get_separations(u)
    x₁, y₁, z₁, x₂, y₂, z₂, x₃, y₃, z₃, dx₁, dy₁, dz₁, dx₂, dy₂, dz₂, dx₃, dy₃, dz₃ = u

    #! format: off
    # C₁
    J[1, 10] = m₁
    J[1, 13] = m₂
    J[1, 16] = m₃

    J[2, 11] = m₁
    J[2, 14] = m₂
    J[2, 17] = m₃

    J[3, 12] = m₁
    J[3, 15] = m₂
    J[3, 18] = m₃

    # C₂
    J[4, 1]  = m₁
    J[4, 4]  = m₂
    J[4, 7]  = m₃
    J[4, 10] = -m₁ * t
    J[4, 13] = -m₂ * t
    J[4, 16] = -m₃ * t

    J[5, 2]  = m₁
    J[5, 5]  = m₂
    J[5, 8]  = m₃
    J[5, 11] = -m₁ * t
    J[5, 14] = -m₂ * t
    J[5, 17] = -m₃ * t

    J[6, 3]  = m₁
    J[6, 6]  = m₂
    J[6, 9]  = m₃
    J[6, 12] = -m₁ * t
    J[6, 15] = -m₂ * t
    J[6, 18] = -m₃ * t

    # C₃
    J[7, 2]  =  m₁ * dz₁
    J[7, 3]  = -m₁ * dy₁
    J[7, 5]  =  m₂ * dz₂
    J[7, 6]  = -m₂ * dy₂
    J[7, 8]  =  m₃ * dz₃
    J[7, 9]  = -m₃ * dy₃
    J[7, 11] = -m₁ * z₁
    J[7, 12] =  m₁ * y₁
    J[7, 14] = -m₂ * z₂
    J[7, 15] =  m₂ * y₂
    J[7, 17] = -m₃ * z₃
    J[7, 18] =  m₃ * y₃

    J[8, 1]  = -m₁ * dz₁
    J[8, 3]  =  m₁ * dx₁
    J[8, 4]  = -m₂ * dz₂
    J[8, 6]  =  m₂ * dx₂
    J[8, 7]  = -m₃ * dz₃
    J[8, 9]  =  m₃ * dx₃
    J[8, 10] =  m₁ * z₁
    J[8, 12] = -m₁ * x₁
    J[8, 13] =  m₂ * z₂
    J[8, 15] = -m₂ * x₂
    J[8, 16] =  m₃ * z₃
    J[8, 18] = -m₃ * x₃

    J[9, 1]  =  m₁ * dy₁
    J[9, 2]  = -m₁ * dx₁
    J[9, 4]  =  m₂ * dy₂
    J[9, 5]  = -m₂ * dx₂
    J[9, 7]  =  m₃ * dy₃
    J[9, 8]  = -m₃ * dx₃
    J[9, 10] = -m₁ * y₁
    J[9, 11] =  m₁ * x₁
    J[9, 13] = -m₂ * y₂
    J[9, 14] =  m₂ * x₂
    J[9, 16] = -m₃ * y₃
    J[9, 17] =  m₃ * x₃

    # C₄
    J[10, 1]  = -m₁ * m₂ * (x₁ - x₂) / r₁₂^3 - m₁ * m₃ * (x₁ - x₃) / r₁₃^3
    J[10, 2]  = -m₁ * m₂ * (y₁ - y₂) / r₁₂^3 - m₁ * m₃ * (y₁ - y₃) / r₁₃^3
    J[10, 3]  = -m₁ * m₂ * (z₁ - z₂) / r₁₂^3 - m₁ * m₃ * (z₁ - z₃) / r₁₃^3
    J[10, 4]  =  m₁ * m₂ * (x₁ - x₂) / r₁₂^3 - m₂ * m₃ * (x₂ - x₃) / r₂₃^3
    J[10, 5]  =  m₁ * m₂ * (y₁ - y₂) / r₁₂^3 - m₂ * m₃ * (y₂ - y₃) / r₂₃^3
    J[10, 6]  =  m₁ * m₂ * (z₁ - z₂) / r₁₂^3 - m₂ * m₃ * (z₂ - z₃) / r₂₃^3
    J[10, 7]  =  m₁ * m₃ * (x₁ - x₃) / r₁₃^3 + m₂ * m₃ * (x₂ - x₃) / r₂₃^3
    J[10, 8]  =  m₁ * m₃ * (y₁ - y₃) / r₁₃^3 + m₂ * m₃ * (y₂ - y₃) / r₂₃^3
    J[10, 9]  =  m₁ * m₃ * (z₁ - z₃) / r₁₃^3 + m₂ * m₃ * (z₂ - z₃) / r₂₃^3
    J[10, 10] = m₁ * dx₁
    J[10, 11] = m₁ * dy₁
    J[10, 12] = m₁ * dz₁
    J[10, 13] = m₂ * dx₂
    J[10, 14] = m₂ * dy₂
    J[10, 15] = m₂ * dz₂
    J[10, 16] = m₃ * dx₃
    J[10, 17] = m₃ * dy₃
    J[10, 18] = m₃ * dz₃
    #! format: on

    return nothing
end
