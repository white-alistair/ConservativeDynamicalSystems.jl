struct HenonHeilesSystem{T} <: AbstractDynamicalSystem{T}
    λ::T
end

HenonHeilesSystem{T}() where {T} = HenonHeilesSystem{T}(1.0)

function get_default_initial_conditions(::HenonHeilesSystem{T}) where {T}
    return T[0.365, 0.365, 0.0, 0.0]  # H0 ≈ 0.1656 < 1/6, Λₘₐₓ ≈ 0.117
end

function rhs(u::AbstractVector{T}, system::HenonHeilesSystem, t) where {T}
    (; λ) = system
    x, y, p_x, p_y = u

    return [p_x, p_y, -x - 2λ * x * y, -y - λ * (x^2 - y^2)]
end

function rhs!(
    du::AbstractVector{T},
    u::AbstractVector{T},
    system::HenonHeilesSystem{T},
    t,
) where {T}
    (; λ) = system
    x, y, p_x, p_y = u

    du[1] = p_x
    du[2] = p_y
    du[3] = -x - 2λ * x * y
    du[4] = -y - λ * (x^2 - y^2)

    return nothing
end

function hamiltonian(u, system::HenonHeilesSystem{T}, t) where {T}
    (; λ) = system
    x, y, p_x, p_y = u
    return T(0.5) * (p_x^2 + p_y^2 + x^2 + y^2) + λ * (x^2 * y - y^3 / 3)
end

function invariants(u, system::HenonHeilesSystem, t)
    return [hamiltonian(u, system, t)]
end

function invariants!(
    constraints,
    u,
    system::HenonHeilesSystem,
    t,
)
    constraints[1] = hamiltonian(u, system, t)
    return nothing
end

function invariants_jacobian(
    u,
    system::HenonHeilesSystem,
    t,
)
    (; λ) = system
    x, y, p_x, p_y = u

    return [x + 2λ * y;; y + λ * (x^2 - y^2);; p_x;; p_y]
end

function invariants_jacobian!(
    J,
    u,
    system::HenonHeilesSystem,
    t,
)
    (; λ) = system
    x, y, p_x, p_y = u

    J[1, 1] = x + 2λ * y
    J[1, 2] = y + λ * (x^2 - y^2)
    J[1, 3] = p_x
    J[1, 4] = p_y

    return nothing
end
