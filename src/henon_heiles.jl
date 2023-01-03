struct HenonHeilesSystem{T} <: AbstractDynamicalSystem{T}
    λ::T
end

HenonHeilesSystem{T}() where {T} = HenonHeilesSystem{T}(1.0)

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

function hamiltonian(u::AbstractVector{T}, system::HenonHeilesSystem{T}, t::T) where {T}
    (; λ) = system
    x, y, p_x, p_y = u
    return 0.5 * (p_x^2 + p_y^2) + 0.5 * (x^2 + y^2) + λ * (x^2 * y - y^3 / 3)
end


function constraints(u::AbstractVector{T}, system::HenonHeilesSystem{T}, t::T) where {T}
    return [hamiltonian(u, system, t)]
end

function constraints!(
    constraints::AbstractVector{T},
    u::AbstractVector{T},
    system::DoublePendulum{T},
    t::T,
) where {T}
    constraints[1] = hamiltonian(u, system, t)
    return nothing
end

function constraints_jacobian(
    u::AbstractVector{T},
    system::HenonHeilesSystem{T},
    t::T,
) where {T}
    (; λ) = system
    x, y, p_x, p_y = u

    return [x + 2λ * y;; y + λ * (x^2 - y^2);; p_x;; p_y]
end

function constraints_jacobian!(
    J::AbstractMatrix{T},
    u::AbstractVector{T},
    system::HenonHeilesSystem{T},
    t::T,
) where {T}
    (; λ) = system
    x, y, p_x, p_y = u

    J[1, 1] = x + 2λ * y
    J[1, 2] = y + λ * (x^2 - y^2)
    J[1, 3] = p_x
    J[1, 4] = p_y

    return nothing
end
