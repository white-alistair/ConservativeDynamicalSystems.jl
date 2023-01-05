struct QuadrupoleBosonHamiltonian{T} <: AbstractDynamicalSystem{T}
    A::T
    B::T
    D::T
end

QuadrupoleBosonHamiltonian{T}() where {T} = QuadrupoleBosonHamiltonian{T}(1.0, 0.55, 0.4)

function get_default_initial_conditions(::QuadrupoleBosonHamiltonian{T}) where {T}
    return T[0.0, -2.5830294658973876, 1.3873470962626937, -4.743416490252585]
end

function rhs(u::AbstractVector{T}, system::QuadrupoleBosonHamiltonian, t) where {T}
    q₀, q₂, p₀, p₂ = u
    (; A, B, D) = system

    return [
        A * p₀,
        A * p₂,
        -1 * A * q₀ - 3 * B / sqrt(2) * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2),
        -1 * q₂ * (A + 3 * sqrt(2) * B * q₀ + D * (q₀^2 + q₂^2)),
    ]
end

function rhs!(
    du::AbstractVector{T},
    u::AbstractVector{T},
    system::QuadrupoleBosonHamiltonian{T},
    t,
) where {T}
    q₀, q₂, p₀, p₂ = u
    (; A, B, D) = system

    du[1] = A * p₀
    du[2] = A * p₂
    du[3] = -A * q₀ - 3 * B / sqrt(2) * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2)
    du[4] = -q₂ * (A + 3 * sqrt(2) * B * q₀ + D * (q₀^2 + q₂^2))

    return nothing
end

function hamiltonian(
    u,
    system::QuadrupoleBosonHamiltonian,
    t,
)
    q₀, q₂, p₀, p₂ = u
    (; A, B, D) = system
    return (
        A / 2 * (q₀^2 + q₂^2 + p₀^2 + p₂^2) +
        B / sqrt(2) * q₀ * (3q₂^2 - q₀^2) +
        D / 4 * (q₀^2 + q₂^2)^2
    )
end

function invariants(
    u,
    system::QuadrupoleBosonHamiltonian,
    t,
)
    return [hamiltonian(u, system, t)]
end

function invariants!(
    invariants,
    u,
    system::QuadrupoleBosonHamiltonian,
    t,
)
    invariants[1] = hamiltonian(u, system, t)
    return nothing
end
