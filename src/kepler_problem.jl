struct KeplerProblem{T} <: AbstractDynamicalSystem{T} end

function get_default_initial_conditions(::KeplerProblem{T}) where {T}
    e = 0.6
    return T[1 - e, 0, 0, sqrt((1 + e) / (1 - e))]
end

function rhs!(du, u, ::KeplerProblem, t)
    q₁, q₂, p₁, p₂ = u

    du[1] = p₁
    du[2] = p₂
    du[3] = -q₁ / (q₁^2 + q₂^2)^(3 / 2)
    du[4] = -q₂ / (q₁^2 + q₂^2)^(3 / 2)

    return nothing
end

function (system::KeplerProblem)(du, u, p, t)
    rhs!(du, u, system, t)
    return nothing
end

function hamiltonian(u, ::KeplerProblem{T}, t) where {T}
    q₁, q₂, p₁, p₂ = u
    return T(0.5) * (p₁^2 + p₂^2) - 1 / sqrt(q₁^2 + q₂^2)
end

function angular_momentum(u, ::KeplerProblem, t)
    q₁, q₂, p₁, p₂ = u
    return q₁ * p₂ - q₂ * p₁
end

function invariants(u, system::KeplerProblem{T}, t) where {T}
    return T[hamiltonian(u, system, t), angular_momentum(u, system, t)]
end

function invariants_jacobian(u, ::KeplerProblem{T}, t) where {T}
    q₁, q₂, p₁, p₂ = u

    #! format: off
    return T[
        q₁*(q₁^2+q₂^2)^(-3/2)       q₂*(q₁^2+q₂^2)^(-3/2)       p₁      p₂
        p₂                          -p₁                         -q₂     q₁
    ]
    #! format: on
end
