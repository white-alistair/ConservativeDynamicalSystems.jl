struct Robertson{T} <: AbstractDynamicalSystem{T} end

function get_default_initial_conditions(::Robertson{T}) where {T}
    return T[1.0, 0.0, 0.0]
end

function rhs!(du, u, ::Robertson, t)
    x, y, z = u
    du[1] = -0.04x + 1e4 * y * z
    du[2] = 0.04x - 1e4* y * z - 3e7 * y^2
    du[3] = 3e7 * y^2
    return nothing
end

function (system::Robertson)(du, u, p, t)
    rhs!(du, u, system, t)
    return nothing
end
