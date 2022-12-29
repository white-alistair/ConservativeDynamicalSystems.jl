struct HenonHeilesSystem{T} <: AbstractDynamicalSystem{T}
    λ::T
end

HenonHeilesSystem{T}() where {T} = HenonHeilesSystem{T}(1.0)

function rhs(u::AbstractVector{T}, system::HenonHeilesSystem, t) where {T}
    (; λ) = system
    x, y, p_x, p_y  = u

    return [
        p_x,
        p_y,
        -x - 2λ * x * y,
        -y - λ * (x^2 - y^2),
    ]
end

function rhs_jacobian(u::AbstractVector{T}, system::HenonHeilesSystem{T}, t) where {T}
    (; λ) = system
    x, y = u

    #! format: off
    return [zero(T)        zero(T)        one(T)    zero(T)
            zero(T)        zero(T)        zero(T)   one(T)
            -1 - 2λ * y    -2λ * x        zero(T)   zero(T)
            -2λ * x        -1 + 2λ * y    zero(T)   zero(T)]  # Check this versus ForwardDiff
    #! format: on
end

function rhs!(du::AbstractVector{T}, u::AbstractVector{T}, system::HenonHeilesSystem{T}, t) where {T}
    (; λ) = system
    x, y, p_x, p_y  = u

    du[1] = p_x
    du[2] = p_y
    du[3] = -x - 2λ * x * y
    du[4] = -y - λ * (x^2 - y^2)

    return nothing
end

function hamiltonian(u::AbstractVector{T}, system::HenonHeilesSystem{T}, t::T) where {T}
    (; λ) = system
    x, y, p_x, p_y  = u
    return 0.5 * (p_x^2 + p_y^2) + 0.5 * (x^2 + y^2) + λ * (x^2 * y - y^3 / 3)
end


function constraints(u::AbstractVector{T}, system::HenonHeilesSystem{T}, t::T) where {T}
    return [hamiltonian(u, system, t)]
end

function constraints_jacobian(u::AbstractVector{T}, system::HenonHeilesSystem{T}, t::T) where {T}
    (; λ) = system
    x, y, p_x, p_y  = u
    
    return [x + 2λ * y;; y + λ * (x^2 - y^2);; p_x;; p_y]
end
