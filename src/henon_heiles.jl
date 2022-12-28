struct HenonHeiles{T} <: AbstractDynamicalSystem{T}
    λ::T
end

HenonHeiles{T}() where {T} = HenonHeiles{T}(1.0)

function rhs(u::AbstractVector{T}, system::HenonHeiles, t) where {T}
    (; λ) = system
    x, y, p_x, p_y  = u

    return [
        p_x,
        p_y,
        -x - 2λ * x * y,
        -y - λ * (x^2 - y^2),
    ]
end

function jacobian_rhs(u::AbstractVector{T}, system::HenonHeiles{T}, t) where {T}
    (; λ) = system
    x, y = u

    #! format: off
    return [zero(T)        zero(T)        one(T)    zero(T)
            zero(T)        zero(T)        zero(T)   one(T)
            -1 - 2λ * y    -2λ * x        zero(T)   zero(T)
            -2λ * x        -1 + 2λ * y    zero(T)   zero(T)]  # Check this versus ForwardDiff
    #! format: on
end

function rhs!(du::AbstractVector{T}, u::AbstractVector{T}, system::HenonHeiles{T}, t) where {T}
    (; λ) = system
    x, y, p_x, p_y  = u

    du[1] = p_x
    du[2] = p_y
    du[3] = -x - 2λ * x * y
    du[4] = -y - λ * (x^2 - y^2)

    return nothing
end

function hamiltonian(system::HenonHeiles{T}, u::AbstractVector{T}) where {T}
    (; λ) = system
    x, y, p_x, p_y  = u

    return [0.5 * (p_x^2 + p_y^2) + 0.5 * (x^2 + y^2) + λ * (x^2 * y - y^3 / 3)]
end

function jacobian_hamiltonian(system::HenonHeiles{T}, u::AbstractVector{T}) where {T}
    (; λ) = system
    x, y, p_x, p_y  = u
    
    return [x + 2λ * y;; y + λ * (x^2 - y^2);; p_x;; p_y]
end
