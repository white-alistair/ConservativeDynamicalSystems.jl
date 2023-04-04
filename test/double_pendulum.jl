# We indirectly test both the dynamics and the constraints by integrating the 
# system forward in time and verifying that the integrals of motion are indeed
# conserved.

@testitem "Dynamics and Invariants: in-place" begin
    using OrdinaryDiffEq, Random
    
    Random.seed!(1)
    T = Float64
    system = DoublePendulum{T}(rand(T, 4)...)
    u0 = rand(T, 4)
    t0 = zero(T)
    tspan = (t0, T(10.0))

    invariants = zeros(T, 1)
    invariants!(invariants, u0, system, t0)
    initial_invariants = copy(invariants)
    prob = ODEProblem(rhs!, u0, tspan, system)
    sol = solve(prob; alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
    invariants!(invariants, sol.u[end], system, sol.t[end])
    @test initial_invariants ≈ invariants
end

@testitem "Dynamics and Invariants: out-of-place" begin
    using OrdinaryDiffEq, Random

    Random.seed!(1)
    T = Float64
    system = DoublePendulum{T}(rand(T, 4)...)
    u0 = rand(T, 4)
    t0 = zero(T)
    tspan = (t0, T(10.0))

    initial_invariants = invariants(u0, system, t0)
    prob = ODEProblem(rhs, u0, tspan, system)
    sol = solve(prob; alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
    final_invariants = invariants(sol.u[end], system, sol.t[end])
    @test initial_invariants ≈ final_invariants
end

@testitem "Invariants Jacobian" begin
    using ForwardDiff, Random

    Random.seed!(1)
    T = Float64
    system = DoublePendulum{T}(rand(T, 4)...)
    u = rand(T, 4)

    jac_forward_diff = ForwardDiff.jacobian(u -> invariants(u, system, nothing), u)
    jac_analytic = invariants_jacobian(u, system, nothing)

    @test jac_analytic ≈ jac_forward_diff atol = 1e-5
end

@testitem "Dynamics and Invariants: in-place (Hamiltonian Formulation)" begin
    using OrdinaryDiffEq, Random
    
    Random.seed!(1)
    T = Float64
    system = DoublePendulumHamiltonian{T}(rand(T, 4)...)
    u0 = rand(T, 4)
    t0 = zero(T)
    tspan = (t0, T(10.0))

    invariants = zeros(T, 1)
    invariants!(invariants, u0, system, t0)
    initial_invariants = copy(invariants)
    prob = ODEProblem(rhs!, u0, tspan, system)
    sol = solve(prob; alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
    invariants!(invariants, sol.u[end], system, sol.t[end])
    @test initial_invariants ≈ invariants
end

@testitem "Dynamics and Invariants: out-of-place (Hamiltonian Formulation)" begin
    using OrdinaryDiffEq, Random

    Random.seed!(1)
    T = Float64
    system = DoublePendulumHamiltonian{T}(rand(T, 4)...)
    u0 = rand(T, 4)
    t0 = zero(T)
    tspan = (t0, T(10.0))

    initial_invariants = invariants(u0, system, t0)
    prob = ODEProblem(rhs, u0, tspan, system)
    sol = solve(prob; alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
    final_invariants = invariants(sol.u[end], system, sol.t[end])
    @test initial_invariants ≈ final_invariants
end

@testitem "Invariants Jacobian (Hamiltonian Formulation)" begin
    using ForwardDiff, Random

    Random.seed!(1)
    T = Float64
    system = DoublePendulumHamiltonian{T}(rand(T, 4)...)
    u = rand(T, 4)

    jac_forward_diff = ForwardDiff.jacobian(u -> invariants(u, system, nothing), u)
    jac_analytic = invariants_jacobian(u, system, nothing)
    
    @test jac_analytic ≈ jac_forward_diff atol = 1e-5
end

@testitem "Cartesian" begin
    system = DoublePendulum{Float64}(rand(4)...)
    (; l₁, l₂) = system

    # Single observation
    u = [π/2, π/2, 1.0, 2.0]
    u_cartesian = ConservativeDynamicalSystems.cartesian(system, u)
    @test u_cartesian ≈ [l₁, 0.0, l₁ + l₂, 0.0, 0.0, l₁, 0.0, l₁ + 2l₂]

    # Whole trajectory
    trajectory = [π/2 π/2
                  π/2 π/4
                  1.0 2.0
                  2.0 1.0]
    traj_cartesian = ConservativeDynamicalSystems.cartesian(system, trajectory)
    @test traj_cartesian ≈ [l₁          l₁
                            0.0         0.0
                            l₁ + l₂     l₁ + l₂ / sqrt(2)
                            0.0         -l₂ / sqrt(2)
                            0.0         0.0
                            l₁          2l₁
                            0.0         l₂ / sqrt(2)
                            l₁ + 2l₂    2l₁ + l₂ / sqrt(2)
                            ]
end
