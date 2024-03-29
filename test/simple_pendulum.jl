# We indirectly test both the dynamics and the constraints by integrating the 
# system forward in time and verifying that the integrals of motion are indeed
# conserved.

@testitem "Dynamics and Invariants: in-place" tags=[:skip] begin
    using OrdinaryDiffEq, Random

    Random.seed!(1)
    T = Float64
    system = SimplePendulum{T}(rand(T, 2)...)
    u0 = rand(T, 2)
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
    system = SimplePendulum{T}(rand(T, 2)...)
    u0 = rand(T, 2)
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
    system = SimplePendulum{T}(rand(T, 2)...)
    u = rand(T, 2)

    jac_forward_diff = ForwardDiff.jacobian(u -> invariants(u, system, nothing), u)
    jac_analytic = invariants_jacobian(u, system, nothing)

    @test jac_analytic ≈ jac_forward_diff atol = 1e-5
end

@testitem "Cartesian: vector" begin
    system = SimplePendulum{Float64}(rand(2)...)
    (; l) = system
    traj_angular = [π/2 π/4
                    1.0 2.0]
    traj_cartesian = ConservativeDynamicalSystems.cartesian(system, traj_angular)
    @test traj_cartesian ≈ [l     l/sqrt(2)
                            0.0   -l/sqrt(2)
                            0.0   2l/sqrt(2)
                            l     2l/sqrt(2)]
end

@testitem "Cartesian" begin
    system = SimplePendulum{Float64}(rand(2)...)
    (; l) = system

    # Single observation
    u = [π/2, 1.0]
    u_cartesian = ConservativeDynamicalSystems.cartesian(system, u)
    @test u_cartesian ≈ [l, 0.0, 0.0, l]

    # Whole trajectory
    trajectory = [π/2 π/4
                  1.0 2.0]
    traj_cartesian = ConservativeDynamicalSystems.cartesian(system, trajectory)
    @test traj_cartesian ≈ [l     l/sqrt(2)
                            0.0   -l/sqrt(2)
                            0.0   2l/sqrt(2)
                            l     2l/sqrt(2)]
end
