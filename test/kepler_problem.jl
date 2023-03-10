# We indirectly test both the dynamics and the constraints by integrating the 
# system forward in time and verifying that the integrals of motion are indeed
# conserved.

@testitem "Dynamics and Invariants" begin
    using OrdinaryDiffEq, Random

    Random.seed!(1)
    T = Float64
    system = KeplerProblem{T}()
    u0 = rand(T, 4)
    t0 = zero(T)
    tspan = (t0, T(100.0))
    initial_invariants = invariants(u0, system, t0)
    prob = ODEProblem(rhs!, u0, tspan, system)
    sol = solve(prob; alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
    final_invariants = invariants(sol.u[end], system, sol.t[end])
    @test initial_invariants ≈ final_invariants
end

@testitem "Invariants Jacobian" begin
    using ForwardDiff, Random

    Random.seed!(1)
    T = Float64
    system = KeplerProblem{T}()
    u = rand(T, 4)

    jac_forward_diff = ForwardDiff.jacobian(u -> invariants(u, system, nothing), u)
    jac = invariants_jacobian(u, system, nothing)
    
    @test jac ≈ jac_forward_diff atol = 1e-5
end
