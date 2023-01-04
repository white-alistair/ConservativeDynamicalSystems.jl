@testset "qbh.jl" begin
    @testset "Dynamics and Invariants" begin
        # We indirectly test both the dynamics and the constraints by integrating the 
        # system forward in time and verifying that the integrals of motion are indeed
        # conserved.
        Random.seed!(1)
        T = Float64
        system = QuadrupoleBosonHamiltonian{T}()
        u0 = rand(T, 4)
        t0 = zero(T)
        tspan = (t0, T(10.0))

        @testset "In-place" begin
            invariants = zeros(T, 1)
            invariants!(invariants, u0, system, t0)
            initial_invariants = copy(invariants)
            prob = ODEProblem(rhs!, u0, tspan, system)
            sol = solve(prob, alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
            invariants!(invariants, sol.u[end], system, sol.t[end])
            @test initial_invariants ≈ invariants
        end

        @testset "Out-of-place" begin
            initial_invariants = invariants(u0, system, t0)
            prob = ODEProblem(rhs, u0, tspan, system)
            sol = solve(prob, alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
            final_invariants = invariants(sol.u[end], system, sol.t[end])
            @test initial_invariants ≈ final_invariants
        end
    end 
end
