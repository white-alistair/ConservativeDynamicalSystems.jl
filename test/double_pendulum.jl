@testset "double_pendulum.jl" begin
    @testset "Dynamics and Constraints" begin
        # We indirectly test both the dynamics and the constraints by integrating the 
        # system forward in time and verifying that the integrals of motion are indeed
        # conserved.
        Random.seed!(1)
        T = Float64
        system = DoublePendulum{T}(rand(T, 4)...)
        u0 = rand(T, 4)
        t0 = zero(T)
        tspan = (t0, T(10.0))

        @testset "In-place" begin
            constraints = zeros(T, 1)
            constraints!(constraints, u0, system, t0)
            initial_constraints = constraints
            prob = ODEProblem(rhs!, u0, tspan, system)
            sol = solve(prob, alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
            constraints!(constraints, sol.u[end], system, sol.t[end])
            @test initial_constraints ≈ constraints
        end

        @testset "Out-of-place" begin
            initial_constraints = constraints(u0, system, t0)
            prob = ODEProblem(rhs, u0, tspan, system)
            sol = solve(prob, alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
            final_constraints = constraints(sol.u[end], system, sol.t[end])
            @test initial_constraints ≈ final_constraints
        end
    end 
end
