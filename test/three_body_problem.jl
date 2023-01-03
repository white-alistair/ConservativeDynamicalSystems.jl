@testset "three_body_problem.jl" begin
    @testset "Dynamics and Invariants" begin
        # We indirectly test both the dynamics and the constraints by integrating the 
        # system forward in time and verifying that the integrals of motion are indeed
        # conserved.
        Random.seed!(1)
        T = Float64
        system = ThreeBodyProblem{T}(rand(T, 3)...)
        u0 = rand(T, 18)
        t0 = zero(T)
        tspan = (t0, T(10.0)) 
        initial_invariants = invariants(u0, system, t0)
        prob = ODEProblem(rhs!, u0, tspan, system)
        sol = solve(prob, alg = Vern9(), abstol = 1e-12, reltol = 1e-12)
        final_invariants = invariants(sol.u[end], system, sol.t[end])
        @test initial_invariants ≈ final_invariants
    end

    @testset "Invariants Jacobian" begin
        Random.seed!(1)
        T = Float64
        system = ThreeBodyProblem{T}(rand(T, 3)...)
        u = rand(T, 18)
        t = rand(T)

        # To test the constraints Jacobian functions, we use a Jacobian matrix pre-computed 
        # from the constraints function using ForwardDiff.jl
        jac_forward_diff = [
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07336635446929285 0.0 0.0 0.34924148955718615 0.0 0.0 0.6988266836914685 0.0 0.0
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07336635446929285 0.0 0.0 0.34924148955718615 0.0 0.0 0.6988266836914685 0.0
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07336635446929285 0.0 0.0 0.34924148955718615 0.0 0.0 0.6988266836914685
            0.07336635446929285 0.0 0.0 0.34924148955718615 0.0 0.0 0.6988266836914685 0.0 0.0 -0.03213872126606809 0.0 0.0 -0.1529880415705085 0.0 0.0 -0.3061266456935804 0.0 0.0
            0.0 0.07336635446929285 0.0 0.0 0.34924148955718615 0.0 0.0 0.6988266836914685 0.0 0.0 -0.03213872126606809 0.0 0.0 -0.1529880415705085 0.0 0.0 -0.3061266456935804 0.0
            0.0 0.0 0.07336635446929285 0.0 0.0 0.34924148955718615 0.0 0.0 0.6988266836914685 0.0 0.0 -0.03213872126606809 0.0 0.0 -0.1529880415705085 0.0 0.0 -0.3061266456935804
            0.0 0.04343587193760876 -0.0006014470177914453 0.0 0.05675305008546285 -0.15140713275274173 0.0 0.13755158356781597 -0.5437910853532599 0.0 -0.05631000161775742 0.050759046999399156 0.0 -0.24278547673047948 0.2988521186388531 0.0 -0.2430077880795771 0.3204455359792997
            -0.04343587193760876 0.0 0.014564755425744298 -0.05675305008546285 0.0 0.2797616920286758 -0.13755158356781597 0.0 0.1096048185765871 0.05631000161775742 0.0 -0.0017677117266357131 0.24278547673047948 0.0 -0.03047238477069215 0.2430077880795771 0.0 -0.5202271091087228
            0.0006014470177914453 -0.014564755425744298 0.0 0.15140713275274173 -0.2797616920286758 0.0 0.5437910853532599 -0.1096048185765871 0.0 -0.050759046999399156 0.0017677117266357131 0.0 -0.2988521186388531 0.03047238477069215 0.0 -0.3204455359792997 0.5202271091087228 0.0
            0.29312356455316224 0.5943946643662436 -0.3037093459059475 0.031692653757900896 -0.7747375097724782 0.1289050569937341 -0.3248162183110631 0.18034284540623474 0.1748042889122134 0.014564755425744298 0.0006014470177914453 0.04343587193760876 0.2797616920286758 0.15140713275274173 0.05675305008546285 0.1096048185765871 0.5437910853532599 0.13755158356781597
        ]

        @testset "In-place" begin
            jac = invariants_jacobian(u, system, t)
            @test jac ≈ jac_forward_diff
        end

        @testset "Out-of-place" begin
            jac = zeros(T, 10, 18)
            invariants_jacobian!(jac, u, system, t)
            @test jac ≈ jac_forward_diff
        end
    end
end
