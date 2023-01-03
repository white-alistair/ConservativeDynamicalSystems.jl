@testset "vector_products.jl" begin
    @testset "dot_product" begin
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        c = ConservativeDynamicalSystems.dot_product(a, b)
        @test c ≈ 32.0
    end

    @testset "cross_product" begin
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        c = ConservativeDynamicalSystems.cross_product(a, b)
        @test c ≈ [-3.0, 6.0, -3.0]
    end
end
