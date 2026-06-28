include("../header.jl")
import ManifoldsBase: check_point, check_size, check_vector, vector_transport_to_project!
include("../../src/manifolds/Veronese.jl")

@testset "Veronese Manifold" begin
    unit_p(n, k) = 1 / sqrt(k) .* [ones(k)..., zeros(n - k)...]
    unit_X(n, k; l = 1.0) = l / sqrt(n - k) .* [zeros(k)..., ones(n - k)...]

    M_even = Veronese(4, 2)
    M_odd = Veronese(4, 3)

    p_even = [[0.5], unit_p(4, 2)]
    q_even = [[0.8], -unit_p(4, 2)]
    X_even = [[0.2], unit_X(4, 2)]

    p_odd = [[0.7], unit_p(4, 2)]
    q_odd = [[0.9], -unit_p(4, 2)]
    X_odd = [[0.3], unit_X(4, 2)]

    @testset "Constructor" begin
        @test M_even == Veronese{ℝ, 4, 2}()
        @test M_odd == Veronese{ℝ, 4, 3}()
    end

    @testset "is_point" begin
        @test is_point(M_even, p_even; error = :error)
        @test is_point(M_odd, p_odd; error = :error)

        @test_throws DomainError is_point(M_even, [[1.0, 0.0], p_even[2]]; error = :error)
        @test_throws DomainError is_point(M_even, [p_even[1], [1.0, 0.0]]; error = :error)
        @test_throws DomainError is_point(M_even, [[-1.0], p_even[2]]; error = :error)
        @test_throws DomainError is_point(M_even, [p_even[1], 2 .* p_even[2]]; error = :error)
    end

    @testset "is_vector" begin
        @test is_vector(M_even, p_even, X_even; error = :error)
        @test is_vector(M_odd, p_odd, X_odd; error = :error)

        @test_throws DomainError is_vector(
            M_even,
            [[1.0, 0.0], p_even[2]],
            X_even;
            error = :error,
        )
        @test_throws DomainError is_vector(
            M_even,
            p_even,
            [[1.0, 0.0], X_even[2]];
            error = :error,
        )
        @test_throws DomainError is_vector(
            M_even,
            p_even,
            [[0.0], p_even[2]];
            error = :error,
        )
    end

    @testset "closest_representative!" begin
        q = deepcopy(q_even)
        closest_representative!(M_even, q, p_even)
        @test q[1] == q_even[1]
        @test isapprox(q[2], p_even[2])

        q = deepcopy(q_odd)
        closest_representative!(M_odd, q, p_odd)
        @test q == q_odd
    end

    @testset "embed!" begin
        q = zeros(4^2)
        embed!(M_even, q, p_even)
        @test isapprox(q, p_even[1][1] .* kron(p_even[2], p_even[2]))

        u = zeros(4^2)
        embed!(M_even, u, p_even, X_even)
        expected =
            X_even[1][1] .* kron(p_even[2], p_even[2]) +
            p_even[1][1] .* kron(X_even[2], p_even[2]) +
            p_even[1][1] .* kron(p_even[2], X_even[2])
        @test isapprox(u, expected)
    end

    @testset "vector_transport_to_project!" begin
        Y = [[0.0], zeros(4)]
        vector_transport_to_project!(M_even, Y, p_even, X_even, p_even)
        @test isapprox(Y[1], X_even[1]; atol = 1.0e-12)
        @test isapprox(Y[2], X_even[2]; atol = 1.0e-12)
        @test is_vector(M_even, p_even, Y; error = :error)

        q = [[0.9], unit_p(4, 1)]
        Y2 = [[0.0], zeros(4)]
        vector_transport_to_project!(M_odd, Y2, p_odd, X_odd, q)
        @test is_vector(M_odd, q, Y2; error = :error)

        Y3 = vector_transport_to(M_even, p_even, X_even, p_even, ProjectionTransport())
        @test isapprox(Y3[1], X_even[1]; atol = 1.0e-12)
        @test isapprox(Y3[2], X_even[2]; atol = 1.0e-12)
    end
end
