include("../header.jl")
import ManifoldsBase: check_point, check_size, check_vector, distance, vector_transport_to_project!
include("../../src/manifolds/Veronese.jl")

@testset "Veronese Manifold" begin
    # Manifolds to test
    Ms = [Veronese(4, 1), Veronese(4, 2), Veronese(4, 3)]

    # Ns[i] is the ambient vector dimension of Ms[i]
    Ns = [4, 4, 4]

    # Ds[i] is the degree of Ms[i]
    Ds = [1, 2, 3]

    # n ≥ k, for same n,k X is in TpM and can be scaled by l
    unit_p(n, k) = 1 / sqrt(k) .* [ones(k)..., zeros(n - k)...]
    unit_X(n, k; l = 1.0) = l / sqrt(n - k) .* [zeros(k)..., ones(n - k)...]

    # ps[i] is a point on Ms[i]
    ps = [
        [[0.6], unit_p(4, 2)],
        [[0.5], unit_p(4, 2)],
        [[0.7], unit_p(4, 2)],
    ]

    # qs[i] is a point on Ms[i] connected to ps[i] by a simple exact example
    qs = [
        [[0.9], unit_p(4, 1)],
        [[0.8], -unit_p(4, 2)],
        [[0.9], -unit_p(4, 2)],
    ]

    # Xs[i] is a tangent vector to Ms[i] at ps[i]
    Xs = [
        [[0.1], unit_X(4, 2)],
        [[0.2], unit_X(4, 2)],
        [[0.3], unit_X(4, 2)],
    ]

    # point_tensors[i] is the exact embedded tensor corresponding to ps[i]
    point_tensors = [
        ps[1][1][1] .* ps[1][2],
        ps[2][1][1] .* kron(ps[2][2], ps[2][2]),
        ps[3][1][1] .* kron(ps[3][2], ps[3][2], ps[3][2]),
    ]

    # tangent_tensors[i] is the exact embedded tangent tensor corresponding to Xs[i] at ps[i]
    tangent_tensors = [
        Xs[1][1][1] .* ps[1][2] + ps[1][1][1] .* Xs[1][2],
        Xs[2][1][1] .* kron(ps[2][2], ps[2][2]) +
        ps[2][1][1] .* kron(Xs[2][2], ps[2][2]) +
        ps[2][1][1] .* kron(ps[2][2], Xs[2][2]),
        Xs[3][1][1] .* kron(ps[3][2], ps[3][2], ps[3][2]) +
        ps[3][1][1] .* kron(Xs[3][2], ps[3][2], ps[3][2]) +
        ps[3][1][1] .* kron(ps[3][2], Xs[3][2], ps[3][2]) +
        ps[3][1][1] .* kron(ps[3][2], ps[3][2], Xs[3][2]),
    ]

    for (M, n, d, p, q, X, p_tensor, X_tensor) in
        zip(Ms, Ns, Ds, ps, qs, Xs, point_tensors, tangent_tensors)
        @testset "Manifold $M" begin
            @testset "Constructor" begin
                @test M == Veronese{ℝ, n, d}()
            end

            @testset "is_point" begin
                @test is_point(M, p; error = :error)

                @test_throws DomainError is_point(M, [[1.0, 0.0], p[2]]; error = :error)
                @test_throws DomainError is_point(M, [p[1], [1.0, 0.0]]; error = :error)
                @test_throws DomainError is_point(M, [[-1.0], p[2]]; error = :error)
                @test_throws DomainError is_point(M, [p[1], 2 .* p[2]]; error = :error)
            end

            @testset "is_vector" begin
                @test is_vector(M, p, X; error = :error)

                @test_throws DomainError is_vector(
                    M,
                    [[1.0, 0.0], p[2]],
                    X;
                    error = :error,
                )
                @test_throws DomainError is_vector(
                    M,
                    p,
                    [[1.0, 0.0], X[2]];
                    error = :error,
                )
                @test_throws DomainError is_vector(
                    M,
                    p,
                    [[0.0], p[2]];
                    error = :error,
                )
            end

            @testset "closest_representative!" begin
                q_ = deepcopy(q)
                closest_representative!(M, q_, p)
                @test q_[1] == q[1]
                if iseven(d)
                    @test isapprox(q_[2], p[2])
                else
                    @test q_ == q
                end
            end

            @testset "distance" begin
                @test isapprox(distance(M, p, deepcopy(p)), 0.0; atol = 1.0e-12)
                if iseven(d)
                    @test isapprox(distance(M, p, deepcopy(q)), abs(p[1][1] - q[1][1]); atol = 1.0e-12)
                else
                    θ = distance(Sphere(n - 1), p[2], q[2])
                    expected = sqrt((p[1][1] - q[1][1])^2 + 4 * p[1][1] * q[1][1] * sin(θ / 2)^2)
                    @test isapprox(distance(M, p, deepcopy(q)), expected; atol = 1.0e-12)
                end
            end

            @testset "embed!" begin
                q_tensor = zeros(n^d)
                embed!(M, q_tensor, p)
                @test isapprox(q_tensor, p_tensor)

                u_tensor = zeros(n^d)
                embed!(M, u_tensor, p, X)
                @test isapprox(u_tensor, X_tensor)
            end

            @testset "vector_transport_to_project!" begin
                Y = [[0.0], zeros(n)]
                ret = vector_transport_to_project!(M, Y, p, X, p)
                @test ret === Y
                @test isapprox(Y[1], X[1]; atol = 1.0e-12)
                @test isapprox(Y[2], X[2]; atol = 1.0e-12)
                @test is_vector(M, p, Y; error = :error)

                Y2 = [[0.0], zeros(n)]
                ret2 = vector_transport_to_project!(M, Y2, p, X, q)
                @test ret2 === Y2
                @test is_vector(M, q, Y2; error = :error)

                Y3 = vector_transport_to(M, p, X, p, ProjectionTransport())
                @test isapprox(Y3[1], X[1]; atol = 1.0e-12)
                @test isapprox(Y3[2], X[2]; atol = 1.0e-12)
            end
        end
    end
end
