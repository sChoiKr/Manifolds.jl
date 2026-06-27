# veronese_manifold.jl
#
# Prototype Veronese manifold implementation in the same spirit as a Segre manifold.
#
# Mathematical model:
#     p = (λ, x),  λ > 0,  x ∈ S^{n-1}
#     Φ(λ, x) = λ x^{⊗d}
#
# Tangent vector:
#     X = (ν, u),  u ∈ T_x S^{n-1}, so dot(x, u) = 0
#     DΦ_p[X] = ν x^{⊗d} + λ Σ_{k=1}^d x⊗...⊗u⊗...⊗x
#
# This prototype stores the ambient symmetric tensor as a full vector of length n^d.
# That is mathematically correct but not memory-optimal. Later, replace this with
# compressed symmetric tensor / monomial-basis storage.

using LinearAlgebra
using Random

# If this is placed inside Manifolds.jl / ManifoldsBase extension code, uncomment:
# using ManifoldsBase
# using Manifolds

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

"""
    kron_power(x, d)

Return vec(x^{⊗d}) using repeated Kronecker products.
"""
function kron_power(x::AbstractVector, d::Int)
    d ≥ 1 || throw(ArgumentError("degree d must be positive"))
    y = x
    for _ in 2:d
        y = kron(y, x)
    end
    return y
end

"""
    kron_replace(x, u, d, k)

Return vec(x ⊗ ... ⊗ u ⊗ ... ⊗ x), with `u` in position `k`.
"""
function kron_replace(x::AbstractVector, u::AbstractVector, d::Int, k::Int)
    d ≥ 1 || throw(ArgumentError("degree d must be positive"))
    1 ≤ k ≤ d || throw(BoundsError(1:d, k))

    y = (k == 1) ? u : x
    for j in 2:d
        y = kron(y, j == k ? u : x)
    end
    return y
end

"""
    normalize_sphere!(x)

Normalize `x` in-place to unit Euclidean norm.
"""
function normalize_sphere!(x::AbstractVector)
    nx = norm(x)
    nx > 0 || throw(DomainError(nx, "cannot normalize the zero vector"))
    x ./= nx
    return x
end

"""
    project_to_tangent_sphere!(u, x)

Project `u` in-place onto T_x S^{n-1}.
"""
function project_to_tangent_sphere!(u::AbstractVector, x::AbstractVector)
    u .-= dot(x, u) .* x
    return u
end

"""
    contract_rank_one_term_to_veronese_tangent!(b, c, zs, x)

For one ambient rank-one term c z₁⊗...⊗z_d, accumulate

    b += Σ_k c (Π_{j≠k} ⟨z_j, x⟩) z_k.

This is the vector part needed for projecting an ambient tensor to the
Veronese tangent space at x.
"""
function contract_rank_one_term_to_veronese_tangent!(
    b::AbstractVector,
    c,
    zs::AbstractVector,
    x::AbstractVector,
)
    d = length(zs)
    dots = [dot(z, x) for z in zs]

    for k in 1:d
        coeff = c
        for j in 1:d
            j == k && continue
            coeff *= dots[j]
        end
        b .+= coeff .* zs[k]
    end

    return b
end

# ----------------------------------------------------------------------
# Manifold type
# ----------------------------------------------------------------------

"""
    Veronese(n, d; field=Real)

Affine cone over the degree-`d` Veronese variety, represented by

    (λ, x) ∈ ℝ⁺ × S^{n-1}

and embedded as

    Φ(λ, x) = λ x^{⊗d} ∈ (ℝ^n)^{⊗d}.

For even `d`, `(λ, x)` and `(λ, -x)` represent the same tensor.
For odd `d`, they represent opposite tensors when λ is constrained positive.
"""
struct Veronese{𝔽, N, D} 
    <: AbstractManifold{𝔽}
end

function Veronese(n::Int, d::Int; field::AbstractNumbers = ℝ)
    n ≥ 1 || throw(ArgumentError("ambient vector dimension n must be positive"))
    d ≥ 1 || throw(ArgumentError("degree d must be positive"))
    return Veronese{field, n, d}()
end

valence(::Veronese{𝔽,N,D}) where {𝔽,N,D} = N
degree(::Veronese{𝔽,N,D}) where {𝔽,N,D} = D

# ----------------------------------------------------------------------
# Allocation and validation
# ----------------------------------------------------------------------

"""
    allocate_point(M::Veronese)

Allocate a point representation `[λ, x]`.
"""
function allocate_point(M::Veronese{𝔽,N,D}) where {𝔽,N,D}
    return [zeros(1), zeros(N)]
end

"""
    allocate_ambient(M::Veronese)

Allocate a full ambient vector representation of λ x^{⊗d}.
"""
function allocate_ambient(M::Veronese{𝔽,N,D}) where {𝔽,N,D}
    return zeros(N^D)
end

"""
    check_size(M, p)
"""
function check_size(M::Veronese{𝔽,N,D}, p) where {𝔽,N,D}
    length(p) == 2 || return DomainError(length(p), "A Veronese point must be [λ, x].")
    length(p[1]) == 1 || return DomainError(size(p[1]), "λ must be stored as a length-1 vector.")
    length(p[2]) == N || return DomainError(length(p[2]), "x must have length $N.")
    return nothing
end

"""
    check_size(M, p, X)
"""
function check_size(M::Veronese{𝔽,N,D}, p, X) where {𝔽,N,D}
    e = check_size(M, p)
    isnothing(e) || return e
    length(X) == 2 || return DomainError(length(X), "A Veronese tangent vector must be [ν, u].")
    length(X[1]) == 1 || return DomainError(size(X[1]), "ν must be stored as a length-1 vector.")
    length(X[2]) == N || return DomainError(length(X[2]), "u must have length $N.")
    return nothing
end

"""
    check_point(M, p; atol=sqrt(eps()))

Check p = [λ, x], λ > 0, ||x|| = 1.
"""
function check_point(M::Veronese{𝔽,N,D}, p; atol=sqrt(eps(Float64))) where {𝔽,N,D}
    e = check_size(M, p)
    isnothing(e) || return e

    λ = p[1][1]
    x = p[2]

    λ > 0 || return DomainError(λ, "Veronese weight λ must be positive.")
    abs(norm(x) - 1) ≤ atol || return DomainError(norm(x), "x must lie on the unit sphere.")

    return nothing
end

"""
    check_vector(M, p, X; atol=sqrt(eps()))

Check X = [ν, u], u ⟂ x.
"""
function check_vector(M::Veronese{𝔽,N,D}, p, X; atol=sqrt(eps(Float64))) where {𝔽,N,D}
    e = check_size(M, p, X)
    isnothing(e) || return e

    x = p[2]
    u = X[2]

    abs(dot(x, u)) ≤ atol || return DomainError(dot(x, u), "u must be tangent to the sphere at x.")

    return nothing
end

# ----------------------------------------------------------------------
# Random points and tangent vectors
# ----------------------------------------------------------------------

"""
    rand_point!(M, p; rng=Random.default_rng())

Fill `p` with a random Veronese point.
"""
function rand_point!(M::Veronese{𝔽,N,D}, p; rng=Random.default_rng()) where {𝔽,N,D}
    p[1][1] = exp(randn(rng))
    randn!(rng, p[2])
    normalize_sphere!(p[2])
    return p
end

"""
    rand_vector!(M, X, p; rng=Random.default_rng())

Fill `X` with a random tangent vector at p.
"""
function rand_vector!(M::Veronese{𝔽,N,D}, X, p; rng=Random.default_rng()) where {𝔽,N,D}
    X[1][1] = randn(rng)
    randn!(rng, X[2])
    project_to_tangent_sphere!(X[2], p[2])
    return X
end

# ----------------------------------------------------------------------
# Embedding and tangent embedding
# ----------------------------------------------------------------------

"""
    embed!(M, q, p)

q .= λ vec(x^{⊗d})
"""
function embed!(M::Veronese{𝔽,N,D}, q::AbstractVector, p) where {𝔽,N,D}
    q .= p[1][1] .* kron_power(p[2], D)
    return q
end

"""
    embed(M, p)
"""
function embed(M::Veronese, p)
    q = allocate_ambient(M)
    return embed!(M, q, p)
end

"""
    embed_tangent!(M, q, p, X)

q .= DΦ_p[X].
"""
function embed_tangent!(M::Veronese{𝔽,N,D}, q::AbstractVector, p, X) where {𝔽,N,D}
    λ = p[1][1]
    ν = X[1][1]
    x = p[2]
    u = X[2]

    q .= ν .* kron_power(x, D)

    for k in 1:D
        q .+= λ .* kron_replace(x, u, D, k)
    end

    return q
end

function embed_tangent(M::Veronese, p, X)
    q = allocate_ambient(M)
    return embed_tangent!(M, q, p, X)
end

# ----------------------------------------------------------------------
# Riemannian metric induced by the full ambient Frobenius metric
# ----------------------------------------------------------------------

"""
    inner(M, p, X, Y)

Induced metric:

    ⟨(ν,u), (ξ,v)⟩_p = νξ + d λ² ⟨u,v⟩,

because u,v are tangent to the sphere and cross terms vanish.
"""
function inner(M::Veronese{F,N,D}, p, X, Y) where {F,N,D}
    λ = p[1][1]
    return X[1][1] * Y[1][1] + D * λ^2 * dot(X[2], Y[2])
end

manifold_dimension(M::Veronese{F,N,D}) where {F,N,D} = N

ambient_dimension(M::Veronese{F,N,D}) where {F,N,D} = N^D

# ----------------------------------------------------------------------
# Projection from structured rank-one terms to Veronese tangent
# ----------------------------------------------------------------------

"""
    project_rank_terms_to_tangent!(M, Y, q, terms)

Project an ambient tensor A onto T_q Veronese, where A is provided as
a list of rank-one terms

    terms = [(c₁, [z₁₁, ..., z₁d]), ..., (c_R, [z_R1, ..., z_Rd])].

The result is Y = [dotλ, v].

Formula:
    dotλ = ⟨A, x^{⊗d}⟩
    b    = Σ_terms Σ_k c (Π_{j≠k} ⟨z_j, x⟩) z_k
    v    = Π_{x⊥}(b) / (λ d)
"""
function project_rank_terms_to_tangent!(M::Veronese{F,N,D}, Y, q, terms) where {F,N,D}
    λ = q[1][1]
    x = q[2]

    Y[1][1] = zero(eltype(Y[1]))
    fill!(Y[2], zero(eltype(Y[2])))

    b = similar(x)
    fill!(b, zero(eltype(b)))

    for (c, zs) in terms
        length(zs) == D || throw(DomainError(length(zs), "Each rank-one term must have D=$D factors."))

        dots = [dot(z, x) for z in zs]
        Y[1][1] += c * prod(dots)

        contract_rank_one_term_to_veronese_tangent!(b, c, zs, x)
    end

    # Tangential projection and metric inverse.
    project_to_tangent_sphere!(b, x)
    Y[2] .= b ./ (λ * D)

    return Y
end

"""
    vector_transport_to_project!(M, Y, p, X, q)

Projection vector transport:
    Y = Π_{T_qM}(DΦ_p[X])

without forming the ambient tensor.

For p=(μ,y), X=(ν,u), the embedded tangent is

    ν y^{⊗d} + μ Σ_r y⊗...⊗u⊗...⊗y.

This is represented as d+1 rank-one terms and projected to T_qM.
"""
function vector_transport_to_project!(M::Veronese{𝔽,N,D}, Y, p, X, q) where {𝔽,N,D}
    μ = p[1][1]
    ν = X[1][1]
    y = p[2]
    u = X[2]

    terms = Vector{Tuple{typeof(ν), Vector{typeof(y)}}}()

    # radial term: ν y⊗...⊗y
    push!(terms, (ν, [y for _ in 1:D]))

    # sphere terms: μ y⊗...⊗u⊗...⊗y
    for r in 1:D
        zs = [k == r ? u : y for k in 1:D]
        push!(terms, (μ, zs))
    end

    return project_rank_terms_to_tangent!(M, Y, q, terms)
end

# ----------------------------------------------------------------------
# Simple retraction and vector projection
# ----------------------------------------------------------------------

"""
    retract!(M, q, p, X)

A simple first-order retraction:
    λ_new = max(λ + ν, eps)
    x_new = normalize(x + u)

This is not the exact exponential map. It is enough for a first working
optimization prototype.
"""
function retract!(M::Veronese, q, p, X)
    q[1][1] = max(p[1][1] + X[1][1], eps(eltype(p[1])))
    q[2] .= p[2] .+ X[2]
    normalize_sphere!(q[2])
    return q
end

"""
    project_tangent!(M, X, p)

Project the vector component of X onto T_x S^{n-1}.
"""
function project_tangent!(M::Veronese, X, p)
    project_to_tangent_sphere!(X[2], p[2])
    return X
end

# ----------------------------------------------------------------------
# Sign representative for even degree
# ----------------------------------------------------------------------

"""
    closest_representative!(M, q, p)

For even degree, x and -x give the same embedded tensor, so choose the
sphere representative of q closest to p. For odd degree, do nothing.
"""
function closest_representative!(M::Veronese{𝔽,N,D}, q, p) where {𝔽,N,D}
    if iseven(D) && dot(p[2], q[2]) < 0
        q[2] .*= -1
    end
    return q
end

# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------

Base.show(io::IO, ::Veronese{𝔽,N,D}) where {𝔽,N,D} =
    print(io, "Veronese($N, $D; field=$𝔽)")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

function _smoke_test()
    M = Veronese(4, 3)

    p = allocate_point(M)
    q = allocate_point(M)
    X = allocate_point(M)
    Y = allocate_point(M)

    rand_point!(M, p)
    rand_point!(M, q)
    rand_vector!(M, X, p)

    A = embed(M, p)
    dA = embed_tangent(M, p, X)

    vector_transport_to_project!(M, Y, p, X, q)

    @assert isnothing(check_point(M, p))
    @assert isnothing(check_vector(M, p, X))
    @assert length(A) == ambient_dimension(M)
    @assert length(dA) == ambient_dimension(M)
    @assert isnothing(check_vector(M, q, Y))

    println("M = ", M)
    println("dimension(M) = ", manifold_dimension(M))
    println("ambient_dimension(M) = ", ambient_dimension(M))
    println("inner(M,p,X,X) = ", inner(M, p, X, X))
    println("smoke test passed")
end

# Run smoke test when this file is executed directly.
if abspath(PROGRAM_FILE) == @__FILE__
    _smoke_test()
end
