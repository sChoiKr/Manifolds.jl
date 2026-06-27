# src/manifolds/Veronese.jl
@doc raw"""
    Veronese{𝔽,N,D} <: AbstractManifold{𝔽}

The Veronese manifold

````math
\mathcal{V} = \operatorname{Ver}_D(𝔽^N)
`````

is the set of nonzero symmetric rank-one tensors of order `D` in

```math
\operatorname{Sym}^D(𝔽^N) \subset (𝔽^N)^{\otimes D}.
```
Equivalently, it is the symmetric, diagonal analogue of the Segre manifold: its points are rank-one tensors whose tensor factors are all equal,

```math
x \otimes x \otimes \cdots \otimes x = x^{\otimes D}.
```

When `𝔽 = ℝ`, the Veronese manifold is a normal Riemannian covering of

```math
\mathcal{P} = ℝ^{+} \times \mathbb{S}^{N - 1}
```

equipped with a warped product metric. The covering map is given by

```math
(\lambda, x) \mapsto \lambda x^{\otimes D}.
```

Here `N` is the dimension of the base vector space and `D` is the degree, or order, of the symmetric tensor.

The geometry of the Veronese manifold is discussed as a special case of the Segre--Veronese geometry in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite). It is named after [Giuseppe Veronese](https://en.wikipedia.org/wiki/Giuseppe_Veronese) (1854–1917).

# Constructor
Veronese(n::Int, d::Int; field::AbstractNumbers=ℝ)

Generate a degree `d` Veronese manifold over `𝔽^n`.
`Veronese(n, 1)` is the same as ``\mathbb{R}^{n} \setminus \{ 0 \}``.
"""
struct Veronese{𝔽,N,D} <: AbstractManifold{𝔽} end

function Veronese(n::Int...; field::AbstractNumbers = ℝ)
    return Veronese{field, (n...,)}()
end

function check_size(M::Veronese{𝔽, V}, p) where {𝔽, V}
    p_size = only.(size.(p))
    M_size = [1, V...]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end

function check_size(M::Veronese{𝔽, V}, p, X) where {𝔽, V}
    p_size = only.(size.(p))
    X_size = only.(size.(X))
    M_size = [1, V...]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    if X_size != M_size
        return DomainError(
            X_size,
            "The vector $(X) can not belong to the manifold $(M), since its size $(X_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end

@doc raw"""
    closest_representative!(M::Veronese{ℝ, N, D}, q, p)

When ``D`` is even, ``\mathcal{V}`` is a two-sheeted Riemannian covering of

````math
\mathcal{P} = ℝ^{+} \times \mathbb{S}^{N - 1}

with a warped product metric. The two representatives of the same equivalence class are

(\lambda, x)
\quad\text{and}\quad
(\lambda, -x),

since

(-x)^{\otimes D} = x^{\otimes D}.

When D is odd, every point has a unique representative in \mathcal{P}.
`closest_representative!(M, q, p)` changes the representative of `q` to the one that is closest to `p` in ``\mathcal{P}``.
"""
function closest_representative!(M::Veronese{ℝ, N, D}, q, p) where {N, D}
    if iseven(D) && distance(Sphere(N - 1), p[2], q[2]) > pi / 2
    q[2] .= -q[2]
    end
    return q
end

@doc raw"""
    embed(M::Veronese{𝔽, N, D}, p)
    embed!(M::Veronese{𝔽, N, D}, q, p)

Embed ``p ≐ (λ, x)`` in ``(𝔽^N)^{\otimes D}`` using the Kronecker product

````math
(λ, x) ↦ λ x^{\otimes D}.

The image is a symmetric rank-one tensor of order D.
"""
embed(::Veronese, p)

function embed!(::Veronese{𝔽, N, D}, q, p) where {𝔽, N, D}
λ = p[1][1]
x = p[2]

q .= λ .* kron(ntuple(_ -> x, Val(D))...)
return q

end

@doc raw"""
    embed!(M::Veronese{𝔽, N, D}, q, p, X)

Embed tangent vector X = (ν, u) at p ≐ (λ, x) in (𝔽^N)^{\otimes D}
using the Kronecker product.

````math
(ν, u) ↦ ν x^{\otimes D} + λ \sum_{i=1}^{D} x^{\otimes(i-1)} \otimes u \otimes x^{\otimes(D-i)}.
````

This is the differential of the Veronese parametrization

````math
Φ(λ, x) = λ x^{\otimes D}.
````
"""
function embed!(::Veronese{𝔽, N, D}, q, p, X) where {𝔽, N, D}
λ = p[1][1]
x = p[2]

ν = X[1][1]
u = X[2]

q .= ν .* kron(ntuple(_ -> x, Val(D))...)

for i in 1:D
    q .+= λ .* kron(ntuple(j -> j == i ? u : x, Val(D))...)
end

return q

end

@doc raw"""
    vector_transport_to(M::Veronese{ℝ, N, D}, Y, p, X, q, ::ProjectionTransport)

Compute projection vector transport on the [`Veronese`](@ref) manifold by projecting
the embedded tangent vector ``DΦ_p[X]`` onto the tangent space of `M` at `q`.

Let

````math
p = (μ, y),
\qquad
X = (ν, u) ∈ T_pM,
````
and let ``q = (λ, x)``.

Under the Veronese parametrization

````math
Φ(λ,x) = λ x^{\otimes D},
````

the embedded tangent vector at p is

````math
DΦ_p[X]
=
ν y^{\otimes D}
+
μ \sum_{r=1}^{D}
y^{\otimes(r-1)}
\otimes u
\otimes
y^{\otimes(D-r)}.
````
Projection vector transport from ``T_pM`` to ``T_qM`` is given by

````math
Y = Π_{T_qM}\bigl(DΦ_p[X]\bigr),
````

where
````math
Y = (\dot{\lambda}, v) ∈ T_qM.
````
Writing
````math
α = ⟨y,x⟩,
\qquad
β = ⟨u,x⟩,
````
the radial component is
````math
\dot{\lambda}
=
ν α^D
+
μ D β α^{D-1},
````
and the spherical tangent component is
````math
v = \frac{1}{λ} \left[ ν α^{D-1} P_x(y) + μ α^{D-1} P_x(u) + μ(D-1)β α^{D-2} P_x(y) \right],
````
where the last term is omitted when ``D=1`` and
````math
P_x(z) = z - ⟨z,x⟩x
````
is the orthogonal projection onto ``T_x\mathbb{S}^{N-1}``.
The implementation uses the symmetric rank-one structure and avoids explicitly
forming the ambient tensor.
"""
vector_transport_to(::Veronese{ℝ, N, D}, Y, p, X, q, ::ProjectionTransport) where {N, D} 

function vector_transport_to_project!(::Veronese{ℝ, N, D}, Y, p, X, q) where {N, D} 
    
    for Yi in Y 
        fill!(Yi, zero(eltype(Yi))) 
    end

    checkbounds(p, 1:2) 
    checkbounds(q, 1:2) 
    checkbounds(X, 1:2) 
    checkbounds(Y, 1:2) 
    
    @inbounds begin 
        λ = q[1][1] 
        μ = p[1][1] 
        ν = X[1][1] 
        x = q[2] 
        y = p[2] 
        u = X[2] 
    end

    α = dot(y, x)
    β = dot(u, x)

    Y[1][1] = ν * α^D + μ * D * β * α^(D - 1)

    coeff_y = ν * α^(D - 1)
    if D > 1
        coeff_y += μ * (D - 1) * β * α^(D - 2)
    end
    coeff_y /= λ
    coeff_u = μ * α^(D - 1) / λ
    v = Y[2]
    @inbounds for i in eachindex(v, x, y, u)
        v[i] = coeff_y * (y[i] - α * x[i]) + coeff_u * (u[i] - β * x[i])
    end
    return Y
end

@doc raw"""
    zero_vector(M::Veronese{𝔽, N, D}, p)

returns the zero tangent vector in the tangent space of the Veronese manifold at `p`.
"""
zero_vector(::Veronese{𝔽, N, D}, ::Any...) where {𝔽, N, D}

function zero_vector!(::Veronese{𝔽, N, D}, v, ::Any...) where {𝔽, N, D}
    fill!(v, zero(eltype(v)))
    return v
end