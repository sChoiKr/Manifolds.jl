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