# src/manifolds/Veronese.jl
@doc raw"""
    Veronese{𝔽, V} <: AbstractManifold{𝔽}

The Veronese manifold is the set of all points on the unit sphere in ``ℝ^{n+1}`` that lie on a line through the origin.

# Constructor
    Veronese(n::Int...; field::AbstractNumbers = ℝ)

Generate a valence `(n, ...)` Veronese manifold.
"""
struct Veronese{𝔽, V} <: AbstractManifold{𝔽} end

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

