function image_basis(χ::Character)
    mpr = isirreducible(χ) ? matrix_projection_irr(χ) : matrix_projection(χ)
    image, pivots = image_basis!(mpr)
    return image[1:length(pivots), :]
end

function image_basis(hom::InducedActionHomomorphism, χ::Character)
    mpr = isirreducible(χ) ? matrix_projection_irr(hom, χ) : matrix_projection(hom, χ)
    image, pivots = image_basis!(mpr)
    return image[1:length(pivots), :]
end

function image_basis(α::AlgebraElement)
    image, pivots = image_basis!(matrix_projection(α))
    return image[1:length(pivots), :]
end

function image_basis(hom::InducedActionHomomorphism, α::AlgebraElement)
    image, pivots = image_basis!(matrix_projection(hom, α))
    return image[1:length(pivots), :]
end

struct DirectSummand{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    basis::M
    multiplicity::Int
    degree::Int
    simple::Bool
end

StarAlgebras.basis(ds::DirectSummand) = ds.basis
issimple(ds::DirectSummand) = ds.simple
degree(ds::DirectSummand) = ds.degree
multiplicity(ds::DirectSummand) = ds.multiplicity

Base.size(ds::DirectSummand) = size(ds.basis)
Base.@propagate_inbounds Base.getindex(ds::DirectSummand, i...) =
    basis(ds)[i...]

function affordable_real(
    irreducible_characters,
    multiplicities=fill(1, length(irreducible_characters)),
)
    irr_real = similar(irreducible_characters, 0)
    mls_real = similar(multiplicities, 0)
    for (i, χ) in pairs(irreducible_characters)
        ι = frobenius_schur(χ)
        if abs(ι) == 1 # real or quaternionic
            @debug "real/quaternionic:" χ
            push!(irr_real, χ)
            push!(mls_real, multiplicities[i])
        else # complex one...
            cχ = conj(χ)
            k = findfirst(==(cχ), irreducible_characters)
            @assert !isnothing(k)
            @debug "complex" χ conj(χ)=irreducible_characters[k]
            if k > i # ... we haven't already observed a conjugate of
                @assert multiplicities[i] == multiplicities[k]
                push!(irr_real, χ + cχ)
                push!(mls_real, multiplicities[i])
            end
        end
    end

    return irr_real, mls_real
end

"""
    symmetry_adapted_basis([T::Type,] G::AbstractPermutationGroup[, S=Rational{Int}])
Compute a basis for the linear space `ℝⁿ` which is invariant under the symmetry of `G`.

The permutation group is acting naturally on `1:degree(G)`. The coefficients of
the invariant basis are returned in (orthogonal) blocks corresponding to irreducible
characters of `G`.

Arguments:
* `S` controls the types of `Cyclotomic`s used in the computation of
character table. Exact type are preferred. For larger groups `G` `Rational{BigInt}`
might be necessary.
* `T` controls the type of coefficients of the returned basis.

!!! Note:
Each block is invariant under the action of `G`, i.e. the action may permute
vectors from symmetry adapted basis within each block. The blocks are guaranteed
to be orthogonal. If `T<:LinearAlgebra.BlasFloat` BLAS routines will be used to
orthogonalize vectors within each block.
"""
symmetry_adapted_basis(G::AbstractPermutationGroup, S::Type = Rational{Int}) =
    _symmetry_adapted_basis(characters_dixon(S, G))

symmetry_adapted_basis(T::Type, G::AbstractPermutationGroup, S::Type = Rational{Int}) =
    symmetry_adapted_basis(T, characters_dixon(S, G))

"""
    symmetry_adapted_basis([T::Type,] G::Group, basis, action[, S=Rational{Int}])
Compute a basis for the linear space spanned by `basis` which is invariant under
the symmetry of `G`.

* The action used in these computations is
> `(b,g) → action(b,g)` for `b ∈ basis`, `g ∈ G`
and needs to be defined by the user.
* It is assumed that `G` acts on a subset of basis and the action needs to be
extended to the whole `basis`. If `G` is a permutation group already acting on
the whole `basis`, a call to `symmetry_adapted_basis(G)` is preferred.
* For inducing the action `basis` needs to be indexable and iterable
(e.g. in the form of an `AbstractVector`).

Arguments:
* `S` controlls the types of `Cyclotomic`s used in the computation of
character table. Exact type are preferred. For larger groups `G` `Rational{BigInt}`
might be necessary.
* `T` controls the type of coefficients of the returned basis.

!!! Note:
Each block is invariant under the action of `G`, i.e. the action may permute
vectors from symmetry adapted basis within each block. The blocks are guaranteed
to be orthogonal. If `T<:LinearAlgebra.BlasFloat` BLAS routines will be used to
orthogonalize vectors within each block.
"""
function symmetry_adapted_basis(G::Group, basis, action, S::Type = Rational{Int})
    chars_ext = extended_characters(S, G, basis, action)
    return symmetry_adapted_basis(chars_ext)
end

function symmetry_adapted_basis(
    ::Type{T},
    G::Group,
    basis,
    action,
    ::Type{S} = Rational{Int},
) where {T,S}
    chars_ext = extended_characters(S, G, basis, action)
    return symmetry_adapted_basis(T, chars_ext)
end

symmetry_adapted_basis(chars::AbstractVector{<:AbstractClassFunction{T}}) where T =
    symmetry_adapted_basis(T, chars)

symmetry_adapted_basis(::Type{T}, chars::AbstractVector{<:AbstractClassFunction}) where T =
    _symmetry_adapted_basis(Character{T}.(chars))

symmetry_adapted_basis(T::Type{<:Real}, chars::AbstractVector{<:AbstractClassFunction}) =
    _symmetry_adapted_basis(Character{T}.(affordable_real(chars)))

symmetry_adapted_basis(::Type{T}, chars::AbstractVector{<:AbstractClassFunction{T}}) where T =
    _symmetry_adapted_basis(chars)


_multiplicities(ψ, chars) = Int[div(Int(dot(ψ, χ)), Int(dot(χ, χ))) for χ in chars]
_multiplicities(ψ, chars::AbstractVector{<:AbstractClassFunction{T}}) where T<:AbstractFloat =
    [round(Int, dot(ψ, χ) / dot(χ, χ)) for χ in chars]
_multiplicities(ψ, chars::AbstractVector{<:AbstractClassFunction{T}}) where T<:Complex =
    [round(Int, real(dot(ψ, χ) / dot(χ, χ))) for χ in chars]

macro spawn_compat(expr)
    @static if VERSION < v"1.3.0"
        return :(@async $(esc(expr)))
    else
        return :(Threads.@spawn $(esc(expr)))
    end
end

function symmetry_adapted_basis2(T::Type, G::Group, S::Type{<:Rational} = Rational{Int}, simple=true)
    tbl = CharacterTable(S, G)
    ψ = action_character(conjugacy_classes(tbl), tbl)

    irr_chars = irreducible_characters(tbl)
    let multiplicities = constituents(ψ), degrees = degree.(irr_chars)

        @debug "Decomposition into character spaces:
        degrees:        $(join([lpad(d, 6) for d in degrees], ""))
        multiplicities: $(join([lpad(m, 6) for m in multiplicities], ""))"

        dot(multiplicities, degrees) == degree(ψ) ||
            @error "Something went wrong: characters do not constitute a complete basis for action:
            $(dot(multiplicities, degrees)) ≠ $(degree(ψ))"
    end

    RG = let G = parent(tbl)
        b = StarAlgebras.Basis{UInt16}(vec(collect(G)))
        StarAlgebra(G, b, (length(b), length(b)))
    end

    existing_chars = [i for (i, m) in enumerate(constituents(ψ)) if m ≠ 0]
    multiplicities = constituents(ψ)[existing_chars]

    mps, simple = minimal_projection_system(irr_chars[existing_chars], small_idempotents(RG))

    res = map(zip(mps, multiplicities, simple)) do (µ, m, s)
        µT = AlgebraElement{T}(µ)
        @spawn_compat SemisimpleSummand(simple_basis(µT), m, s)
    end

    return fetch.(res)
end
