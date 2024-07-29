function affordable_real(
    irreducible_characters,
    multiplicities,
)
    irr_real = similar(irreducible_characters, 0)
    mls_real = similar(multiplicities, 0)
    for (i, χ) in pairs(irreducible_characters)
        ι = Characters.frobenius_schur(χ)
        if ι == 1 # real
            @debug "real" χ
            push!(irr_real, χ)
            push!(mls_real, multiplicities[i])
        elseif ι == -1 # quaternionic
            @debug "quaternionic" χ
            @assert iseven(multiplicities[i]) "Multiplicities of quaternionic character must be even."
            push!(irr_real, 2 * χ)
            push!(mls_real, div(multiplicities[i], 2))
        else # complex
            cχ = conj(χ)
            k = findfirst(==(cχ), irreducible_characters)
            @assert k !== nothing "Conjugate character not found."
            @debug "complex" χ conj(χ) = irreducible_characters[k]
            if k > i # we haven't already observed a conjugate
                @assert multiplicities[i] == multiplicities[k] "Multiplicities of complex conjugates must be equal."
                push!(irr_real, χ + cχ)
                push!(mls_real, multiplicities[i])
            end
        end
    end

    return irr_real, mls_real
end

function symmetry_adapted_basis(
    G::PG.AbstractPermutationGroup,
    S::Type = Rational{Int};
    semisimple::Bool = false,
)
    tbl = CharacterTable(S, G)
    return symmetry_adapted_basis(eltype(tbl), tbl; semisimple = semisimple)
end

"""
    symmetry_adapted_basis([T::Type,] G::AbstractPermutationGroup[, S=Rational{Int};
        semisimple=false])
Compute a decomposition of `V=𝕂ⁿ` (with `n = degree(G)`) into
subspaces invariant under the natural permutation action of `G`.

Consider the decomposition of `V` into irreducible (simple) subspaces
    `V ≅ m₁V₁ ⊕ ⋯ ⊕ mᵣVᵣ`
and write `Wᵢ = mᵢVᵢ` (the `mᵢ`-fold direct sum of `Vᵢ`). The decomposition is
returned as a vector of `DirectSummand{T}`s (blocks) corresponding
to the distinct irreducible characters of `G` (i.e. types of action, here from `1`
to `r`). Each block contains a basis for a `G`-invariant subspace in `V` (`Vᵢ` or `Wᵢ`). The blocks are guaranteed to be orthogonal to **each other**, however vectors within a single block may *not* be orthogonal.

If `T<:LinearAlgebra.BlasFloat` BLAS routines will be used to orthogonalize
vectors within each block.

!!! note:
Each returned block is invariant (as a subspace) under the action of `G`, which
means that the action may still e.g. permute (row) vectors, but only *within* each block.

Arguments:
* `T` controls the type of coefficients of the returned basis. Unless specified,
the coefficients will be computed exactly in the field of cyclotomic numbers.
If you know that the group has rational characters only (which happens e.g. for
the full symmetric groups) You may specify `Rational{Int}` here. For a group
with complex characters specifying `T<:Real` will result in the computation of the realified basis.
* `S` controls the types of `Cyclotomic`s used in the computation of
character table. Exact type are preferred. For larger groups `G` (or if
overflow occurs during the computation of characters) specifying
`Rational{BigInt}` might be necessary.
* `semisimple::Bool` controls the nature of the the returned basis.
  - `semisimple=true`: the returned basis consists of orthogonal blocks
  (`DirectSummand`s) which define an isomorphism
  `V ≅ Wₖ ⊕ ⋯ ⊕ Wₖ`.
  associated to isotypical components `Wₖ`, which are (in general) semi-simple.
  I.e. each direct summand `ds` may further decopose into `G`-invariant
  subspaces `Wₖ ≅ mₖVₖ`, all of the same _type_. Multiplicity `mₖ` can be
  obtained by calling `multiplicity(@ref)`.
  - `semisimple=false`: (the default) In addition to finding blocks `Wₖ`, an
  effort to find _minimal projection system_ is made, i.e all, some (or none!)
  of the returned blocks corresponds to a **projection** `V → Wₖ ≅ mₖVₖ → Vₖ` for a single irreducible subspace `Vₖ`. This means that some blocks can not be further decomposed into nontrivial `G`-invariant subspaces.
"""
function symmetry_adapted_basis(
    T::Type,
    G::PG.AbstractPermutationGroup,
    S::Type = Rational{Int};
    semisimple::Bool = false,
)
    return symmetry_adapted_basis(
        T,
        CharacterTable(S, G);
        semisimple = semisimple,
    )
end

function symmetry_adapted_basis(
    T::Type,
    tbl::CharacterTable;
    semisimple::Bool = false,
)
    irr, multips = _constituents_decomposition(
        action_character(T, conjugacy_classes(tbl), tbl),
        tbl,
    )

    if T <: Real
        irr, multips = affordable_real(irr, multips)
    end

    if semisimple || all(isone ∘ degree, irr)
        return _symmetry_adapted_basis(T, irr, multips)
    else
        RG = _group_algebra(parent(tbl))
        return _symmetry_adapted_basis(T, irr, multips, RG)
    end
end

"""
    symmetry_adapted_basis([T::Type,] G::Group, action, basis[, S=Rational{Int}];
        semisimple=false)
Compute a decomposition of `V=⟨basis⟩` into subspaces invariant under the
given action of `G`.

It is assumed that `G` acts on a subset of basis and the action needs to be
extended to the whole `basis`. If `G` is a permutation group already acting on
the whole `basis`, a call to `symmetry_adapted_basis(G)` is preferred.

* For inducing the action `basis` needs to be indexable and iterable
(e.g. in the form of an `AbstractVector`).

!!! note !!!
    A [GroupActionError](@ref) is an indicator that the group action defined by
    `(G, action, basis)` might be incorrect.
"""
function symmetry_adapted_basis(
    G::Group,
    action::Action,
    basis,
    S::Type = Rational{Int};
    semisimple = false,
)
    tbl = CharacterTable(S, G)
    ehom = SchreierExtensionHomomorphism(
        parent(tbl),
        action,
        basis;
        memoize = true,
    )
    check_group_action(G, ehom; full_check = false)
    return symmetry_adapted_basis(
        eltype(tbl),
        tbl,
        ehom;
        semisimple = semisimple,
    )
end

function symmetry_adapted_basis(
    T::Type,
    G::Group,
    action::Action,
    basis,
    S::Type = Rational{Int};
    semisimple = false,
)
    tbl = CharacterTable(S, G)
    ehom = SchreierExtensionHomomorphism(
        parent(tbl),
        action,
        basis;
        memoize = true,
    )
    check_group_action(G, ehom; full_check = false)
    return symmetry_adapted_basis(T, tbl, ehom; semisimple = semisimple)
end

function symmetry_adapted_basis(
    T::Type,
    tbl::CharacterTable,
    ehom::InducedActionHomomorphism;
    semisimple = false,
)
    ψ = action_character(ehom, tbl)

    irr, multips = _constituents_decomposition(ψ, tbl)
    if T <: Real
        irr, multips = affordable_real(irr, multips)
        @debug "Decomposition into real character spaces:
        degrees:        $(join([lpad(d, 6) for d in degree.(irr)], ""))
        multiplicities: $(join([lpad(m, 6) for m in multips], ""))"
    end

    if semisimple || all(isone ∘ degree, irr)
        return _symmetry_adapted_basis(T, irr, multips, ehom)
    else
        RG = _group_algebra(parent(tbl))
        return _symmetry_adapted_basis(T, irr, multips, RG, ehom)
    end
end

function _constituents_decomposition(ψ::Character, tbl::CharacterTable)
    irr = irreducible_characters(tbl)
    degrees = degree.(irr)
    multips = multiplicities(ψ)

    @debug "Decomposition into character spaces:
    degrees:        $(join([lpad(d, 6) for d in degrees], ""))
    multiplicities: $(join([lpad(m, 6) for m in multips], ""))"

    @assert dot(multips, degrees) == degree(ψ)
    "Something went wrong: characters do not constitute a complete basis for action:
    $(dot(multips, degrees)) ≠ $(degree(ψ))"

    present_irreps = [i for (i, m) in pairs(multips) if m ≠ 0]
    return irr[present_irreps], multips[present_irreps]
end

function _symmetry_adapted_basis(
    T::Type,
    irr::AbstractVector{<:Character},
    multiplicities::AbstractVector{<:Integer},
    hom = nothing,
)
    res = map(zip(irr, multiplicities)) do (χ, m)
        Threads.@spawn begin
            χT = eltype(χ) == T ? χ : Character{T}(χ)
            # here we use algebra to compute the dimension of image;
            # direct summand is simple only if rk == m, i.e. deg == 1
            rk = m * degree(χ)
            image =
                isnothing(hom) ? image_basis(χT, rk) : image_basis(hom, χT, rk)
            DirectSummand(image, m, χ)
        end
    end
    return fetch.(res)
end

function _symmetry_adapted_basis(
    T::Type,
    irr::AbstractVector{<:Character},
    multips::AbstractVector{<:Integer},
    RG::StarAlgebra{<:Group},
    hom = nothing,
)
    mps, ranks = minimal_projection_system(irr, RG)
    @debug "ranks of projections obtained by mps:" degrees
    res = map(zip(mps, irr, multips, ranks)) do (µ, χ, m, r)
        Threads.@spawn begin
            µT = eltype(µ) == T ? µ : AlgebraElement{T}(µ)
            # here we use algebra to compute the dimension of image;
            # direct summand is simple only if rk == m, i.e. r == 1
            rk = m * r
            image =
                isnothing(hom) ? image_basis(µT, rk) : image_basis(hom, µT, rk)
            return DirectSummand(image, m, χ)
        end
    end
    direct_summands = fetch.(res)

    for (χ, ds) in zip(irr, direct_summands)
        if issimple(ds) &&
           (d = size(ds, 1)) !=
           (e = multiplicity(ds) * Int(dot(χ,χ)))
            throw(
                "The dimension of the projection doesn't match with simple summand multiplicity: $d ≠ $e",
            )
        end
    end

    return direct_summands
end
