## characters defined by actions/homomorphisms
function nfixedpoints(p::PermutationGroups.AbstractPermutation, deg = degree(p))
    return count(i -> i^p == i, Base.OneTo(deg))
end

function _action_class_fun(
    conjugacy_cls::AbstractVector{CCl},
) where {CCl<:AbstractOrbit{<:PermutationGroups.AbstractPermutation}}
    deg = mapreduce(cc -> maximum(degree, cc), max, conjugacy_cls)
    vals = map(conjugacy_cls) do cc
        repr = first(cc)
        return nfixedpoints(repr, deg)
    end
    # in general:
    # vals = [tr(matrix_representative(first(cc))) for cc in conjugacy_cls]
    return Characters.ClassFunction(vals, conjugacy_cls)
end

function _action_class_fun(
    conjugacy_cls::AbstractVector{CCl},
) where {CCl<:AbstractOrbit{<:AbstractMatrix}}
    vals = [tr(first(cc)) for cc in conjugacy_cls]
    return Characters.ClassFunction(vals, conjugacy_cls)
end

function _action_class_fun(
    hom::InducedActionHomomorphism{<:ByPermutations},
    conjugacy_cls,
)
    deg = degree(hom)
    vals = map(conjugacy_cls) do cc
        repr = induce(hom, first(cc))
        return nfixedpoints(repr, deg)
    end
    # in general:
    # vals = [tr(matrix_representative(first(cc))) for cc in conjugacy_cls]
    return Characters.ClassFunction(vals, conjugacy_cls)
end

function _action_class_fun(
    hom::InducedActionHomomorphism{<:ByLinearTransformation},
    conjugacy_cls,
)
    vals = [tr(induce(hom, first(cc))) for cc in conjugacy_cls]
    return Characters.ClassFunction(vals, conjugacy_cls)
end

"""
    action_character([::Type{T}, ]conjugacy_cls, tbl::CharacterTable)
Return the character of the representation given by the elements in the
conjugacy classes `conjugacy_cls`.

This corresponds to the classical definition of characters as a traces of the
representation matrices.
"""
function action_character(conjugacy_clss, tbl::CharacterTable)
    return action_character(eltype(tbl), conjugacy_clss, tbl)
end

function action_character(
    ::Type{T},
    conjugacy_clss,
    tbl::CharacterTable,
) where {T}
    ac_char = _action_class_fun(conjugacy_clss)
    constituents = Int[dot(ac_char, χ) for χ in irreducible_characters(tbl)]
    return Character{T}(tbl, constituents)
end

function action_character(
    ::Type{T},
    conjugacy_cls,
    tbl::CharacterTable,
) where {T<:Union{AbstractFloat,ComplexF64}}
    ac_char = _action_class_fun(conjugacy_cls)

    constituents = [dot(ac_char, χ) for χ in irreducible_characters(tbl)]
    all(constituents) do c
        ac = abs(c)
        return abs(ac - round(ac)) < eps(real(T)) * length(conjugacy_cls)
    end

    return Character{T}(tbl, round.(Int, abs.(constituents)))
end

"""
    action_character([::Type{T}, ]hom::InducedActionHomomorphism, tbl::CharacterTable)
Return the character of the representation given by the images under `hom` of
elements in `conjugacy_classes(tbl)`.

This corresponds to the classical definition of characters as a traces of the
representation matrices.
"""
function action_character(hom::InducedActionHomomorphism, tbl::CharacterTable)
    return action_character(eltype(tbl), hom, tbl)
end

function action_character(
    ::Type{T},
    hom::InducedActionHomomorphism,
    tbl::CharacterTable,
) where {T}
    ac_char = _action_class_fun(hom, conjugacy_classes(tbl))
    vals = [dot(ac_char, χ) for χ in irreducible_characters(T, tbl)]
    constituents = convert(Vector{Int}, vals)
    return Character{T}(tbl, constituents)
end
