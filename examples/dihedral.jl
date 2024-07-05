import GroupsCore #GroupsCore API

struct DihedralGroup <: GroupsCore.Group
    n::Int
end

function Base.show(io::IO, G::DihedralGroup)
    return print(
        io,
        "Dihedral Group of order $(order(Int, G)) generated by `r` (rotation) and `s` (mirror symmetry)",
    )
end
struct DihedralElement <: GroupsCore.GroupElement
    n::Int
    reflection::Bool
    id::Int
end

function Base.show(io::IO, g::DihedralElement)
    if isone(g)
        print(io, "(id)")
    else
        if g.id ≥ 1
            print(io, 'r')
            if g.id ≥ 2
                print(io, '^')
                print(io, g.id ≤ g.n ÷ 2 ? g.id : -g.n + g.id)
            end
        end
        if g.reflection
            if g.id > 0
                print(io, '*')
            end
            print(io, 's')
        end
    end
end

Base.one(G::DihedralGroup) = DihedralElement(G.n, false, 0)

Base.eltype(::DihedralGroup) = DihedralElement
function Base.iterate(
    G::DihedralGroup,
    prev::DihedralElement = DihedralElement(G.n, false, -1),
)
    if prev.id + 1 >= G.n
        if prev.reflection
            return nothing
        else
            next = DihedralElement(G.n, true, 0)
        end
    else
        next = DihedralElement(G.n, prev.reflection, prev.id + 1)
    end
    return next, next
end
Base.IteratorSize(::Type{DihedralGroup}) = Base.HasLength()

GroupsCore.order(::Type{T}, G::DihedralGroup) where {T} = convert(T, 2G.n)
function GroupsCore.gens(G::DihedralGroup)
    return [DihedralElement(G.n, false, 1), DihedralElement(G.n, true, 0)]
end

Base.parent(g::DihedralElement) = DihedralGroup(g.n)
function Base.:(==)(g::DihedralElement, h::DihedralElement)
    return g.n == h.n && g.reflection == h.reflection && g.id == h.id
end
function Base.hash(g::DihedralElement, h::UInt)
    return hash((g.n, g.reflection, g.id, hash(DihedralElement)), h)
end

function Base.inv(el::DihedralElement)
    (el.reflection || iszero(el.id)) && return el
    return DihedralElement(el.n, false, el.n - el.id)
end
function Base.:*(a::DihedralElement, b::DihedralElement)
    a.n == b.n ||
        error("Cannot multiply elements from different Dihedral groups")
    id = mod(a.reflection ? a.id - b.id : a.id + b.id, a.n)
    return DihedralElement(a.n, a.reflection != b.reflection, id)
end

Base.copy(a::DihedralElement) = DihedralElement(a.n, a.reflection, a.id)

# optional functions:
function GroupsCore.order(T::Type, el::DihedralElement)
    el.reflection && return T(2)
    iszero(el.id) && return T(1)
    return T(div(el.n, gcd(el.n, el.id)))
end

# this is needed for using them in StarAlgebra:
SA.comparable(::Type{DihedralElement}) = (a, b) -> hash(a) < hash(b)
