module FiniteFields

# taken from https://github.com/kalmarek/RamanujanGraphs.jl/blob/master/src/gf.jl

using Primes
export GF, int, characteristic
export generator, issqrt

struct GF{q} <: Number
    value::Int

    function GF{q}(n, check=true) where {q}
        if check
            @assert q > 1
            @assert isprime(q)
        end
        return new{q}(mod(n, q))
    end
end

GF{q}(n::GF{q}) where q = GF{q}(Int(n), false)

Base.Int(n::GF) = Int(n.value)
characteristic(::Type{GF{q}}) where {q} = q
characteristic(::GF{q}) where {q} = q

Base.:(==)(n::GF{q}, m::GF{q}) where {q} = Int(n) == Int(m)
# hash(RamanujanGraphs.GF) == 0x04fd9e474909f8bf
Base.hash(n::GF{q}, h::UInt) where {q} =
    xor(0x04fd9e474909f8bf, hash(q, hash(Int(n), h)))

Base.:+(n::GF{q}, m::GF{q}) where {q} = GF{q}(Int(n) + Int(m), false)
Base.:-(n::GF{q}, m::GF{q}) where {q} = GF{q}(Int(n) - Int(m), false)
Base.:*(n::GF{q}, m::GF{q}) where {q} = GF{q}(Int(n) * Int(m), false)
Base.:/(n::GF{q}, m::GF{q}) where {q} = n * inv(m)

Base.:-(n::GF{q}) where {q} = GF{q}(q - Int(n), false)
Base.inv(n::GF{q}) where {q} = GF{q}(invmod(Int(n), q), false)

function Base.:^(n::GF{q}, i::Integer) where {q}
    i < 0 && return inv(n)^-i
    return GF{q}(powermod(Int(n), i, q), false)
end

Base.zero(::Type{GF{q}}) where {q} = GF{q}(0, false)
Base.one(::Type{GF{q}}) where {q} = GF{q}(1, false)
Base.iszero(n::GF) = Int(n) == 0
Base.isone(n::GF) = Int(n) == 1

Base.promote_rule(::Type{GF{q}}, ::Type{I}) where {q,I<:Integer} = GF{q}
Base.promote_rule(::Type{GF{p}}, ::Type{GF{q}}) where {p,q} = throw(DomainError(
    (GF{p}, GF{q}),
    "Cannot perform arithmetic on elements from different fields",
))

Base.div(n::GF{q}, m::Integer) where q = n/GF{q}(m, false)
Base.div(m::Integer, n::GF{q}) where q = GF{q}(m, false)/n

# taken from ValidatedNumerics, under under the MIT "Expat" License:
# https://github.com/JuliaIntervals/ValidatedNumerics.jl/blob/master/LICENSE.md
function subscriptify(n::Integer)
    subscript_0 = Int(0x2080) # Char(0x2080) -> subscript 0
    return join((Char(subscript_0 + d) for d in reverse(digits(n))), "")
end

Base.show(io::IO, n::GF{q}) where {q} = print(io, "$(Int(n))" * subscriptify(q))
Base.isless(n::GF{q}, m::GF{q}) where {q} = isless(Int(n), Int(m))

function legendresymbol(n, q)
    iszero(mod(n, q)) && return zero(n)
    isone(powermod(n, (q - 1) ÷ 2, q)) && return one(n)
    return -one(n)
end

function generator(::Type{GF{q}}) where {q}
    q == 2 && return one(GF{2})
    for i = 2:q-1 # bruteforce loop
        g = GF{q}(i, false)
        any(isone, g^k for k = 2:q-2) && continue
        return g
    end
    return zero(GF{q}) # never hit, to keep compiler happy
end

Base.sqrt(n::GF{q}) where {q} = GF{q}(sqrtmod(Int(n), q), false)
issquare(n::GF{q}) where {q} = legendresymbol(Int(n), q) >= 0

function sqrtmod(n::Integer, q::Integer)
    l = legendresymbol(n, q)
    l == 0 && return zero(n)
    l == -1 && throw(DomainError(n, "$n is not a square modulo $q"))
    for i = 1:q # bruteforce loop
        y = powermod(i, 2, q)
        y == n && return oftype(n, i)
    end
    return zero(n) # never hit, to keep compiler happy
end

Base.iterate(::Type{GF{q}}, s = 0) where {q} =
    s >= q ? nothing : (GF{q}(s, false), s + 1)
Base.eltype(::Type{GF{q}}) where {q} = GF{q}
Base.size(gf::Type{<:GF}) = (characteristic(gf),)
end # module
