module Lattices

import Base: iterate

export Configuration, Observer, Sampler
export calibrate!

export CartesianGeometry
export volume, translate, coordinate, adjacent

function Configuration end

function Observer end

function Sampler end
function calibrate! end

abstract type CartesianGeometry end

function volume(lat::L)::Int where {L<:CartesianGeometry}
    return lat.β*lat.L^(lat.d-1)
end

function iterate(lat::L, i::Int64=0) where {L<:CartesianGeometry}
    i < volume(lat) ? (i+1,i+1) : nothing
end

function translate(lat, i::Int, μ::Int; n::Int=1)::Int
    i -= 1
    v = lat.L^(μ-1)
    V = if μ == lat.d
        lat.β*lat.L^(μ-1)
    else
        lat.L^μ
    end
    return 1 + mod(i+n*v,V) + (i÷V)*V
end

function coordinate(lat, i::Int, μ::Int)::Int
    @assert μ ≥ 1 && μ ≤ lat.d
    i -= 1
    if μ == lat.d
        return 1 + i÷(lat.L^(lat.d-1))
    else
        v = lat.L^(μ-1)
        return 1 + (i÷v)%lat.L
    end
end

struct Adjacency{lat}
    i::Int
end

function adjacent(lat, i::Int)::Adjacency{lat}
    return Adjacency{lat}(i)
end

function iterate(adj::Adjacency{lat}, s::Tuple{Int64,Bool}=(0,true)) where {lat}
    μ, δ = s
    if δ
        μ += 1
        δ = false
    else
        δ = true
    end
    if μ > lat.d
        return nothing
    else
        j = translate(lat, adj.i, μ, n=(δ ? 1 : -1))
        return (j, (μ,δ))
    end
end

function adjacent(f, lat::L, i::Int) where {L}
    for j in adjacent(lat, i)
        f(j)
    end
end

end
