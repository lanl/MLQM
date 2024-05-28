module Geometries

import Base: iterate

export CartesianGeometry
export volume, translate, coordinate, adjacent

abstract type Geometry end

struct CartesianGeometry <: Geometry
    d::Int
    L::Int
    β::Int
end

function CartesianGeometry(d::Int, L::Int)::CartesianGeometry
    return CartesianGeometry(d, L, L)
end

function volume(g::CartesianGeometry)::Int
    return g.β*g.L^(g.d-1)
end

function iterate(g::CartesianGeometry, i::Int64=0)
    i < volume(g) ? (i+1,i+1) : nothing
end

function translate(g::CartesianGeometry, i::Int, μ::Int; n::Int=1)::Int
    i -= 1
    v = g.L^(μ-1)
    V = if μ == g.d
        g.β*g.L^(μ-1)
    else
        g.L^μ
    end
    return 1 + mod(i+n*v,V) + (i÷V)*V
end

function coordinate(g::CartesianGeometry, i::Int, μ::Int)::Int
    @assert μ ≥ 1 && μ ≤ g.d
    i -= 1
    if μ == g.d
        return 1 + i÷(g.L^(g.d-1))
    else
        v = g.L^(μ-1)
        return 1 + (i÷v)%g.L
    end
end

struct CartesianAdjacency{g}
    i::Int
end

function adjacent(g, i::Int)::CartesianAdjacency{g}
    return CartesianAdjacency{g}(i)
end

function iterate(adj::CartesianAdjacency{g}, s::Tuple{Int64,Bool}=(0,true)) where {g}
    μ, δ = s
    if δ
        μ += 1
        δ = false
    else
        δ = true
    end
    if μ > g.d
        return nothing
    else
        j = translate(g, adj.i, μ, n=(δ ? 1 : -1))
        return (j, (μ,δ))
    end
end

function adjacent(f, g::CartesianGeometry, i::Int)
    for j in adjacent(g, i)
        f(j)
    end
end

end
