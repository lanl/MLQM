module Scalar

struct Lattice
    L::Int
    β::Int
    N::Int
    d::Int
    m²::Float64
    λ::Float64
end

volume(lat::Lattice)::Int = lat.β*lat.L^(lat.d-1)

function iterate(lat::Lattice, i::Int64=0)
    i < volume(lat) ? (i+1,i+1) : nothing
end

struct Configuration{lat}
    ϕ::Array{Float64,2}
end

function zero(::Type{Configuration{lat}})::Configuration{lat} where {lat}
    V = volume(lat)
    ϕ = zeros(Float64, (lat.N,V))
    return Configuration{lat}(ϕ)
end

function action(cfg::Configuration{lat})::Float64 where {lat}
    # TODO
end

end
