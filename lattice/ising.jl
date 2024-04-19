module Ising

struct Lattice
    L::Int
    β::Int
    J::Float64
    d::Int
end

volume(lat::Lattice)::Int = lat.β*lat.L^(lat.d-1)

function iterate(lat::Lattice, i::Int64=0)
    i < volume(lat) ? (i+1,i+1) : nothing
end

struct Configuration{lat}
    σ::Vector{Bool}
end

function zero(::Type{Configuration{lat}})::Configuration{lat} where {lat}
    V = volume(lat)
    σ = zeros(Bool, (lat.N,V))
    return Configuration{lat}(σ)
end

function action(cfg::Configuration{lat})::Float64 where {lat}
    # TODO
end

end
