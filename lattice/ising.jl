module Ising

import Base: iterate, rand, read, write, zero

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
    σ = zeros(Bool, (V))
    return Configuration{lat}(σ)
end

function rand(T::Type{Configuration{lat}})::Configuration{lat} where {lat}
    cfg = zero(T)
    for i in lat
        cfg.σ[i] = rand(Bool)
    end
    return cfg
end

struct Observer{lat}
end

function (obs::Observer{lat})(cfg::Configuration{lat})::Dict{String,Any} where {lat}
    r = Dict{String,Any}()
    r["action"] = action(obs,cfg)
    return r
end

function action(obs::Observer{lat}, cfg::Configuration{lat})::Float64 where {lat}
    # TODO
end

function write(io::IO, cfg::Configuration{lat}) where {lat}
    for i in lat
        write(io, hton(cfg.σ[i]))
    end
end

function read(io::IO, T::Type{Configuration{lat}})::Configuration{lat} where {lat}
    cfg = zero(T)
    for i in lat
        σ = read(io, Bool)
        cfg.σ[i] = ntoh(σ)
    end
    return cfg
end

end
