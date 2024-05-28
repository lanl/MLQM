module Scalar

using Random: randn!

import Base: iterate, rand, read, write, zero
import Random: rand

using ..Geometries
using ..Lattices

import ..Lattices: Sampler, calibrate!

struct IsotropicLattice
    geom::CartesianGeometry
    N::Int
    m²::Float64
    λ::Float64
end

struct Cfg{lat}
    ϕ::Array{Float64,2}
end

function zero(::Type{Cfg{lat}})::Cfg{lat} where {lat}
    V = volume(lat.geom)
    ϕ = zeros(Float64, (lat.N,V))
    return Cfg{lat}(ϕ)
end

function rand(::Type{Cfg{lat}})::Cfg{lat} where {lat}
    cfg = zero(Cfg{lat})
    randn!(cfg.ϕ)
    return cfg
end

struct Heatbath{lat}
end

struct Wolff{lat}
end

struct Obs{lat}
end

function (obs::Obs{lat})(cfg::Cfg{lat})::Dict{String,Any} where {lat}
    r = Dict{String,Any}()
    r["action"] = action(obs,cfg)
    return r
end

function action(obs::Obs{lat}, cfg::Cfg{lat})::Float64 where {lat}
    # TODO
end

function write(io::IO, cfg::Cfg{lat}) where {lat}
    for i in lat.geom
        for n in 1:lat.N
            write(io, hton(cfg.ϕ[n,i]))
        end
    end
end

function read(io::IO, T::Type{Cfg{lat}})::Cfg{lat} where {lat}
    cfg = zero(T)
    for i in lat.geom
        for n in 1:lat.N
            c = read(io, Float64)
            cfg.ϕ[n,i] = ntoh(c)
        end
    end
    return cfg
end

end
