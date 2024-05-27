module Ising

import Base: iterate, rand, read, write, zero

using ..Geometries
using ..Lattices

abstract type IsingLattice <: Lattice end

struct IsotropicLattice <: IsingLattice
    geom::CartesianGeometry
    J::Float64
end

struct Cfg{geom} <: Configuration{IsingLattice}
    σ::Vector{Bool}

    function Cfg{geom}() where {geom}
        V = volume(geom)
        σ = zeros(Bool, (v))
        return new{geom}(σ)
    end
end

function zero(::Type{Cfg{geom}})::Cfg{geom} where {geom}
    return Cfg{geom}(σ)
end

function rand(T::Type{Cfg{lat}})::Cfg{lat} where {lat}
    cfg = zero(T)
    for i in lat
        cfg.σ[i] = rand(Bool)
    end
    return cfg
end

struct Heatbath{lat} <: Sampler{lat}
end

function calibrate!(hb!::Heatbath{lat}, cfg::Cfg{lat}) where {lat}
end

function (hb::Heatbath{lat})(cfg::Cfg{lat}) where {lat}
    geom = hb.geom
    J = lat.J
    for i′ in lat
        i = rand(1:volume(geom))
        ntrue::Int = 0
        for j in adjacent(lat, i)
            ntrue += cfg.σ[j]
        end
        nfalse = 2^lat.d - ntrue
        S = J * (ntrue - nfalse)
        ptrue = exp(-S) / (exp(S) + exp(-S))
        if rand() < ptrue
            cfg.σ[i] = true
        else
            cfg.σ[i] = false
        end
    end
end

struct SwendsenWang{lat}
end

function calibrate!(sw!::SwendsenWang{lat}, cfg::Cfg{lat}) where {lat}
end

function (sw::SwendsenWang{lat})(cfg::Cfg{lat}) where {lat}
end

struct Obs{Lat}
end

function (obs::Obs{lat})(cfg::Cfg{lat})::Dict{String,Any} where {lat}
    r = Dict{String,Any}()
    r["action"] = action(obs,cfg)
    return r
end

function action(obs::Obs{lat}, cfg::Cfg{lat})::Float64 where {lat}
    S::Float64 = 0.
    for i in lat
    end
    # TODO
    return S
end

function write(io::IO, cfg::Cfg{lat}) where {lat}
    for i in lat
        write(io, hton(cfg.σ[i]))
    end
end

function read(io::IO, T::Type{Cfg{lat}})::Cfg{lat} where {lat}
    cfg = zero(T)
    for i in lat
        σ = read(io, Bool)
        cfg.σ[i] = ntoh(σ)
    end
    return cfg
end

end
