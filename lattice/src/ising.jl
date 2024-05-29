module Ising

import Base: iterate, rand, read, write, zero

using ..Geometries
using ..Lattices

import ..Lattices: Sampler, calibrate!

abstract type IsingLattice <: Lattice end

struct IsotropicLattice <: IsingLattice
    geom::CartesianGeometry
    J::Float64
end

struct Cfg{geom} <: Configuration
    σ::Vector{Bool}

    function Cfg{geom}() where {geom}
        V = volume(geom)
        σ = zeros(Bool, V)
        return new{geom}(σ)
    end
end

function zero(::Type{Cfg{geom}})::Cfg{geom} where {geom}
    return Cfg{geom}()
end

function rand(T::Type{Cfg{geom}})::Cfg{geom} where {geom}
    cfg = zero(T)
    for i in geom
        cfg.σ[i] = rand(Bool)
    end
    return cfg
end

struct Heatbath{lat} <: Sampler
end

function calibrate!(hb!::Heatbath{lat}, cfg::Cfg{geom}) where {lat,geom}
    @assert lat.geom == geom
end

function (hb::Heatbath{lat})(cfg::Cfg{geom}) where {lat,geom}
    @assert lat.geom == geom
    J = lat.J
    for i′ in geom
        i = rand(1:volume(geom))
        ntrue::Int = 0
        for j in adjacent(geom, i)
            ntrue += cfg.σ[j]
        end
        nfalse = 2^geom.d - ntrue
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

function Sampler(lat::IsotropicLattice, algorithm=:Heatbath)
    cfg = zero(Cfg{lat.geom})
    if algorithm == :Heatbath
        sample! = Heatbath{lat}()
    elseif algorithm == :SwendsenWang
        sample! = SwendsenWang{lat}()
    else
        error("Unknown algorithm requested")
    end
    return sample!, cfg
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

function action(obs::Obs{lat}, cfg::Cfg{geom})::Float64 where {lat,geom}
    S::Float64 = 0.
    for i in geom
        for j in adjacent(geom, i)
            # Only include each pair once.
            if j < i
                if cfg.σ[j] == cfg.σ[i]
                    S -= lat.J
                else
                    S += lat.J
                end
            end
        end
    end
    return S
end

function write(io::IO, cfg::Cfg{geom}) where {geom}
    for i in geom
        write(io, hton(cfg.σ[i]))
    end
end

function read(io::IO, T::Type{Cfg{geom}})::Cfg{geom} where {geom}
    cfg = zero(T)
    for i in geom
        σ = read(io, Bool)
        cfg.σ[i] = ntoh(σ)
    end
    return cfg
end

end
