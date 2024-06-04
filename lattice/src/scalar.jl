module Scalar

using DataStructures: CircularDeque
using Random: randn!

import Base: iterate, rand, read, write, zero
import Random: rand

using ..Geometries
using ..Lattices

import ..Lattices: Sampler, Observer, calibrate!, CfgType

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
    σ::Float64
    ϕp::Vector{Float64}

    function Heatbath{lat}(σ::Float64) where {lat}
        ϕp = zeros(Float64, lat.N)
        return new{lat}(σ, ϕp)
    end
end

function Heatbath{lat}() where {lat}
    return Heatbath{lat}(1.)
end

function calibrate!(hb!::Heatbath{lat}, cfg::Cfg{lat}) where {lat}
    ar = hb!(cfg)
    while ar < 0.3 || ar > 0.5
        if ar < 0.3
            hb!.sample!.σ *= 0.95
        end
        if ar > 0.5
            hb!.sample!.σ *= 1.05
        end
        ar = hb!(cfg)
    end
end

function (hb::Heatbath{lat})(cfg::Cfg{lat})::Float64 where {lat}
    acc = 0
    tot = 0
    for k in lat.geom
        i = rand(1:volume(lat.geom))
        for n in 1:1
            # TODO
        end
    end
    return acc / tot
end

struct Wolff{lat}
    σ::Float64
    ϕp::Vector{Float64}
    b::Array{Bool,2}
    v::Vector{Bool}
    q::CircularDeque{Int}

    function Wolff{lat}(σ::Float64) where {lat}
        ϕp = zeros(Float64, lat.N)
        geom = lat.geom
        if geom.L ≤ 2 || geom.β ≤ 2
            error("Wolff will not be correct for L ≤ 2")
        end
        b = zeros(Bool, (geom.d,volume(geom)))
        v = zeros(Bool, volume(geom))
        q = CircularDeque{Int}(volume(geom))
        return new{lat}(σ, ϕp, b, v, q)
    end
end

function Wolff{lat}() where {lat}
    return Wolff{lat}(1.)
end

function calibrate!(wolff!::Wolff{lat}, cfg::Cfg{lat}) where {lat}
    ar = wolff!(cfg)
    while ar < 0.3 || ar > 0.5
        if ar < 0.3
            wolff!.sample!.σ *= 0.95
        end
        if ar > 0.5
            wolff!.sample!.σ *= 1.05
        end
        ar = wolff!(cfg)
    end
end

function (wolff::Wolff{lat})(cfg::Cfg{lat})::Float64 where {lat}
    # Cluster update
    # TODO

    # Sweep
    # TODO
    acc = 0
    tot = 0
    return acc/tot
end

struct Obs{lat}
end

function (obs::Obs{lat})(cfg::Cfg{lat})::Dict{String,Any} where {lat}
    r = Dict{String,Any}()
    r["action"] = action(obs,cfg)
    return r
end

function action(obs::Obs{lat}, cfg::Cfg{lat})::Float64 where {lat}
    S::Float64 = 0.
    for i in lat.geom
        S += lat.m² * cfg.ϕ[i]^2 / 2
        S += lat.λ * cfg.ϕ[i]^4 / 4
        for j in adjacent(lat.geom, i)
            if j < i
                S += (cfg.ϕ[i] - cfg.ϕ[j])^2 / 2.
            end
        end
    end
    return S
end

function Observer(lat::IsotropicLattice)
    return Obs{lat}()
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
