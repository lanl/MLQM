module Ising

import Base: iterate, rand, read, write, zero

using ..Lattices
import ..Lattices: Configuration, Observer, Sampler, calibrate!

export IsingLattice

struct Lattice <: CartesianGeometry
    d::Int
    L::Int
    β::Int
    J::Float64
end
IsingLattice = Lattice

struct Cfg{lat}
    σ::Vector{Bool}
end

function zero(::Type{Cfg{lat}})::Cfg{lat} where {lat}
    V = volume(lat)
    σ = zeros(Bool, (V))
    return Cfg{lat}(σ)
end

function configuration(lat::Lattice)::Cfg{lat}
    zero(Cfg{lat})
end

function rand(T::Type{Cfg{lat}})::Cfg{lat} where {lat}
    cfg = zero(T)
    for i in lat
        cfg.σ[i] = rand(Bool)
    end
    return cfg
end

struct Heatbath{lat}
end

function calibrate!(hb!::Heatbath{lat}, cfg::Cfg{lat}) where {lat}
end

function (hb::Heatbath{lat})(cfg::Cfg{lat}) where {lat}
    J = lat.J
    for i′ in lat
        i = rand(1:volume(lat))
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

struct Obs{lat}
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

Configuration(lat::Lattice) = Cfg{lat}
Observer(lat::Lattice) = Obs{lat}
Sampler(lat::Lattice) = Heatbath{lat}()

end
