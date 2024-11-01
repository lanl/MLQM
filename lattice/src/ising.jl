module Ising

using DataStructures: CircularDeque

import Base: iterate, rand, read, write, zero

using ..Geometries
using ..Lattices

import ..Lattices: Sampler, Observer, calibrate!, CfgType

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

CfgType(lat::IsotropicLattice) = Cfg{lat.geom}

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
        nfalse::Int = 0
        for j in adjacent(geom, i)
            ntrue += cfg.σ[j]
            nfalse += !(cfg.σ[j])
        end
        S = J * (ntrue - nfalse)
        ptrue = exp(S) / (exp(S) + exp(-S))
        if rand() < ptrue
            cfg.σ[i] = true
        else
            cfg.σ[i] = false
        end
    end
end

struct SwendsenWang{lat}
    b::Array{Bool,2}
    v::Vector{Bool}
    q::CircularDeque{Int}

    function SwendsenWang{lat}() where {lat}
        geom = lat.geom
        if geom.L ≤ 2 || geom.β ≤ 2
            error("Swedsen-Wang will not be correct for L ≤ 2")
        end
        b = zeros(Bool, (geom.d,volume(geom)))
        v = zeros(Bool, volume(geom))
        q = CircularDeque{Int}(volume(geom))
        new(b,v,q)
    end
end

function calibrate!(sw!::SwendsenWang{lat}, cfg::Cfg{geom}) where {lat,geom}
end

function (sw::SwendsenWang{lat})(cfg::Cfg{geom}) where {lat,geom}
    # Set all the bonds.
    prob = exp(-2*lat.J)
    for i in geom
        for μ in 1:geom.d
            j = translate(geom, i, μ)
            sw.b[μ,i] = cfg.σ[i] == cfg.σ[j] && rand() > prob
        end
    end

    sw.v .= false
    for i in geom
        if sw.v[i]
            continue
        end
        # The new state for this cluster:
        σ = rand(Bool)
        # Flood-fill
        sw.v[i] = true
        push!(sw.q, i)
        while !isempty(sw.q)
            k = pop!(sw.q)
            cfg.σ[k] = σ
            for μ in 1:geom.d
                j = translate(geom, k, μ, 1)
                if sw.b[μ,k] && !sw.v[j]
                    sw.v[j] = true
                    push!(sw.q, j)
                end

                j = translate(geom, k, μ, -1)
                if sw.b[μ,j] && !sw.v[j]
                    sw.v[j] = true
                    push!(sw.q, j)
                end
            end
        end
    end
end

function Sampler(lat::IsotropicLattice, algorithm::Symbol=:Heatbath)
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

struct Obs{lat}
end

function (obs::Obs{lat})(cfg::Cfg{geom})::Dict{String,Any} where {lat,geom}
    r = Dict{String,Any}()
    r["action"] = action(obs,cfg)
    r["susceptibility"] = susceptibility(obs,cfg)
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

function susceptibility(obs::Obs{lat}, cfg::Cfg{geom})::Float64 where {lat,geom}
    s::Float64 = 0.
    for i in geom
        for j in adjacent(geom, i)
            # Only include each pair once.
            if j < i
                if cfg.σ[j] == cfg.σ[i]
                    s -= 1/volume(geom)
                else
                    s += 1/volume(geom)
                end
            end
        end
    end
    return s
end

function Observer(lat::IsotropicLattice)
    return Obs{lat}()
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
