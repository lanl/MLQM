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

mutable struct Heatbath{lat}
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
            hb!.σ *= 0.95
        end
        if ar > 0.5
            hb!.σ *= 1.05
        end
        ar = hb!(cfg)
    end
end

function (hb::Heatbath{lat})(cfg::Cfg{lat})::Float64 where {lat}
    C = 1
    acc = 0
    tot = 0
    for k in lat.geom
        i = rand(1:volume(lat.geom))
        for c in 1:C
            for n in 1:lat.N
                hb.ϕp[n] = cfg.ϕ[n,i] + hb.σ * randn()
            end
            # Compute local action.
            S = action(cfg, i)
            S′ = action(cfg, i, hb.ϕp)
            if rand() < exp(S-S′)
                cfg.ϕ[:,i] .= hb.ϕp
                acc += 1
            end
            tot += 1
        end
    end
    return acc / tot
end

mutable struct Wolff{lat}
    σ::Float64
    ϕp::Vector{Float64}
    r::Vector{Float64}
    b::Array{Bool,2}
    v::Vector{Bool}
    q::CircularDeque{Int}

    function Wolff{lat}(σ::Float64) where {lat}
        ϕp = zeros(Float64, lat.N)
        geom = lat.geom
        if geom.L ≤ 2 || geom.β ≤ 2
            error("Wolff will not be correct for L ≤ 2")
        end
        r = zeros(Float64, lat.N)
        b = zeros(Bool, (geom.d,volume(geom)))
        v = zeros(Bool, volume(geom))
        q = CircularDeque{Int}(volume(geom))
        return new{lat}(σ, ϕp, r, b, v, q)
    end
end

function Wolff{lat}() where {lat}
    return Wolff{lat}(1.)
end

function calibrate!(wolff!::Wolff{lat}, cfg::Cfg{lat}) where {lat}
    ar = wolff!(cfg)
    while ar < 0.3 || ar > 0.5
        if ar < 0.3
            wolff!.σ *= 0.95
        end
        if ar > 0.5
            wolff!.σ *= 1.05
        end
        ar = wolff!(cfg)
    end
end

function (wolff::Wolff{lat})(cfg::Cfg{lat})::Float64 where {lat}
    J = 1.
    # Cluster update. Pick the axis.
    randn!(wolff.r)
    nrm = 0.
    for n in 1:lat.N
        nrm += wolff.r[n]^2
    end
    nrm = √nrm
    wolff.r ./= nrm
    # Set bonds.
    for i in lat.geom
        for μ in 1:lat.geom.d
            j = translate(lat.geom, i, μ)
            ipi = 0.
            ipj = 0.
            for n in 1:lat.N
                ipi += cfg.ϕ[n,i] * wolff.r[n]
                ipj += cfg.ϕ[n,j] * wolff.r[n]
            end
            prob = 1 - exp(min(0,-2*J*ipi*ipj))
            wolff.b[μ,i] = rand() < prob
        end
    end

    wolff.v .= false
    for i in lat.geom
        if wolff.v[i]
            continue
        end
        σ = rand(Bool)
        # Flood-fill
        wolff.v[i] = true
        push!(wolff.q, i)
        while !isempty(wolff.q)
            k = pop!(wolff.q)
            if σ
                ip = 0.
                for n in 1:lat.N
                    ip += cfg.ϕ[n,k] * wolff.r[n]
                end
                for n in 1:lat.N
                    cfg.ϕ[n,k] -= 2*ip*wolff.r[n]
                end
            end
            for μ in 1:lat.geom.d
                j = translate(lat.geom, k, μ, 1)
                if wolff.b[μ,k] && !wolff.v[j]
                    wolff.v[j] = true
                    push!(wolff.q, j)
                end

                j = translate(lat.geom, k, μ, -1)
                if wolff.b[μ,j] && !wolff.v[j]
                    wolff.v[j] = true
                    push!(wolff.q, j)
                end
            end
        end
    end

    # Sweep
    C = 1
    acc = 0
    tot = 0
    for k in lat.geom
        i = rand(1:volume(lat.geom))
        for c in 1:C
            for n in 1:lat.N
                wolff.ϕp[n] = cfg.ϕ[n,i] + wolff.σ * randn()
            end
            # Compute local action.
            S = action(cfg, i)
            S′ = action(cfg, i, wolff.ϕp)
            if rand() < exp(S-S′)
                cfg.ϕ[:,i] .= wolff.ϕp
                acc += 1
            end
            tot += 1
        end
    end
    return acc/tot
end

function Sampler(lat::IsotropicLattice, algorithm::Symbol=:Heatbath)
    cfg = zero(Cfg{lat.geom})
    if algorithm == :Heatbath
        sample! = Heatbath{lat}()
    elseif algorithm == :Wolff
        sample! = Wolff{lat}()
    else
        error("Unknown algorithm requested")
    end
    return sample!, cfg
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
        ϕ² = 0.
        for n in 1:lat.N
            ϕ² += cfg.ϕ[n,i]^2
        end
        S += lat.m² * ϕ² / 2
        S += lat.λ * ϕ²^2 / 4
        for j in adjacent(lat.geom, i)
            if j < i
                for n in 1:lat.N
                    S += (cfg.ϕ[n,i] - cfg.ϕ[n,j])^2 / 2.
                end
            end
        end
    end
    return S
end

function action(cfg::Cfg{lat}, i::Int)::Float64 where {lat}
    return action(cfg, i, @view cfg.ϕ[:,i])
end

function action(cfg::Cfg{lat}, i::Int, ϕ′)::Float64 where {lat}
    S::Float64 = 0.
    ϕ² = 0.
    for n in 1:lat.N
        ϕ² += cfg.ϕ[n,i]^2
    end
    S += lat.m² * ϕ² / 2
    S += lat.λ * ϕ²^2 / 4
    for j in adjacent(lat.geom, i)
        for n in 1:lat.N
            S += (cfg.ϕ[n,i] - cfg.ϕ[n,j])^2 / 2.
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
