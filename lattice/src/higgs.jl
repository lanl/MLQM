module Higgs

import Base: iterate, rand, read, write, zero

using ..Geometries
using ..Lattices

import ..Lattices: calibrate!

struct Lattice
    geom::CartesianGeometry
    Nc::Int
    Nf::Int
end

function iterate(lat::Lattice, i::Int64=0)
    i < volume(lat) ? (i+1,i+1) : nothing
end

struct Cfg{lat} <: Configuration
    U::Array{ComplexF64,4}
    ϕ::Array{Float64,2}
end

function zero(::Type{Cfg{lat}})::Cfg{lat} where {lat}
    V = volume(lat)
    U = zeros(ComplexF64, (lat.N,lat.N,lat.d,V))
    for i in 1:V
        for μ in 1:lat.d
            for a in 1:lat.Nc
                U[a,a,μ,i] = 1
            end
        end
    end
    ϕ = zeros(Float64, (lat.N,V))
    return Cfg{lat}(U, ϕ)
end

struct Heatbath{lat}
end

function calibrate!(hb!::Heatbath{lat}, cfg::Cfg{lat}) where {lat}
    # TODO
end

function (hb::Heatbath{lat})(cfg::Cfg{lat})::Float64 where {lat}
    tot = 0
    acc = 0
end

struct Observer{lat}
end

function write(io::IO, cfg::Cfg{lat}) where {lat}
    # TODO
end

function read(io::IO, T::Type{Cfg{lat}})::Cfg{lat} where {lat}
    # TODO
end

end

