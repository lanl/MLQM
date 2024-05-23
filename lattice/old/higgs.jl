module Higgs

import Base: iterate, rand, read, write, zero

struct Lattice
    L::Int
    β::Int
    d::Int
    Nc::Int
    Nf::Int
end

volume(lat::Lattice)::Int = lat.β*lat.L^(lat.d-1)

function iterate(lat::Lattice, i::Int64=0)
    i < volume(lat) ? (i+1,i+1) : nothing
end

struct Configuration{lat}
    U::Array{ComplexF64,4}
    ϕ::Array{Float64,2}
end

function zero(::Type{Configuration{lat}})::Configuration{lat} where {lat}
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
    return Configuration{lat}(U, ϕ)
end

struct Heatbath{lat}
end

function calibrate!(hb!::Heatbath{lat}, cfg::Configuration{lat}) where {lat}
    # TODO
end

function (hb::Heatbath{lat})(cfg::Configuration{lat})::Float64 where {lat}
    tot = 0
    acc = 0
end

struct Observer{lat}
end

function write(io::IO, cfg::Configuration{lat}) where {lat}
    # TODO
end

function read(io::IO, T::Type{Configuration{lat}})::Configuration{lat} where {lat}
    # TODO
end

end

