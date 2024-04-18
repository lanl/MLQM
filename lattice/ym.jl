import Base: iterate, read, rand, write, zero

using LinearAlgebra: I

struct UnitarySampler
    V::Array{ComplexF64,3}
    function UnitarySampler(N::Int)
        # TODO
    end
end

# TODO support a different temperature direction
struct Lattice
    L::Int
    g::Float64
    N::Int
    d::Int

    function Lattice(L::Int, g::Float64, N::Int=3, d::Int=4)
        new(L,g,N,d)
    end
end

volume(lat::Lattice)::Int = lat.L^lat.d

function iterate(lat::Lattice, i::Int64=0)
    i < volume(lat) ? (i+1,i+1) : nothing
end

function step(lat::Lattice, i::Int, μ::Int; n::Int=1)
    # TODO
end

struct Configuration{lat}
    U::Array{ComplexF64,4}
end

function zero(::Type{Configuration{lat}})::Configuration{lat} where {lat}
    V = lat.L^lat.d
    U = zeros(ComplexF64, (lat.N,lat.N,lat.d,V))
    return Configuration{lat}(U)
end

function rand(::Type{Configuration{lat}})::Configuration{lat} where {lat}
    V = lat.L^lat.d
    U = rand(ComplexF64, (lat.N,lat.N,lat.d,V))
    # TODO unitarize
    return Configuration{lat}(U)
end

function heatbath!(cfg::Configuration, lat::Lattice, i::Int, μ::Int)
    # The local action is -1/g² Re Tr A U. First we must compute the "staple" A.
    A = zeros(ComplexF64, (lat.N,lat.N))
    for ν in 1:lat.d
        if μ == ν
            continue
        end
        B = I
        # TODO
        A += B
    end
    # TODO
end

function sweep!(cfg::Configuration, lat::Lattice)
    for i in lat
        for μ in 1:lat.d
            heatbath!(cfg, lat, i, μ)
        end
    end
end

function write(io::IO, cfg::Configuration{lat}) where {lat}
    for i in lat
        for μ in 1:lat.d
            for a in 1:lat.N
                for b in 1:lat.N
                    write(io, hton(cfg.U[a,b,μ,i]))
                end
            end
        end
    end
end

function read(io::IO, T::Type{Configuration{lat}})::Configuration{lat} where {lat}
    cfg = zero(T)
    for i in lat
        for μ in 1:lat.d
            for a in 1:lat.N
                for b in 1:lat.N
                    c = read(io, ComplexF64)
                    cfg.U[a,b,μ,i] = ntoh(c)
                end
            end
        end
    end
    cfg
end

