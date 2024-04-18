import Base: iterate, read, rand, write, zero

using LinearAlgebra: ⋅,I,norm

function unitarize!(U::AbstractArray{ComplexF64,2})
    N = size(U)[1]
    for n in 1:N
        for m in 1:n-1
            # Orthogonalize
            @views ip = U[:,m] ⋅ U[:,n]
            for k in 1:N
                U[k,n] -= ip*U[k,m]
            end
        end
        # Normalize
        nrm = norm(@view U[:,n])
        @views U[:,n] ./= nrm
    end
end

struct UnitarySampler
    V::Array{ComplexF64,3}
    M::Array{ComplexF64,3}
    function UnitarySampler(N::Int, σ::Float64; K::Int=100)
        if K < 2
            error("K≥2 required")
        end
        V = zeros(ComplexF64, (N,N,K))
        M = zeros(ComplexF64, (N,N,2))
        s = new(V, M)
        resample(s, σ)
        return s
    end
end

function resample(s::UnitarySampler, σ::Float64)
    K = size(s.V)[end]
    for k in 1:K
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
    for i in lat
        for μ in 1:lat.d
            @views unitarize!(U[:,:,μ,i])
        end
    end
    return Configuration{lat}(U)
end

struct Heatbath{lat}
    sampler::UnitarySampler
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    function Heatbath{lat}() where {lat}
        sampler = UnitarySampler(lat.N, lat.g)
        A = zeros(ComplexF64, (lat.N,lat.N))
        B = zeros(ComplexF64, (lat.N,lat.N))
        new(sampler, A)
    end
end

function (hb::Heatbath{lat})(cfg::Configuration{lat}) where {lat}
    for i in lat
        for μ in 1:lat.d
            # The local action is -1/g² Re Tr A U. First compute the staple A.
            hb.A .= 0
            for ν in 1:lat.d
                if μ == ν
                    continue
                end
                # TODO compute sample
            end
            # TODO Repeatedly propose and acc/rej
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

