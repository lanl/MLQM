import Base: iterate, read, rand, write, zero

using LinearAlgebra: ⋅,I,mul!,norm
using Random: randn!

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
        M = zeros(ComplexF64, (N,N,3))
        s = new(V, M)
        resample(s, σ)
        return s
    end
end

function resample(s::UnitarySampler, σ::Float64)
    N = size(s.V)[1]
    K = size(s.V)[end]
    for k in 1:K
        # The generator is a random Hermitian matrix.
        @views randn!(s.M[:,:,1])
        for i in 1:N
            for j in 1:i
                m = σ * (s.M[i,j,1] + conj(s.M[j,i,1]))/2
                s.M[i,j,1] = m
                s.M[j,i,1] = conj(m)
            end
        end
        # Exponentiate.
        s.M[:,:,2] .= 0
        for n in 1:N
            s.M[n,n,2] = 1
        end
        @views s.V[:,:,k] .= s.M[:,:,2]
        for c in 1:8
            @views mul!(s.M[:,:,3], s.M[:,:,1], s.M[:,:,2])
            @views s.M[:,:,2] .= s.M[:,:,3]
            @views s.M[:,:,2] .*= 1im/c
            @views s.V[:,:,k] .+= s.M[:,:,2]
        end
        # Unitarize.
        @views unitarize!(s.V[:,:,k])
    end
end

function (s::UnitarySampler)(U::AbstractArray{ComplexF64,2})
    K = size(s.V)[end]
    k = rand(1:K)
    @views U .= s.V[:,:,k]
end

struct Lattice
    L::Int
    β::Int
    g::Float64
    N::Int
    d::Int

    function Lattice(L::Int, g::Float64; β::Int=L, N::Int=3, d::Int=4)
        new(L,β,g,N,d)
    end
end

volume(lat::Lattice)::Int = lat.β*lat.L^(lat.d-1)

function iterate(lat::Lattice, i::Int64=0)
    i < volume(lat) ? (i+1,i+1) : nothing
end

function step(lat::Lattice, i::Int, μ::Int; n::Int=1)::Int
    i -= 1
    v = lat.L^(μ-1)
    V = if μ == lat.d
        lat.β*lat.L^(μ-1)
    else
        lat.L^μ
    end
    return 1 + mod(i+n*v,V) + (i÷V)*V
end

function coordinate(lat::Lattice, i::Int, μ::Int)::Int
    @assert μ ≥ 1 && μ ≤ lat.d
    i -= 1
    if μ == lat.d
        return 1 + i÷(lat.L^(lat.d-1))
    else
        v = lat.L^(μ-1)
        return 1 + (i÷v)%lat.L
    end
end

struct Configuration{lat}
    U::Array{ComplexF64,4}
end

function zero(::Type{Configuration{lat}})::Configuration{lat} where {lat}
    V = volume(lat)
    U = zeros(ComplexF64, (lat.N,lat.N,lat.d,V))
    return Configuration{lat}(U)
end

function rand(::Type{Configuration{lat}})::Configuration{lat} where {lat}
    V = volume(lat)
    U = rand(ComplexF64, (lat.N,lat.N,lat.d,V))
    for i in lat
        for μ in 1:lat.d
            @views unitarize!(U[:,:,μ,i])
        end
    end
    return Configuration{lat}(U)
end

struct Heatbath{lat}
    sample!::UnitarySampler
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    function Heatbath{lat}() where {lat}
        sample! = UnitarySampler(lat.N, lat.g)
        A = zeros(ComplexF64, (lat.N,lat.N))
        B = zeros(ComplexF64, (lat.N,lat.N))
        new(sampler, A)
    end
end

function (hb::Heatbath{lat})(cfg::Configuration{lat}) where {lat}
    tot = 0
    acc = 0
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

function action(cfg::Configuration{lat})::Float64 where {lat}
    for i in lat
        for μ in 1:lat.d
            for ν in 1:(μ-1)
                # TODO
            end
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

