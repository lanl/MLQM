module YangMills

import Base: iterate, read, rand, write, zero

using LinearAlgebra: ⋅,I,det,mul!,norm,tr,adjoint!
using Random: randn!

export UnitarySampler, SpecialUnitarySampler
export unitarize!, sunitarize!, resample!
export Configuration, Lattice, Observer, Heatbath, PseudoHeatbath
export action, plaquette
export gauge!, calibrate!

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

function sunitarize!(U::AbstractArray{ComplexF64,2})
    N = size(U)[1]
    unitarize!(U)
    d = det(U)
    U ./= d^(1/N)
end

mutable struct UnitarySampler
    V::Array{ComplexF64,3}
    M::Array{ComplexF64,3}
    σ::Float64
    function UnitarySampler(N::Int, σ::Float64; K::Int=200)
        if K < 2
            error("K≥2 required")
        end
        V = zeros(ComplexF64, (N,N,K))
        M = zeros(ComplexF64, (N,N,3))
        s = new(V, M, σ)
        resample!(s, σ)
        return s
    end
end

function resample!(s::UnitarySampler, σ::Float64)
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
    s.σ = σ
end

function (s::UnitarySampler)(U::AbstractArray{ComplexF64,2})
    K = size(s.V)[end]
    k = rand(1:K)
    @views U .= s.V[:,:,k]
end

mutable struct SpecialUnitarySampler
    V::Array{ComplexF64,3}
    M::Array{ComplexF64,3}
    σ::Float64
    function SpecialUnitarySampler(N::Int, σ::Float64; K::Int=200)
        if K < 2
            error("K≥2 required")
        end
        V = zeros(ComplexF64, (N,N,K))
        M = zeros(ComplexF64, (N,N,3))
        s = new(V, M, σ)
        resample!(s, σ)
        return s
    end
end

function resample!(s::SpecialUnitarySampler, σ::Float64)
    N = size(s.V)[1]
    K = size(s.V)[end]
    for k in 1:K
        # The generator is a random traceless Hermitian matrix.
        @views randn!(s.M[:,:,1])
        # Hermitize:
        for i in 1:N
            for j in 1:i
                m = σ * (s.M[i,j,1] + conj(s.M[j,i,1]))/2
                s.M[i,j,1] = m
                s.M[j,i,1] = conj(m)
            end
        end
        # Remove the trace:
        trace::Float64 = 0.
        for i in 1:N
            trace += s.M[i,i,1]
        end
        for i in 1:N
            s.M[i,i,1] -= trace/N
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
    s.σ = σ
end

function (s::SpecialUnitarySampler)(U::AbstractArray{ComplexF64,2})
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

# TODO replace all geometrical functions with versions in lattices.jl

volume(lat::Lattice)::Int = lat.β*lat.L^(lat.d-1)

function iterate(lat::Lattice, i::Int64=0)
    i < volume(lat) ? (i+1,i+1) : nothing
end

function trans(lat::Lattice, i::Int, μ::Int; n::Int=1)::Int
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

wilson_beta(lat::Lattice)::Float64 = 2*lat.N/(lat.g^2)

struct Configuration{lat}
    U::Array{ComplexF64,4}
end

function zero(::Type{Configuration{lat}})::Configuration{lat} where {lat}
    V = volume(lat)
    U = zeros(ComplexF64, (lat.N,lat.N,lat.d,V))
    for i in 1:V
        for μ in 1:lat.d
            for a in 1:lat.N
                U[a,a,μ,i] = 1
            end
        end
    end
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

# Perform a gauge transformation.
function gauge!(cfg::Configuration{lat}, i::Int, U::AbstractMatrix{ComplexF64}, V=nothing) where {lat}
    if isnothing(V)
        V = zeros(ComplexF64, (lat.N,lat.N))
    end
    for μ in 1:lat.d
        @views mul!(V, cfg.U[:,:,μ,i], U)
        cfg.U[:,:,μ,i] .= V
    end
    adjoint!(V, U)
    U .= V
    for μ in 1:lat.d
        j = trans(lat, i, μ, n=-1)
        @views mul!(V, U, cfg.U[:,:,μ,j])
        cfg.U[:,:,μ,j] .= V
    end
end

struct Heatbath{lat}
    sample!::SpecialUnitarySampler
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    C::Matrix{ComplexF64}
    D::Matrix{ComplexF64}
    function Heatbath{lat}() where {lat}
        sample! = SpecialUnitarySampler(lat.N, lat.g)
        A = zeros(ComplexF64, (lat.N,lat.N))
        B = zeros(ComplexF64, (lat.N,lat.N))
        C = zeros(ComplexF64, (lat.N,lat.N))
        D = zeros(ComplexF64, (lat.N,lat.N))
        new(sample!, A, B, C, D)
    end
end

function trmul(A::AbstractMatrix{ComplexF64}, B::AbstractMatrix{ComplexF64})::ComplexF64
    n, m = size(A)
    @assert size(B) == (m,n)
    r::ComplexF64 = 0.
    for i in 1:n
        for j in 1:m
            r += A[i,j] * B[j,i]
        end
    end
    return r
end

function calibrate!(hb!::Heatbath{lat}, cfg::Configuration{lat}) where {lat}
    ar = hb!(cfg)
    while ar < 0.3 || ar > 0.5
        if ar < 0.3
            hb!.sample!.σ *= 0.95
        end
        if ar > 0.5
            hb!.sample!.σ *= 1.05
        end
        resample!(hb!.sample!, hb!.sample!.σ)
        ar = hb!(cfg)
    end
end

function (hb::Heatbath{lat})(cfg::Configuration{lat})::Float64 where {lat}
    resample!(hb.sample!, hb.sample!.σ)
    tot = 0
    acc = 0
    @views for i′ in lat
        i = rand(1:volume(lat))
        μ = rand(1:lat.d)
        # The local action is -1/(2*g²) Re Tr A U. First compute the staple A.
        iμ = trans(lat, i, μ)
        hb.A .= 0
        for ν in 1:lat.d
            if μ == ν
                continue
            end
            iν = trans(lat, i, ν, n=1)
            iν′ = trans(lat, i, ν, n=-1)
            iμν′ = trans(lat, iμ, ν, n=-1)

            adjoint!(hb.D, cfg.U[:,:,μ,iν])
            mul!(hb.C, hb.D, cfg.U[:,:,ν,iμ])
            adjoint!(hb.D, cfg.U[:,:,ν,i])
            mul!(hb.B, hb.D, hb.C)
            hb.A .+= hb.B

            adjoint!(hb.B, cfg.U[:,:,ν,iμν′])
            adjoint!(hb.D, cfg.U[:,:,μ,iν′])
            mul!(hb.C, hb.D, hb.B)
            mul!(hb.B, cfg.U[:,:,ν,iν′], hb.C)
            hb.A .+= hb.B
        end
        # Current action.
        S = -2/(lat.g^2) * real(trmul(cfg.U[:,:,μ,i], hb.A))
        for p in 1:2*lat.N^2
            # Propose. hb.D stores the unitary; hb.C stores the new link.
            hb.sample!(hb.D)
            mul!(hb.C, hb.D, cfg.U[:,:,μ,i])
            S′ = -2/lat.g^2 * real(trmul(hb.C, hb.A))

            # Accept/reject
            tot += 1
            if rand() < exp(S-S′)
                acc += 1
                cfg.U[:,:,μ,i] .= hb.C
                S = S′
            end
        end
    end
    return acc / tot
end

struct PseudoHeatbath{lat}
    # TODO
end

function (phb::PseudoHeatbath{lat})(cfg::Configuration{lat})::Float64 where {lat}
    # TODO
end

struct Observer{lat}
    U::Matrix{ComplexF64}
    V::Matrix{ComplexF64}
    W::Matrix{ComplexF64}
    function Observer{lat}() where {lat}
        U = zeros(ComplexF64, (lat.N,lat.N))
        V = zeros(ComplexF64, (lat.N,lat.N))
        W = zeros(ComplexF64, (lat.N,lat.N))
        new(U,V,W)
    end
end

function (obs::Observer{lat})(cfg::Configuration{lat})::Dict{String,Any} where {lat}
    r = Dict{String,Any}()
    r["action"] = action(obs,cfg)
    r["polyakov"] = polyakov(obs,cfg)
    return r
end

function plaquette(obs::Observer{lat}, cfg::Configuration{lat}, i::Int, μ::Int, ν::Int)::ComplexF64 where {lat}
    iμ = trans(lat, i, μ)
    iν = trans(lat, i, ν)
    # Evaluate: U(i,ν)† U(iν,μ)† U(iμ,ν) U(i,μ)
    # This is: (U(iν,μ) U(i,ν))† U(iμ,ν) U(i,μ)
    @views mul!(obs.U, cfg.U[:,:,μ,iν], cfg.U[:,:,ν,i])
    adjoint!(obs.V, obs.U)
    @views mul!(obs.U, obs.V, cfg.U[:,:,ν,iμ])
    @views mul!(obs.V, obs.U, cfg.U[:,:,μ,i])
    return 1-tr(obs.V)/lat.N
end

function plaquette(obs::Observer{lat}, cfg::Configuration{lat})::Float64 where {lat}
    r::Float64 = 0.
    for i in lat
        for μ in 1:lat.d
            for ν in 1:(μ-1)
                r += real(plaquette(obs, cfg, i, μ, ν))
            end
        end
    end
    r *= 2 / (volume(lat) * lat.d * (lat.d-1))
    return r
end

function action(obs::Observer{lat}, cfg::Configuration{lat})::Float64 where {lat}
    S::Float64 = 0.
    @views for i in lat
        for μ in 1:lat.d
            for ν in 1:(μ-1)
                iμ = trans(lat, i, μ)
                iν = trans(lat, i, ν)
                # Evaluate: U(i,ν)† U(iν,μ)† U(iμ,ν) U(i,μ)
                # This is: (U(iν,μ) U(i,ν))† U(iμ,ν) U(i,μ)
                mul!(obs.U, cfg.U[:,:,μ,iν], cfg.U[:,:,ν,i])
                adjoint!(obs.V, obs.U)
                mul!(obs.U, obs.V, cfg.U[:,:,ν,iμ])
                mul!(obs.V, obs.U, cfg.U[:,:,μ,i])
                S += 2/(lat.g^2) * (lat.N - real(tr(obs.V)))
            end
        end
    end
    return S
end

function action(cfg::Configuration{lat})::Float64 where {lat}
    obs = Observer{lat}()
    action(obs, cfg)
end

function wilsonloop(obs::Observer{lat}, cfg::Configuration{lat}, i, μ, Lx, ν, Ly)::ComplexF64 where {lat}
    obs.U .= 0
    for a in 1:lat.N
        obs.U[a,a] = 1
    end
    for n in 1:Lx
        @views mul!(obs.V, cfg.U[:,:,μ,i], obs.U)
        obs.U .= obs.V
        i = trans(lat, i, μ)
    end
    for n in 1:Ly
        @views mul!(obs.V, cfg.U[:,:,ν,i], obs.U)
        obs.U .= obs.V
        i = trans(lat, i, ν)
    end
    for n in 1:Lx
        i = trans(lat, i, μ, n=-1)
        @views adjoint!(obs.W, cfg.U[:,:,μ,i])
        mul!(obs.V, obs.W, obs.U)
        obs.U .= obs.V
    end
    for n in 1:Ly
        i = trans(lat, i, ν, n=-1)
        @views adjoint!(obs.W, cfg.U[:,:,ν,i])
        mul!(obs.V, obs.W, obs.U)
        obs.U .= obs.V
    end
    return 1 - tr(obs.U)/lat.N
end

function wilsonloop(obs::Observer{lat}, cfg::Configuration{lat}, Lx::Int, Lt::Int)::ComplexF64 where {lat}
    r::ComplexF64 = 0.
    ν = lat.d
    for i in lat
        for μ in 1:lat.d-1
            r += wilsonloop(obs, cfg, i, μ, Lx, ν, Lt)
        end
    end
    return r/(volume(lat)*(lat.d-1))
end

function polyakov(obs::Observer{lat}, cfg::Configuration{lat}, i::Int)::ComplexF64 where {lat}
    obs.V .= 0
    for a in 1:lat.N
        obs.V[a,a] = 1.
    end
    for t in 1:lat.β
        @views mul!(obs.U, cfg.U[:,:,lat.d,i], obs.V)
        obs.V .= obs.U
        i = trans(lat, i, lat.d)
    end
    return tr(obs.U)
end

function polyakov(obs::Observer{lat}, cfg::Configuration{lat})::ComplexF64 where {lat}
    r::ComplexF64 = 0.
    for i in 1:(lat.L^(lat.d-1))
        r += polyakov(obs, cfg, i)
    end
    return r/(lat.L^(lat.d-1))
end

function quarkpotential(obs::Observer{lat}, cfg::Configuration{lat}, x::Int)::Float64 where {lat}
    z::Float64 = 0.
    for i in 1:(lat.L^(lat.d-1))
        for μ in 1:lat.d-1
            j = trans(lat, i, μ, n=x)
            P = polyakov(obs, cfg, i)
            P′ = polyakov(obs, cfg, j)
            z += real((P*conj(P′))/(lat.d-1)/(lat.L^(lat.d-1)))
        end
    end
    if z < 0
        z = 0
    end
    return -log(z)/lat.β
end

function quarkpotential!(v::Vector{Float64}, obs::Observer{lat}, cfg::Configuration{lat}) where {lat}
    for x in 1:lat.L
        v[i] = quarkpotential(obs, cfg, x)
    end
end

function quarkpotential(obs::Observer{lat}, cfg::Configuration{lat})::Vector{Float64} where {lat}
    v = zeros(Float64, lat.L)
    quarkpotential!(v, obs, cfg)
    return v
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
    return cfg
end

end

