module YangMills

import Base: iterate, read, rand, write, zero

using LinearAlgebra: ⋅,I,det,mul!,norm,tr,adjoint!
using Random: randn!

using ..Geometries
using ..Lattices

import ..Lattices: Sampler, Observer, calibrate!, CfgType

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

struct WilsonLattice
    geom::CartesianGeometry
    g::Float64
    N::Int

end

function WilsonLattice(L::Int, g::Float64; β::Int=L, N::Int=3, d::Int=4)
    WilsonLattice(CartesianGeometry(d,L,β),g,N)
end

wilson_beta(lat::WilsonLattice)::Float64 = 2*lat.N/(lat.g^2)

struct Cfg{lat}
    U::Array{ComplexF64,4}
end

function zero(::Type{Cfg{lat}})::Cfg{lat} where {lat}
    V = volume(lat.geom)
    U = zeros(ComplexF64, (lat.N,lat.N,lat.geom.d,V))
    for i in lat.geom
        for μ in 1:lat.geom.d
            for a in 1:lat.N
                U[a,a,μ,i] = 1
            end
        end
    end
    return Cfg{lat}(U)
end

function rand(::Type{Cfg{lat}})::Cfg{lat} where {lat}
    V = volume(lat.geom)
    U = rand(ComplexF64, (lat.N,lat.N,lat.geom.d,V))
    for i in lat.geom
        for μ in 1:lat.geom.d
            @views unitarize!(U[:,:,μ,i])
        end
    end
    return Cfg{lat}(U)
end

CfgType(lat::WilsonLattice) = Cfg{lat}

# Perform a gauge transformation.
function gauge!(cfg::Cfg{lat}, i::Int, U::AbstractMatrix{ComplexF64}, V=nothing) where {lat}
    if isnothing(V)
        V = zeros(ComplexF64, (lat.N,lat.N))
    end
    for μ in 1:lat.geom.d
        @views mul!(V, cfg.U[:,:,μ,i], U)
        cfg.U[:,:,μ,i] .= V
    end
    adjoint!(V, U)
    U .= V
    for μ in 1:lat.geom.d
        j = translate(lat.geom, i, μ, -1)
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

function calibrate!(hb!::Heatbath{lat}, cfg::Cfg{lat}) where {lat}
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

function (hb::Heatbath{lat})(cfg::Cfg{lat})::Float64 where {lat}
    resample!(hb.sample!, hb.sample!.σ)
    tot = 0
    acc = 0
    @views for i′ in lat.geom
        i = rand(1:volume(lat.geom))
        μ = rand(1:lat.geom.d)
        # The local action is -1/(2*g²) Re Tr A U. First compute the staple A.
        iμ = translate(lat.geom, i, μ)
        hb.A .= 0
        for ν in 1:lat.geom.d
            if μ == ν
                continue
            end
            iν = translate(lat.geom, i, ν, 1)
            iν′ = translate(lat.geom, i, ν, -1)
            iμν′ = translate(lat.geom, iμ, ν, -1)

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

function (phb::PseudoHeatbath{lat})(cfg::Cfg{lat})::Float64 where {lat}
    # TODO
end

function Sampler(lat::WilsonLattice, algorithm::Symbol=:Heatbath)
    cfg = zero(Cfg{lat})
    if algorithm == :Heatbath
        sample! = Heatbath{lat}()
    elseif algorithm == :PseudoHeatbath
        sample! = PseudoHeatbath{lat}()
    else
        error("Unknown algorithm requested")
    end
    return sample!, cfg
end

struct Obs{lat}
    U::Matrix{ComplexF64}
    V::Matrix{ComplexF64}
    W::Matrix{ComplexF64}
    function Obs{lat}() where {lat}
        U = zeros(ComplexF64, (lat.N,lat.N))
        V = zeros(ComplexF64, (lat.N,lat.N))
        W = zeros(ComplexF64, (lat.N,lat.N))
        new(U,V,W)
    end
end

function (obs::Obs{lat})(cfg::Cfg{lat})::Dict{String,Any} where {lat}
    r = Dict{String,Any}()
    r["action"] = action(obs,cfg)
    r["polyakov"] = polyakov(obs,cfg)
    r["quarkpotential"] = quarkpotential(obs, cfg)
    return r
end

function plaquette(obs::Obs{lat}, cfg::Cfg{lat}, i::Int, μ::Int, ν::Int)::ComplexF64 where {lat}
    iμ = translate(lat.geom, i, μ)
    iν = translate(lat.geom, i, ν)
    # Evaluate: U(i,ν)† U(iν,μ)† U(iμ,ν) U(i,μ)
    # This is: (U(iν,μ) U(i,ν))† U(iμ,ν) U(i,μ)
    @views mul!(obs.U, cfg.U[:,:,μ,iν], cfg.U[:,:,ν,i])
    adjoint!(obs.V, obs.U)
    @views mul!(obs.U, obs.V, cfg.U[:,:,ν,iμ])
    @views mul!(obs.V, obs.U, cfg.U[:,:,μ,i])
    return 1-tr(obs.V)/lat.N
end

function plaquette(obs::Obs{lat}, cfg::Cfg{lat})::Float64 where {lat}
    r::Float64 = 0.
    for i in lat.geom
        for μ in 1:lat.geom.d
            for ν in 1:(μ-1)
                r += real(plaquette(obs, cfg, i, μ, ν))
            end
        end
    end
    r *= 2 / (volume(lat.geom) * lat.geom.d * (lat.geom.d-1))
    return r
end

function action(obs::Obs{lat}, cfg::Cfg{lat})::Float64 where {lat}
    S::Float64 = 0.
    @views for i in lat.geom
        for μ in 1:lat.geom.d
            for ν in 1:(μ-1)
                iμ = translate(lat.geom, i, μ)
                iν = translate(lat.geom, i, ν)
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

function action(cfg::Cfg{lat})::Float64 where {lat}
    obs = Obs{lat}()
    action(obs, cfg)
end

function wilsonloop(obs::Obs{lat}, cfg::Cfg{lat}, i, μ, Lx, ν, Ly)::ComplexF64 where {lat}
    obs.U .= 0
    for a in 1:lat.N
        obs.U[a,a] = 1
    end
    for n in 1:Lx
        @views mul!(obs.V, cfg.U[:,:,μ,i], obs.U)
        obs.U .= obs.V
        i = translate(lat.geom, i, μ)
    end
    for n in 1:Ly
        @views mul!(obs.V, cfg.U[:,:,ν,i], obs.U)
        obs.U .= obs.V
        i = translate(lat.geom, i, ν)
    end
    for n in 1:Lx
        i = translate(lat.geom, i, μ, -1)
        @views adjoint!(obs.W, cfg.U[:,:,μ,i])
        mul!(obs.V, obs.W, obs.U)
        obs.U .= obs.V
    end
    for n in 1:Ly
        i = translate(lat.geom, i, ν, -1)
        @views adjoint!(obs.W, cfg.U[:,:,ν,i])
        mul!(obs.V, obs.W, obs.U)
        obs.U .= obs.V
    end
    return 1 - tr(obs.U)/lat.N
end

function wilsonloop(obs::Obs{lat}, cfg::Cfg{lat}, Lx::Int, Lt::Int)::ComplexF64 where {lat}
    r::ComplexF64 = 0.
    ν = lat.geom.d
    for i in lat.geom
        for μ in 1:lat.geom.d-1
            r += wilsonloop(obs, cfg, i, μ, Lx, ν, Lt)
        end
    end
    return r/(volume(lat.geom)*(lat.geom.d-1))
end

function polyakov(obs::Obs{lat}, cfg::Cfg{lat}, i::Int)::ComplexF64 where {lat}
    obs.V .= 0
    for a in 1:lat.N
        obs.V[a,a] = 1.
    end
    for t in 1:lat.geom.β
        @views mul!(obs.U, cfg.U[:,:,lat.geom.d,i], obs.V)
        obs.V .= obs.U
        i = translate(lat.geom, i, lat.geom.d)
    end
    return tr(obs.U)
end

function polyakov(obs::Obs{lat}, cfg::Cfg{lat})::ComplexF64 where {lat}
    r::ComplexF64 = 0.
    for i in 1:(lat.geom.L^(lat.geom.d-1))
        r += polyakov(obs, cfg, i)
    end
    return r/(lat.geom.L^(lat.geom.d-1))
end

function quarkpotential(obs::Obs{lat}, cfg::Cfg{lat}, x::Int)::Float64 where {lat}
    z::Float64 = 0.
    for i in 1:(lat.geom.L^(lat.geom.d-1))
        for μ in 1:lat.geom.d-1
            j = translate(lat.geom, i, μ, x)
            P = polyakov(obs, cfg, i)
            P′ = polyakov(obs, cfg, j)
            z += real((P*conj(P′))/(lat.geom.d-1)/(lat.geom.L^(lat.geom.d-1)))
        end
    end
    if z < 0
        z = 0
    end
    return -log(z)/lat.geom.β
end

function quarkpotential!(v::Vector{Float64}, obs::Obs{lat}, cfg::Cfg{lat}) where {lat}
    for x in 1:lat.geom.L
        v[x] = quarkpotential(obs, cfg, x)
    end
end

function quarkpotential(obs::Obs{lat}, cfg::Cfg{lat})::Vector{Float64} where {lat}
    v = zeros(Float64, lat.geom.L)
    quarkpotential!(v, obs, cfg)
    return v
end

function Observer(lat::WilsonLattice)
    return Obs{lat}()
end

function write(io::IO, cfg::Cfg{lat}) where {lat}
    for i in lat.geom
        for μ in 1:lat.geom.d
            for a in 1:lat.N
                for b in 1:lat.N
                    write(io, hton(cfg.U[a,b,μ,i]))
                end
            end
        end
    end
end

function read(io::IO, T::Type{Cfg{lat}})::Cfg{lat} where {lat}
    cfg = zero(T)
    for i in lat.geom
        for μ in 1:lat.geom.d
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

