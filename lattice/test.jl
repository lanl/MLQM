#!/usr/bin/env julia

using Test

using LinearAlgebra: det

include("ising.jl")
include("scalar.jl")
include("qcd.jl")
include("ym.jl")

using .YangMills

@testset verbose=true "QCD" begin
end

@testset verbose=true "Ising" begin
    @testset "Serializing configurations" begin
        io = IOBuffer()
        lat = Ising.Lattice(5, 3, 0.1, 3)
        cfg = rand(Ising.Configuration{lat})
        write(io, cfg)
        seekstart(io)
        cfg′ = read(io, Ising.Configuration{lat})
        for i in lat
            @test cfg.σ[i] == cfg′.σ[i]
        end
    end
end

@testset verbose=true "Scalar" begin
    @testset "Serializing configurations" begin
        # TODO
    end
end

@testset verbose=true "Yang-Mills" begin
    @testset "Serializing configurations" begin
        io = IOBuffer()
        lat = Lattice(5,0.1)
        cfg = rand(Configuration{lat})
        write(io, cfg)
        seekstart(io)
        cfg′ = read(io, Configuration{lat})
        for i in lat
            for μ in 1:lat.d
                for a in 1:lat.N
                    for b in 1:lat.N
                        @test cfg.U[a,b,μ,i] == cfg′.U[a,b,μ,i]
                    end
                end
            end
        end
    end

    @testset verbose=true "unitarize!" begin
        @testset "is unitary" begin
            for N in 2:10
                U = randn(ComplexF64, (N,N))
                unitarize!(U)
                M = U'U
                for n in 1:N
                    for m in 1:N
                        if n == m
                            @test abs(M[n,m] - 1) < 1e-8
                        else
                            @test abs(M[n,m]) < 1e-8
                        end
                    end
                end
            end
        end

        @testset "no allocations" begin
            for N in 2:10
                U = randn(ComplexF64, (N,N))
                allocs = @allocations unitarize!(U)
                @test allocs == 0
            end
        end
    end

    @testset verbose=true "sunitarize!" begin
        @testset "is unitary" begin
            for N in 2:10
                U = randn(ComplexF64, (N,N))
                sunitarize!(U)
                M = U'U
                for n in 1:N
                    for m in 1:N
                        if n == m
                            @test abs(M[n,m] - 1) < 1e-8
                        else
                            @test abs(M[n,m]) < 1e-8
                        end
                    end
                end
            end
        end

        @testset "is special" begin
            for N in 2:10
                U = randn(ComplexF64, (N,N))
                sunitarize!(U)
                @test det(U) ≈ 1.
            end
        end
    end

    @testset verbose=true "UnitarySampler" begin
        N = 3
        s! = UnitarySampler(N, 0.2)
        @testset "is unitary" begin
            U = zeros(ComplexF64, (N,N))
            for k in 1:10
                s!(U)
                M = U'U
                for n in 1:N
                    for m in 1:N
                        if n == m
                            @test abs(M[n,m] - 1) < 1e-8
                        else
                            @test abs(M[n,m]) < 1e-8
                        end
                    end
                end
            end
        end
        @testset "resample does not allocate" begin
            allocs = @allocations resample!(s!, 0.2)
            @test allocs == 0
        end
    end

    @testset verbose=true "SpecialUnitarySampler" begin
        N = 3
        s! = SpecialUnitarySampler(N, 0.2)
        @testset "is unitary" begin
            U = zeros(ComplexF64, (N,N))
            for k in 1:10
                s!(U)
                M = U'U
                for n in 1:N
                    for m in 1:N
                        if n == m
                            @test abs(M[n,m] - 1) < 1e-8
                        else
                            @test abs(M[n,m]) < 1e-8
                        end
                    end
                end
            end
        end
        @testset "is special" begin
            U = zeros(ComplexF64, (N,N))
            for k in 1:10
                s!(U)
                d = det(U)
                @test d ≈ 1.
            end
        end
        @testset "resample does not allocate" begin
            allocs = @allocations resample!(s!, 0.2)
            @test allocs == 0
        end
    end

    @testset "lattice geometry" begin
        for β in 2:20
            lat = Lattice(12, 0.1, β=β)
            for k in 1:100
                μ = rand(1:lat.d)
                i = rand(1:volume(lat))
                n = rand(-lat.L:lat.L)
                j = trans(lat, i, μ, n=n)
                for ν in 1:lat.d
                    x = coordinate(lat, i, ν)
                    y = coordinate(lat, j, ν)
                    if μ == ν
                        if μ == lat.d
                            @test 1+mod(x+n-1,lat.β) == y
                        else
                            @test 1+mod(x+n-1,lat.L) == y
                        end
                    else
                        @test x == y
                    end
                end
            end
        end
    end

    @testset verbose=true "observables" begin
        lat = Lattice(5, 0.1, β=6)
        cfg = rand(Configuration{lat})
        obs = Observer{lat}()
        @testset "action does not allocate" begin
            allocs = @allocations action(obs, cfg)
            @test allocs == 0
        end

        @testset "are gauge invariant" begin
            for k in 1:100
                i = rand(1:volume(lat))
                U = rand(ComplexF64, (lat.N,lat.N))
                obs1 = obs(cfg)
                sunitarize!(U)
                gauge!(cfg, i, U)
                obs2 = obs(cfg)
                @test abs(obs1["action"] - obs2["action"]) / obs1["action"] < 1e-6
            end
        end
    end
end

