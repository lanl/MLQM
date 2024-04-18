#!/usr/bin/env julia

using Test

include("ym.jl")

@testset "Serializing configurations" begin
    io = IOBuffer()
    lat = Lattice(5,0.1,3,4)
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

