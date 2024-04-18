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

