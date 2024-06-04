module TestYangMills

using LinearAlgebra
using Test

using LatticeFieldTheories

@testset "Serializing configurations" begin
    io = IOBuffer()
    lat = YangMills.WilsonLattice(5,0.1)
    cfg = rand(YangMills.Cfg{lat})
    write(io, cfg)
    seekstart(io)
    cfg′ = read(io, YangMills.Cfg{lat})
    for i in lat.geom
        for μ in 1:lat.geom.d
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
            YangMills.unitarize!(U)
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
            allocs = @allocations YangMills.unitarize!(U)
            @test allocs == 0
        end
    end
end

@testset verbose=true "sunitarize!" begin
    @testset "is unitary" begin
        for N in 2:10
            U = randn(ComplexF64, (N,N))
            YangMills.sunitarize!(U)
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
            YangMills.sunitarize!(U)
            @test det(U) ≈ 1.
        end
    end
end

@testset verbose=true "UnitarySampler" begin
    N = 3
    s! = YangMills.UnitarySampler(N, 0.2)
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
        allocs = @allocations YangMills.resample!(s!, 0.2)
        @test allocs == 0
    end
end

@testset verbose=true "SpecialUnitarySampler" begin
    N = 3
    s! = YangMills.SpecialUnitarySampler(N, 0.2)
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
            @test abs(d-1.) ≤ 1e-6
        end
    end
    @testset "resample does not allocate" begin
        allocs = @allocations YangMills.resample!(s!, 0.2)
        @test allocs == 0
    end
end

@testset verbose=true "observables" begin
    lat = YangMills.WilsonLattice(5, 0.1, β=6)
    cfg = rand(YangMills.Cfg{lat})
    obs = YangMills.Obs{lat}()
    @testset "action does not allocate" begin
        allocs = @allocations YangMills.action(obs, cfg)
        @test allocs == 0
    end

    @testset "polyakov does not allocate" begin
        allocs = @allocations YangMills.polyakov(obs, cfg)
        @test allocs == 0
    end

    @testset "quarkpotential does not allocate" begin
        for x in 1:4
            allocs = @allocations YangMills.quarkpotential(obs, cfg, x)
            @test allocs == 0
        end
    end

    @testset "wilsonloop does not allocate" begin
        for Lt in 1:4
            for Lx in 1:4
                allocs = @allocations YangMills.wilsonloop(obs, cfg, Lx, Lt)
                @test allocs == 0
            end
        end
    end

    @testset "wilsonloop is gauge invariant" begin
        for Lt in 1:4
            for Lx in 1:4
                wl1 = YangMills.wilsonloop(obs, cfg, Lx, Lt)
                i = rand(1:volume(lat.geom))
                U = rand(ComplexF64, (lat.N,lat.N))
                YangMills.sunitarize!(U)
                YangMills.gauge!(cfg, i, U)
                wl2 = YangMills.wilsonloop(obs, cfg, Lx, Lt)
                @test wl1 ≈ wl2
            end
        end
    end

    @testset "are gauge invariant" begin
        for k in 1:100
            i = rand(1:volume(lat.geom))
            U = rand(ComplexF64, (lat.N,lat.N))
            obs1 = obs(cfg)
            YangMills.sunitarize!(U)
            YangMills.gauge!(cfg, i, U)
            obs2 = obs(cfg)
            @test abs(obs1["action"] - obs2["action"]) / obs1["action"] < 1e-6
            @test abs(obs1["polyakov"] - obs2["polyakov"]) / abs(obs1["polyakov"]) < 1e-6
        end
    end
end

@testset verbose=true "heatbath" begin
    lat = YangMills.WilsonLattice(5, 0.1)
    cfg = rand(YangMills.Cfg{lat})
    hb! = YangMills.Heatbath{lat}()
    @testset "does not allocate" begin
        allocs = @allocations hb!(cfg)
        @test allocs == 0
    end
end

end

