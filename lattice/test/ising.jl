module TestIsing

using Test

using LatticeFieldTheories

@testset "Non-allocation" begin
    @testset "Heatbath does not allocate" begin
        lat = Ising.IsotropicLattice(CartesianGeometry(3,5,3),0.3)
        sample!, cfg = Sampler(lat, :Heatbath)
        allocs_calibrate = @allocations calibrate!(sample!, cfg)
        @test allocs_calibrate == 0
        allocs = @allocations sample!(cfg)
        @test allocs == 0
    end

    @testset "Swendesen-Wang does not allocate" begin
        lat = Ising.IsotropicLattice(CartesianGeometry(3,5,3),0.3)
        sample!, cfg = Sampler(lat, :SwendsenWang)
        allocs_calibrate = @allocations calibrate!(sample!, cfg)
        @test allocs_calibrate == 0
        sample!(cfg) # The first call allocates a bit.
        allocs = @allocations sample!(cfg)
        @test allocs == 0
    end

    @testset "Action does not allocate" begin
        lat = Ising.IsotropicLattice(CartesianGeometry(3,5,3),0.3)
        sample!, cfg = Sampler(lat)
        obs = Ising.Obs{lat}()
        allocs = @allocations Ising.action(obs, cfg)
        @test allocs == 0
    end
end

@testset "Serialization" begin
    io = IOBuffer()
    lat = Ising.IsotropicLattice(CartesianGeometry(3,5,3), 0.1)
    cfg = rand(Ising.Cfg{lat.geom})
    write(io, cfg)
    seekstart(io)
    cfg′ = read(io, Ising.Cfg{lat.geom})
    for i in lat.geom
        @test cfg.σ[i] == cfg′.σ[i]
    end
end

end
