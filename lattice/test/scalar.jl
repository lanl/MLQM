module TestScalar

using Test

using LatticeFieldTheories

@testset verbose=true "Non-allocation" begin
    @testset "Heatbath does not allocate" begin
        lat = Scalar.IsotropicLattice(CartesianGeometry(3,5,3), 2, 0.1, 0.1)
        cfg = rand(Scalar.Cfg{lat})
        hb! = Scalar.Heatbath{lat}()
        allocs = @allocations calibrate!(hb!, cfg)
        @test allocs == 0
        allocs = @allocations hb!(cfg)
        @test allocs == 0
    end

    @testset "Wolff does not allocate" begin
        lat = Scalar.IsotropicLattice(CartesianGeometry(3,5,3), 2, 0.1, 0.1)
        cfg = rand(Scalar.Cfg{lat})
        wolff! = Scalar.Wolff{lat}()
        allocs = @allocations calibrate!(wolff!, cfg)
        @test allocs == 0
        allocs = @allocations wolff!(cfg)
        @test allocs == 0
    end

    @testset "action() does not allocate" begin
        lat = Scalar.IsotropicLattice(CartesianGeometry(3,5,3), 2, 0.1, 0.1)
        cfg = rand(Scalar.Cfg{lat})
        obs = Observer(lat)
        allocs = @allocations Scalar.action(obs, cfg)
        @test allocs == 0
    end
end

@testset "Serialization" begin
    io = IOBuffer()
    lat = Scalar.IsotropicLattice(CartesianGeometry(3,5,3), 2, 0.1, 0.1)
    cfg = rand(Scalar.Cfg{lat})
    write(io, cfg)
    seekstart(io)
    cfg′ = read(io, Scalar.Cfg{lat})
    for i in lat.geom
        for n in 1:lat.N
            @test cfg.ϕ[n,i] == cfg′.ϕ[n,i]
        end
    end
end

end

