module TestScalar

using Test

using LatticeFieldTheories

@testset "Non-allocation" begin
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

