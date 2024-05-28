module TestGeometry

using Test

using LatticeFieldTheories

@testset "Non-allocation" begin
    @testset "Lattice iteration does not allocate" begin
        geom = CartesianGeometry(3,5,3)
        s::Int = 0
        allocs = @allocations for i in geom
            s += 1
        end
        @test s == volume(geom)
        @test allocs == 0
    end

    @testset "Adjacency does not allocate" begin
        geom = CartesianGeometry(3,5,3)
        # TODO
    end
end

end
