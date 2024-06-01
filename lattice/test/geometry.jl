module TestGeometry

using Test

using LatticeFieldTheories

@testset verbose=true "Non-allocation" begin
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
        allocs = @allocations adjacent(geom, 1)
        @test allocs == 0
        s::Int = 0
        allocs = @allocations for i in adjacent(geom, 1)
            s += 1
        end
        @test allocs == 0
        @test s == 6
    end
end

@testset "CartesianGeometry" begin
    for β in 2:20
        geom = CartesianGeometry(4, 12, β)
        for k in 1:100
            μ = rand(1:geom.d)
            i = rand(1:volume(geom))
            n = rand(-geom.L:geom.L)
            j = translate(geom, i, μ, n)
            for ν in 1:geom.d
                x = coordinate(geom, i, ν)
                y = coordinate(geom, j, ν)
                if μ == ν
                    if μ == geom.d
                        @test 1+mod(x+n-1,geom.β) == y
                    else
                        @test 1+mod(x+n-1,geom.L) == y
                    end
                else
                    @test x == y
                end
            end
        end
    end
end

end
