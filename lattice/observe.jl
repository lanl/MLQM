#!/usr/bin/env julia

using ArgParse

include("dos.jl")
include("ym.jl")

using .YangMills

function bootstrap(f, x; K::Int=1000)
    m = f(x)
    y = Vector{typeof(m)}(undef, K)
    for k in 1:K
        y[k] = f(rand(x, length(x)))
    end
    return m, std(y)
end
bootstrap(x; K) = bootstrap(mean, x; K=K)

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "sampleDirectory"
                help = "Directory to retrieve samples"
                arg_type = String
                required = true
        end
        parse_args(s)
    end

    dos = DOS(args["sampleDirectory"])
    lat = Lattice(dos["L"], dos["g"]; N=dos["N"], d=dos["d"], β=dos["β"])
    obs = Observer{lat}()
    for sample in dos
        cfg = open(sample) do f
            read(f, Configuration{lat})
        end
        println(action(obs,cfg))
    end
end

main()

