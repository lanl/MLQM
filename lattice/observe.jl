#!/usr/bin/env julia

using AlgebraOfGraphics
using ArgParse
import CairoMakie: save as savefig
using DataFrames
using Statistics: mean, std

include("dos.jl")
include("ising.jl")
include("qcd.jl")
include("scalar.jl")
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
bootstrap(x; K=1000) = bootstrap(mean, x; K=K)

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "-q","--quiet"
                help = "Quiet mode"
                action = :store_true
            "-V","--vis"
                help = "Create visualizations"
                action = :store_true
            "sampleDirectory"
                help = "Directory to retrieve samples"
                arg_type = String
                required = true
        end
        parse_args(s)
    end

    dos = DOS(args["sampleDirectory"])
    if !args["quiet"]
        for (k,v) in dos.metadata
            println("# $k $v")
        end
    end
    lat = Lattice(dos["L"], dos["g"]; N=dos["N"], d=dos["d"], β=dos["β"])
    obs = Observer{lat}()
    acts = []
    plaqs = []
    polys = []
    ns = []
    for sample in dos
        cfg = open(sample) do f
            read(f, Configuration{lat})
        end
        push!(polys, YangMills.polyakov(obs,cfg))
        push!(plaqs, plaquette(obs,cfg))
        push!(acts, action(obs,cfg))
        push!(ns, sample["n"])
    end
    plaqm, plaqe = bootstrap(plaqs)
    polym, polye = bootstrap(polys)
    println("$plaqm $plaqe    $polym $polye")
    if args["vis"]
        vdir = joinpath(args["sampleDirectory"], "vis")
        mkpath(vdir)

        # Mixing
        df = DataFrame(n=ns, S=acts)
        fig = draw(data(df) * mapping(:n, :S))
        mixing_fn = joinpath(vdir, "mixing.png")
        savefig(mixing_fn, fig)
    end
end

main()

