using AlgebraOfGraphics
using ArgParse
import CairoMakie: save as savefig
using DataFrames
using Statistics: mean, std

using LatticeFieldTheories

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
    modelExpr = Meta.parse(dos["lattice"])
    lat = eval(modelExpr)
    obs = Observer(lat)()
    df = DataFrame(:n => [])
    for sample in dos
        cfg = open(sample) do f
            read(f, Configuration(lat))
        end
        observation = obs(cfg)
        observation["n"] = sample["n"]
        if isempty(df)
            for cn in keys(observation)
                df[!,cn] = []
            end
        end
        push!(df, observation)
    end
    if args["vis"]
        vdir = joinpath(args["sampleDirectory"], "vis")
        mkpath(vdir)

        # Mixing
        fig = draw(data(df) * mapping(:n, :action))
        mixing_fn = joinpath(vdir, "mixing.png")
        savefig(mixing_fn, fig)
    end
end

main()

