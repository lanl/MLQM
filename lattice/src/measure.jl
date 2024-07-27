using AlgebraOfGraphics
using ArgParse
import CairoMakie: save as savefig
using DataFrames
using Statistics: mean, std, cov

using LatticeFieldTheories

function bootstrap(f, x; K::Int=1000)
    m = f(x)
    y = Vector{typeof(m)}(undef, K)
    for k in 1:K
        y[k] = f(rand(x, length(x)))
    end
    return m, cov(y)
end
bootstrap(x::Vector{T}; K=1000) where {T} = bootstrap(mean, x; K=K)
function bootstrap(x::Matrix{T}, K=1000) where {T}
    bootstrap(x; K=K) do y
        mean(y, dim=1)
    end
end

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
            "-O","--observable"
                help = "Print particular observable"
                arg_type = String
                required = false
            "-C","--correlators"
                help = "Save correlators to file"
                arg_type = String
                required = false
            "-M","--measurements"
                help = "Directory for writing measurements"
                arg_type = String
                required = false
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
    if !isnothing(args["measurements"])
        mmeta = Dict{String,Any}()
        mdos = DOS(args["measurements"], mmeta)
    end
    modelExpr = Meta.parse(dos["lattice"])
    lat = eval(modelExpr)
    Cfg = CfgType(lat)
    obs = Observer(lat)
    dat = Dict{String,Any}()
    for sample in dos
        cfg = open(sample) do f
            read(f, Cfg)
        end
        observation = obs(cfg)
        for (k,v) in observation
            if !(k in keys(dat))
                dat[k] = typeof(v)[]
            end
            push!(dat[k], v)
        end
        # Write measurements
        if !isnothing(args["measurements"])
            fn = "measured-$(basename(sample.filename))"
            fmeta = Dict{String,Any}()
            save(fn, mdos, fmeta) do f
                println(f, observation)
            end
        end
        if !isnothing(args["observable"])
            println(observation[args["observable"]])
        end
    end
    if !isnothing(args["correlators"])
        c00, c00cov = bootstrap(dat["cor00"])
        open(args["correlators"], "w") do f
            println(f, c00)
            println(f, c00cov)
        end
    end
    if args["vis"]
        vdir = joinpath(args["sampleDirectory"], "vis")
        mkpath(vdir)

        # Mixing
        #fig = draw(data(df) * mapping(:n, :action))
        #mixing_fn = joinpath(vdir, "mixing.png")
        #savefig(mixing_fn, fig)
    end
end

main()

