using ArgParse
using Dates: now
using Printf: @sprintf
using Statistics: mean

import Base: push!

using LatticeFieldTheories

mutable struct MeanBuffer
    N::Int
    K::Int
    hist::Vector{Any}
    recent::Vector{Any}

    function MeanBuffer(N::Int)
        new(N, 1, Any[], Any[])
    end
end

function push!(b::MeanBuffer, x)::Bool
    push!(b.recent, x)
    if length(b.recent) == b.K
        push!(b.hist, mean(b.recent))
        empty!(b.recent)
        if length(b.hist) == b.N
            # Squeeze!
            nhist = Any[]
            for n in 1:floor(Int,b.N/2)
                push!(nhist, mean(b.hist[2*n-1:2*n]))
            end
            b.hist = nhist
            b.K *= 2
        end
        return true
    end
    return false
end

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "-d","--directory"
                help = "Directory to store samples"
                arg_type = String
            "-o","--output"
                help = "File for storing running average of observables"
                arg_type = String
            "model"
                help = "Model specification file"
                arg_type = String
                required = true
            "-S","--samples"
                help = "Number of samples"
                arg_type = Int
                default = 10000
            "--skip"
                help = "Number of steps to skip for thermalization"
                arg_type = Int
                default = 1
            "--steps"
                help = "Number of steps to take in between samples"
                arg_type = Int
                default = 1
            "-A","--algorithm"
                help = "Sampling algorithm"
                arg_type = String
            "-D","--define"
                help = "Set variable in model evaluation"
                arg_type = String
                action = :append_arg
            "-O","--observe"
                help = "Only store observables"
                arg_type = String
                required = false
            "--time"
                help = "Time limit"
                arg_type = Float64
                default = Inf
        end
        parse_args(s)
    end

    start = now()

    modelModule = Module()
    Base.eval(modelModule, :(using LatticeFieldTheories))
    for defstr in args["define"]
        varname, val = split(defstr, "=")
        varsym = Symbol(varname)
        valexp = Meta.parse(val)
        Base.eval(modelModule, :($varsym = $valexp))
    end
    modelExpr = Meta.parse(read(args["model"], String))
    lat = Base.eval(modelModule, modelExpr)
    sample!, cfg = if isnothing(args["algorithm"])
        Sampler(lat)
    else
        Sampler(lat, args["algorithm"])
    end

    if isnothing(args["observe"])
        latmeta = Dict("START" => start,
                       "MACHINE" => Sys.MACHINE,
                       "lattice" => lat,
                      )
        dos = DOS(args["sample-directory"], latmeta)
    else
        observations = MeanBuffer(200)
    end

    calibrate!(sample!, cfg)
    for s in 1:args["skip"]
        sample!(cfg)
    end
    if isnothing(args["observe"])
        for n in 1:args["samples"]
            for s in 1:args["steps"]
                sample!(cfg)
            end
            cfgmeta = Dict("NOW" => now(),
                           "n" => n
                          )
            save((@sprintf "cfg%05d" n), dos, cfgmeta) do f
                write(f, cfg)
            end
        end
    else
        function saveMeasurements()
            open(args["output"], "w") do f
                println(f, "# $(observations.K)")
                for o in observations.hist
                    println(f, o)
                end
            end
        end
        obs = Observer(lat)
        try
            tstart = time()
            while true
                if time() - tstart > args["time"]
                    saveMeasurements()
                    break
                end
                for s in 1:args["steps"]
                    sample!(cfg)
                end
                o = obs(cfg)[args["observe"]]
                if push!(observations, o) && !isnothing(args["output"])
                    saveMeasurements()
                end
            end
        catch e
            if e isa InterruptException
                println(stderr, "Interrupted; writing measurements one last time")
                saveMeasurements()
            else
                rethrow(e)
            end
        end
    end
end

Base.exit_on_sigint(false)
main()

