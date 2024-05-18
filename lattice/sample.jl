#!/usr/bin/env julia

using ArgParse
using Dates: now

include("dos.jl")
include("lattices.jl")

include("ising.jl")
include("qcd.jl")
include("higgs.jl")
include("negahiggs.jl")
include("scalar.jl")
include("ym.jl")

using .Lattices
using .Ising

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "sampleDirectory"
                help = "Directory to store samples"
                arg_type = String
                required = true
            "model"
                help = "Model specification"
                arg_type = String
                required = true
            "-S","--samples"
                help = "Number of samples"
                arg_type = Int
                default = 100
        end
        parse_args(s)
    end

    start = now()

    modelExpr = Meta.parse(args["model"])
    lat = eval(modelExpr)
    cfg = zero(Configuration(lat))

    latmeta = Dict("START" => start,
                   "MACHINE" => Sys.MACHINE,
                   "lattice" => modelExpr,
                  )
    dos = DOS(args["sampleDirectory"], latmeta)

    sample! = Sampler(lat)
    calibrate!(sample!, cfg)
    for n in 1:args["samples"]
        for s in 1:100
            sample!(cfg)
        end
        cfgmeta = Dict("NOW" => now(),
                       "n" => n
                      )
        save("cfg$n", dos, cfgmeta) do f
            write(f, cfg)
        end
    end
end

main()

