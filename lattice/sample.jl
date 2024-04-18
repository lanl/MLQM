#!/usr/bin/env julia

using ArgParse
using Dates: now

include("dos.jl")
include("ym.jl")

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "sampleDirectory"
                help = "Directory to store samples"
                arg_type = String
                required = true
            "-L"
                help = "Lattice size"
                arg_type = Int
                required = true
            "-g"
                help = "Coupling"
                arg_type = Float64
                required = true
        end
        parse_args(s)
    end

    start = now()

    lat = Lattice(args["L"], args["g"])
    cfg = zero(Configuration{lat})

    latmeta = Dict("START" => start,
                   "L" => lat.L,
                   "g" => lat.g,
                   "N" => lat.N,
                   "d" => lat.d
                  )
    dos = DOS(args["sampleDirectory"], latmeta)

    heatbath! = Heatbath{lat}()
    for n in 1:10
        for s in 1:10
            heatbath!(cfg)
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

