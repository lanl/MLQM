#!/usr/bin/env julia

using ArgParse
using Dates: now

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

    latmeta = ("START" => start,
               "L" => lat.L,
               "g" => lat.g,
               "N" => lat.N,
               "d" => lat.d
              )

    for n in 1:10
        for s in 1:10
            sweep!(cfg, lat)
        end
        cfgmeta = ("NOW" => now(),
                   "n" => n
                  )
        open(args["sampleDirectory"]*"/cfg$n", "w") do f
            for (k,v) in latmeta
                println(f, "$k $v")
            end
            println(f, "")
            for (k,v) in cfgmeta
                println(f, "$k $v")
            end
            println(f, "")
            write(f, cfg)
        end
    end
end

main()

