#!/usr/bin/env julia

using ArgParse

include("dos.jl")
include("ym.jl")

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
    lat = Lattice(dos["L"], dos["g"], dos["N"], dos["d"])
    for sample in dos
        cfg = open(sample) do f
            read(f, Configuration{lat})
        end
    end
end

main()

