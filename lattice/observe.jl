#!/usr/bin/env julia

using ArgParse

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

    for fn in readdir(args["sampleDirectory"])
        open(args["sampleDirectory"] * "/" * fn) do f
        end
    end
end

main()

