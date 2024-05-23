#!/usr/bin/env julia

using ArgParse

include("ising.jl")
include("scalar.jl")
include("qcd.jl")
include("ym.jl")

using .YangMills

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
        end
        parse_args(s)
    end
end

main()

