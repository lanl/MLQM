#!/usr/bin/env julia

using ArgParse
using Dates: now

include("dos.jl")
include("ising.jl")
include("qcd.jl")
include("scalar.jl")
include("ym.jl")

using .YangMills

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "sampleDirectory"
                help = "Directory to store samples"
                arg_type = String
                required = true
            "-N"
                help = "Number of colors"
                arg_type = Int
                required = false
                default = 3
            "-d"
                help = "Spacetime dimension"
                arg_type = Int
                required = false
                default = 4
            "-T"
                help = "Time dimension (defaults to L)"
                arg_type = Int
                required = false
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

    lat = if isnothing(args["T"])
        Lattice(args["L"], args["g"], N=args["N"], d=args["d"])
    else
        Lattice(args["L"], args["g"], β=args["T"], N=args["N"], d=args["d"])
    end
    cfg = zero(Configuration{lat})

    latmeta = Dict("START" => start,
                   "MACHINE" => Sys.MACHINE,
                   "L" => lat.L,
                   "β" => lat.β,
                   "g" => lat.g,
                   "N" => lat.N,
                   "d" => lat.d
                  )
    dos = DOS(args["sampleDirectory"], latmeta)

    heatbath! = Heatbath{lat}()
    calibrate!(heatbath!, cfg)
    for n in 1:50
        for s in 1:200
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

