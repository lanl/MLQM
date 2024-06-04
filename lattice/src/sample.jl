using ArgParse
using Dates: now
using Printf: @sprintf

using LatticeFieldTheories

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
        end
        parse_args(s)
    end

    start = now()

    modelExpr = Meta.parse(args["model"])
    lat = eval(modelExpr)
    latmeta = Dict("START" => start,
                   "MACHINE" => Sys.MACHINE,
                   "lattice" => modelExpr,
                  )
    dos = DOS(args["sampleDirectory"], latmeta)

    sample!, cfg = if isnothing(args["algorithm"])
        Sampler(lat)
    else
        Sampler(lat, args["algorithm"])
    end
    calibrate!(sample!, cfg)
    for s in 1:args["skip"]
        sample!(cfg)
    end
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
end

main()

