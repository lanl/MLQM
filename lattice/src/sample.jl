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
                help = "Model specification file"
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
            "-D","--define"
                help = "Set variable in model evaluation"
                arg_type = String
                action = :append_arg
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

    latmeta = Dict("START" => start,
                   "MACHINE" => Sys.MACHINE,
                   "lattice" => lat,
                  )
    dos = DOS(args["sampleDirectory"], latmeta)

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

