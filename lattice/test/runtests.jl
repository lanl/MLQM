
tests = ["geometry.jl"
         "ising.jl"
         "scalar.jl"
         "higgs.jl"
         "negahiggs.jl"
         "qcd.jl"
         "ym.jl"
        ]

for test in tests
    println(test)
    include(test)
end

