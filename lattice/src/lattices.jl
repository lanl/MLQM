module Lattices

export Lattice, Configuration, Observer, Sampler
export CfgType
export calibrate!

abstract type Lattice end
abstract type Configuration end
abstract type Observer end
abstract type Sampler end

function CfgType end

function calibrate! end

Sampler(lat, alg::String) = Sampler(lat, Symbol(alg))

end
