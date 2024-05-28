module Lattices

export Lattice, Configuration, Observer, Sampler
export calibrate!

abstract type Lattice end
abstract type Configuration end
abstract type Observer end
abstract type Sampler end

function calibrate! end

end
