module Lattices

export Lattice, Configuration, Observer, Sampler
export calibrate!

abstract type Lattice end
abstract type Configuration{L<:Lattice} end
abstract type Observer{L<:Lattice} end
abstract type Sampler{L<:Lattice} end

function calibrate! end

end
