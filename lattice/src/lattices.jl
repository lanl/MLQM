module Lattices

export Lattice, Configuration, Observer, Sampler
export calibrate!

abstract type Lattice end

function Configuration end
function Observer end
function Sampler end
function calibrate! end

end
