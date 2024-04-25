# Directory of samples

import Base: getindex, iterate, open

function str2any(s::AbstractString)::Any
    try
        return parse(Int, s)
    catch
    end
    try
        return parse(Float64, s)
    catch
    end
    return s
end

function read_metadata(io::IO)::Dict{String,Any}
    md = Dict{String,Any}()
    l = readline(io)
    while !isempty(l)
        k, v = split(l, limit=2)
        md[k] = str2any(v)
        l = readline(io)
    end
    return md
end

function write_metadata(io::IO, md::Dict{String,Any})
    for (k,v) in md
        println(io, "$k $v")
    end
    println(io, "")
end

struct Sample
    filename::String
    metadata::Dict{String,Any}

    function Sample(fn::String)
        md = open(fn) do f
            read_metadata(f)
            read_metadata(f)
        end
        new(fn, md)
    end

    function Sample(fn::String, metadata::Dict{String,Any})
        md = open(fn) do f
            md′ = read_metadata(f)
            if metadata != md′
                error("Metadata do not match")
            end
            read_metadata(f)
        end
        new(fn, md)
    end
end

getindex(s::Sample, k::String) = s.metadata[k]

struct DOS
    dirname::String
    metadata::Dict{String,Any}
    samples::Vector{Sample}

    function DOS(dn::String)
        fns = filter(isfile, readdir(dn, join=true))
        md = open(read_metadata, fns[1])
        ss = Vector{Sample}()
        for fn in fns
            push!(ss, Sample(fn, md))
        end
        new(dn, md, ss)
    end

    function DOS(dn::String, meta::Dict{String,Any})
        if isdir(dn)
            if length(readdir(dn)) > 0
                error("Cannot overwrite metadata")
            end
        else
            mkdir(dn)
        end
        new(dn, meta, [])
    end
end

iterate(dos::DOS) = iterate(dos.samples)
iterate(dos::DOS, i) = iterate(dos.samples, i)

getindex(dos::DOS, k::String) = dos.metadata[k]

function open(s::Sample)
    f = open(s.filename)
    read_metadata(f)
    read_metadata(f)
    return f
end

function save(name::String, dos::DOS, fmeta)
    f = open(joinpath(dos.dirname, name), "w")
    write_metadata(f, dos.metadata)
    write_metadata(f, fmeta)
    return f
end

function save(f::Function, name::String, dos::DOS, fmeta)
    io = save(name, dos, fmeta)
    f(io)
    close(io)
end

