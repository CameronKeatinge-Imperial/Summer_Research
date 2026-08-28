
println("julia   : ", VERSION)
println("project : ", Base.active_project())
try
    @eval using Orchid
    println("Orchid  : loaded OK")
catch err
    println("Orchid  : FAILED TO LOAD")
    showerror(stdout, err); println()
    exit(2)
end

println("exports : ", join(sort(string.(names(Orchid))), ", "))

probe(prefixes, name) = begin
    hits = [p * name for p in prefixes if isdefined(Orchid, Symbol(p * name))]
    println(rpad(name, 20), isempty(hits) ? "NOT FOUND (tried " * join(prefixes .* name, ", ") * ")" :
                                            "-> " * join(hits, ", "))
end
probe(["Disperse", "Dispersion"], ARGS[1])
probe(["Aggregate", "Aggregation"], ARGS[2])
probe(["Cost"], ARGS[3])

if isdefined(Orchid, :hypergraph_curvatures)
    println("\nmethods(hypergraph_curvatures):")
    for m in methods(Orchid.hypergraph_curvatures); println("  ", m); end
else
    println("\nhypergraph_curvatures is NOT defined in Orchid")
end
