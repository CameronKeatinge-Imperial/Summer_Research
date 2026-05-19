# first run
println("First file!")
using Pkg
using CSV
using DataFrames
using DelimitedFiles
using Orchid
using SparseArrays
#installing ORCHID
#Pkg.add(["CSV", "DataFrames"])
 #read in the toy data
X = readdlm("toy_data/toy_incidence_matrix.csv", ',', Int)
println(typeof(X))
println(X)
X_sparse = sparse(X) 
results = hypergraph_curvatures(
    Orchid.DisperseUnweightedClique,
    Orchid.AggregateMean,
    X_sparse,
    0.01,
    Orchid.CostOndemand
)
println(propertynames(results))


using Dates
using DelimitedFiles

timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
output_dir = joinpath("orchid_output", timestamp)
mkpath(output_dir)

for field in propertynames(results)
    
    file_path = joinpath(output_dir, "$(field).csv")
    value = getproperty(results, field)
    
    try
        writedlm(file_path, value, ',')
    catch e
        println("Warning: Could not save field '$field' as CSV. It might not be tabular data.")
    end
end

#not saving edge curvature explicitly right now, but saving some outputs.
println("Success! All array outputs saved as CSV files to: $output_dir")