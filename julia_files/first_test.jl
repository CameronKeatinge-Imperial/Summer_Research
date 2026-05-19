# first run
println("First file!")
using Pkg
#installing ORCHID
#Pkg.add(["CSV", "DataFrames"])
 #read in the toy data
using CSV
using DataFrames

# Read the TSV file into a DataFrame
individual_hypergraph_df = CSV.read("toy_data/toy.ihg.tsv", DataFrame, delim='\t', header=false)
# View the first few rows
println(individual_hypergraph_df)



using Orchid
#import Orchid: DisperseUnweightedClique, AggregationMax
X = Matrix{Float64}(individual_hypergraph_df)
print(X)

Orchid.hypergraph_curvatures(Orchid.DisperseUnweightedClique, Orchid.AggregateMean, X, 0.01)
#Orchid.hypergraph_curvatures