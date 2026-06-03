using Pkg

function convert_to_incidence_matrix(input_filepath::String, output_filepath::String)
    # 1. Read all lines from the file
    lines = readlines(input_filepath)
    
    hyperedges = Vector{Vector{Int}}()
    max_vertex_id = 0
    
    for line in lines
        if strip(line) == ""
            continue
        end
        
        vertices = parse.(Int, split(line))
        push!(hyperedges, vertices)
        
        if !isempty(vertices)
            max_vertex_id = max(max_vertex_id, maximum(vertices))
        end
    end
    
    M = length(hyperedges)
    N = max_vertex_id
    
    println("Identified $M hyperedges and a maximum vertex ID of $N.")
    println("Constructing $M x $N incidence matrix...")
    
    incidence_matrix = zeros(Int8, M, N)
    
    for (row_idx, edge_vertices) in enumerate(hyperedges)
        for vertex_idx in edge_vertices
            incidence_matrix[row_idx, vertex_idx] = 1
        end
    end
    
    open(output_filepath, "w") do io
        for row in 1:M
            # Extract the row, join the numbers with commas, and write the line
            line_string = join(incidence_matrix[row, :], ",")
            println(io, line_string)
        end
    end
    
    println("Successfully saved the incidence matrix to: $output_filepath")
end

input_file = "toy_data/disease-gene.csv"
#input_file = "toy_data/reconstructed_hypernetwork.csv" -- works on the reconstructed file.
output_file = "toy_data/disease_gene_output.csv"

convert_to_incidence_matrix(input_file, output_file)