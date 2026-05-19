function convert_to_hyperedge_list(input_filepath::String, output_filepath::String)
    lines = readlines(input_filepath)
    
    total_rows = length(lines)
    println("Reading incidence matrix with $total_rows potential hyperedges...")
    
    open(output_filepath, "w") do io
        for (row_idx, line) in enumerate(lines)
            # Skip any empty lines
            if strip(line) == ""
                continue
            end
            
            row_values = parse.(Int8, split(strip(line), ","))
            
            vertices = findall(x -> x == 1, row_values)
            
            line_string = join(vertices, " ")
            println(io, line_string)
        end
    end
    
    println("Successfully reversed the process!")
    println("Hyperedge list saved to: $output_filepath")
end

input_file = "toy_data/toy_incidence_matrix.csv"
output_file = "toy_data/reconstructed_hypernetwork.csv"

convert_to_hyperedge_list(input_file, output_file)