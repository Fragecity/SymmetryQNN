signoid(x; κ = 4) = 1 / (1 + exp(-κ*x))

loss((ob1, ob2), label) = (signoid(ob1) + signoid(ob2) - label)^2

function cost(
    guidance::Function, encoding::Function,
    data_batch::DataFrame, circuit::AbstractBlock, 
    observable1::AbstractBlock, observable2::AbstractBlock
    )

    total_loss = 0
    for (x, y, label) in eachrow(data_batch)
        expectation1, expectation2 = quantum_nodes((x, y), encoding, circuit, [observable1, observable2]; is_CUDA = train_config["is_CUDA"])
        g = guidance(circuit, observable1, observable2)
        total_loss += loss((expectation1, expectation2), label) + train_config["guidance_coeff"] * g
    end
    return total_loss/nrow(data_batch)
end

cost(data_batch::DataFrame, circuit::AbstractBlock, observable1::AbstractBlock, observable2::AbstractBlock, guided::Symbol) = let 
    if guided == :raw
        encoder = binary_encode
    elseif guided == :guid
        encoder = twirl_encode
    end
    func(circuit, observable1, observable2) = 0
    cost(func, encoder, data_batch, circuit, observable1, observable2)
end

# function accuracy(circuit, param)
    
# end

