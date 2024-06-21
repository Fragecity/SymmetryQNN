observable1 = kron(NUM_BIT, 1=>Z)
observable2 = kron(NUM_BIT, NUM_BIT=>Z)

swap_symmetry = let 
    swap_lst = [Swap(NUM_BIT, (i, i+ NUM_BIT÷2)) for i in 1:NUM_BIT÷2]
    symmetry = reduce(*, swap_lst)
end

function guidance(circuit::AbstractBlock, observable1::AbstractBlock, observable2::AbstractBlock, symmetry_gate::T) where T
    return twirl(circuit, observable1, symmetry_gate) + twirl(circuit, observable2, symmetry_gate)
end

function twirl(circuit::AbstractBlock, observable::AbstractBlock, symmetry_gate::T) where T
    Õ = circuit' * observable * circuit
    SÕS = symmetry_gate * Õ * symmetry_gate
    return 1/2^NUM_BIT * norm(Õ - SÕS, 1)
end
