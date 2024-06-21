function step!(param::Vector, circuit::AbstractBlock, opt;
	cost_lst::Union{Vector, Nothing} = nothing, data, guided::Symbol, observable_lst::Vector,
)

	data_batch = shuffle(data)[1:train_config["batch_size"], :]
	grad = let
		func(paras::Vector) = begin
			cost(data_batch, paras, circuit, guided, observable_lst)
		end
		gradient(func, param)
	end

	Optimisers.update!(opt, param, grad)
	push!(cost_lst, cost(data_batch, param, circuit, guided, observable_lst))
end

observable1 = kron(NUM_BIT, 1 => Z)
observable2 = kron(NUM_BIT, NUM_BIT => Z)

step!(param::Vector, opt; cost_lst, guided::Symbol) = begin
	step!(param, circuit, opt; cost_lst = cost_lst, data = data, guided = guided, observable_lst = [observable1])
end


function gradient(f::Function, x::Vector; ϵ = 1e-8)
	gradients = zero(x)
	for i in eachindex(x)
		dxᵢ = zero(x)
		dxᵢ[i] = ϵ
		df_dxᵢ = (f(x + dxᵢ) - f(x - dxᵢ)) / 2ϵ
		gradients[i] = df_dxᵢ
	end
	return gradients
end

signoid(x; κ = 4) = 1 / (1 + exp(-κ * x))

loss(obs, label) = (sum(signoid, obs) - label)^2

function cost(
	guidance::Function, encoding::Function,
	data_batch::DataFrame, param::Vector, circuit::AbstractBlock,
	observable_lst::Vector,
)::Float64
	dispatch!(circuit, param)

	total_loss = 0
	for (x, y, label) in eachrow(data_batch)
		expectations = quantum_nodes((x, y), encoding, circuit, observable_lst; is_CUDA = train_config["is_CUDA"])
		g = guidance(circuit, observable_lst)
		total_loss += loss(expectations, label) + train_config["guidance_coeff"] * g
	end
	return total_loss / nrow(data_batch)
end

cost(data_batch, paras, circuit, guided, observable_lst) = begin
	dispatch!(circuit, paras)
	total_loss = 0
	for (x, y, label) in eachrow(data_batch)
		if guided == :raw
			expectations = quantum_nodes((x, y), binary_encode, circuit, observable_lst; is_CUDA = train_config["is_CUDA"])
		elseif guided == :guided
			expectations = guided_point_node((x, y), circuit, observable_lst)
		end
		total_loss += loss(expectations, label)
	end
	return total_loss / nrow(data_batch)
end

guided_point_node((x,y)::Tuple, circuit::AbstractBlock, observable_lst) = begin
    expectations1 = quantum_nodes((x, y), binary_encode, circuit, observable_lst; is_CUDA = train_config["is_CUDA"])
    expectations2 = quantum_nodes((y, x), binary_encode, circuit, observable_lst; is_CUDA = train_config["is_CUDA"])
    return 1 / 2 * (expectations1 + expectations2)
end

"""
Encoding the data into a quantum register, then apply the circuit to the encoded register. 
Finally, getting the expectation value of the Hamiltonian.
"""
function quantum_nodes(data::Tuple, encoder::Function, circuit::AbstractBlock, hamiltonian_ls::Vector; is_CUDA = false)
	num_qubits = nqubits(circuit)
	# Encode the data
	encoded_state = encoder(num_qubits ÷ 2, data, is_CUDA)

	rst = [expect(ob, encoded_state => circuit) for ob in hamiltonian_ls]
	return rst
end


function twirl_encode(num_bit::Int, point::Tuple, is_CUDA = false)::AbstractRegister
	x, y = point
	state1 = binary_encode(num_bit, (x, y), is_CUDA) |> density_matrix
	state2 = binary_encode(num_bit, (y, x), is_CUDA) |> density_matrix
	# state3 = binary_encode(num_bit, (1-x, y), is_CUDA)
	# state4 = binary_encode(num_bit, (y, 1-x), is_CUDA)
	res = 1 / 2 * (state1 + state2)
	# normalize!(res)
	return res
end

function binary_encode(num_bit::Int, point::Tuple, is_CUDA = false)::AbstractRegister
	"""
	This function encode a number between 0 and 1 to a quantum register.
	"""
	x_bitstring = binary_convert(point[1], num_bit)
	y_bitstring = binary_convert(point[2], num_bit)

	bit_ls = []
	for b in x_bitstring * y_bitstring
		if b == '0'
			push!(bit_ls, bit"0")
		else
			push!(bit_ls, bit"1")
		end
	end

	whole_bitstring = join(bit_ls...)
	prod_state = product_state(whole_bitstring)

	return is_CUDA ? cu(prod_state) : prod_state
end


function binary_convert(num, n)
	# 将小数部分乘以 2 并取整
	binary_digits = Int[]
	while length(binary_digits) < n
		num *= 2
		push!(binary_digits, floor(Int, num))
		num -= binary_digits[end]
	end

	# 返回二进制数字符串
	return join(binary_digits, "")
end

