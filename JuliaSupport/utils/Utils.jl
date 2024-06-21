module Utils
export binary_encode, twirl_encode, generate_datas, quantum_nodes, label_data

using DataFrames, Yao, CUDA

function twirl_encode(num_bit::Int, point::Tuple, is_CUDA = false)
	x,y = point
	state1 = binary_encode(num_bit, (x, y), is_CUDA)
	state2 = binary_encode(num_bit, (y, x), is_CUDA)
	state3 = binary_encode(num_bit, (1-x, y), is_CUDA)
	state4 = binary_encode(num_bit, (y, 1-x), is_CUDA)
	return 1/2 * (state1 + state2 + state3 + state4)
end

function binary_encode(num_bit::Int, point::Tuple, is_CUDA = false)
    """
    This function encode a number between 0 and 1 to a quantum register.
    """
    x_bitstring = binary_convert(point[1], num_bit)
    y_bitstring = binary_convert(point[2], num_bit)
    
    bit_ls = []
    for b in  x_bitstring * y_bitstring
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

"""
Encoding the data into a quantum register, then apply the circuit to the encoded register. 
Finally, getting the expectation value of the Hamiltonian.
"""
function quantum_nodes(data::Tuple, encoder::Function, circuit::AbstractBlock, hamiltonian_ls::Vector; is_CUDA = false)
	num_qubits = nqubits(circuit)
	# Encode the data
	encoded_state = encoder(num_qubits÷2, data, is_CUDA)

	rst = [expect(ob, encoded_state=> circuit) for ob in hamiltonian_ls]
	return rst
end






function is_inCircle((x, y)::Tuple, center::Tuple, radius::AbstractFloat)
	distance_square = (x - center[1])^2 + (y - center[2])^2
	return distance_square <= radius^2
end


"""
Generating a data (x,y) in [0, x_range] × [0, y_range].
"""
function generate_a_data((x_range, y_range)::Tuple)
	eps = 1e-10
	x = rand() * x_range
	y = rand() * y_range
	return (x, y)
end

function label_data((x, y)::Tuple, radius_inner::AbstractFloat, radius_outer::AbstractFloat, center::Tuple)
	label_inner = is_inCircle((x, y), center, radius_inner) ? 0 : 1
	label_outer = is_inCircle((x, y), center, radius_outer) ? 0 : 1
	return label_inner + label_outer
end

function generate_datas(num_data::Int, x_range::AbstractFloat, y_range::AbstractFloat, radius_inner::AbstractFloat, radius_outer::AbstractFloat, center::Tuple; bias::Bool = false)
	xs = []
	ys = []
	labels = []

	while length(xs) < num_data
		(x, y) = generate_a_data((x_range, y_range))
		(!bias || x ≥ y) && begin
			push!(xs, x)
			push!(ys, y)
			push!(labels, label_data((x, y), radius_inner, radius_outer, center))
		end
	end

	return DataFrame("x" => xs, "y" => ys, "labels" => labels)
end
end


# using .OtherUtils