using CSV, JSON, DataFrames, Plots, DrawFuncs

using Yao, Yao.EasyBuild

using VecOperations

config = JSON.parsefile("./src/config.json")
circuit_config = config["circuit_info"]
train_config = config["train_info"]
data_config = config["data_info"]

NUM_BIT = circuit_config["num_qubits"]
NUM_EPOCH = train_config["num_epoch"]
CENTER_X = data_config["center_x"]
CENTER_Y = data_config["center_y"]
RADIUS_INNER = data_config["radius_inner"]
RADIUS_OUTER = data_config["radius_outer"]

inner_para1 = CSV.read("./results/star_in 06-13_14:49/paras.csv", DataFrame)
inner_para2 = CSV.read("./results/star_in 06-13_20:40/paras.csv", DataFrame)
outer_para1 = CSV.read("./results/star_out 06-13_16:20/paras.csv", DataFrame)
outer_para2 = CSV.read("./results/star_out 06-13_18:57/paras.csv", DataFrame)
NUM_BIT = config["circuit_info"]["num_qubits"]

circuit = variational_circuit(NUM_BIT, circuit_config["num_layer"]; do_cache = train_config["do_cache"])
observable1 = kron(NUM_BIT, 1 => Z)


include("sharedfunctions.jl")

#%%
acc, c0, c1, c2 = compose_draw(inner_para1[!, :para], inner_para2[!, :para], outer_para1[!, :para], outer_para2[!, :para], :raw, info = "raw_cost")
accg, c0g, c1g, c2g = compose_draw(inner_para1[!, :parag], inner_para2[!, :parag], outer_para1[!, :parag], outer_para2[!, :parag], :guid, info = "guid_cost")



function myscatter(p, x, y, color, label)
	scatter!(p, x, y, label=label, 
	mark = :square, ms = 10.8, markerstrokewidth=0)
end

# scatter!(p, x0, y0, color=:red, label="class0", mark = :square, ms = 10)
# scatter!(p, x1, y1, color=:green, label="class1", mark = :square, ms = 10)
# scatter!(p, x2, y2, color=:blue, label="class2", mark = :square, ms = 10)
function drawsubfigure(c0, c1, c2; kwargs...)
	x0, y0 = unzip(c0)
	x1, y1 = unzip(c1)
	x2, y2 = unzip(c2)
	p = Plots.plot(aspect_ratio=:equal, xlims=(0, 1), ylims=(0, 1); kwargs...)
	myscatter(p, x0, y0, :lightred, "class 0")
	myscatter(p, x1, y1, :blue, "class 1")
	myscatter(p, x2, y2, :green, "class 2")

	draw_circle(p, (CENTER_X, CENTER_Y), RADIUS_INNER)
	draw_circle(p, (CENTER_X, CENTER_Y), RADIUS_OUTER)

	return p
end

p1 = drawsubfigure(c0, c1, c2; title = "classification without guidance")
p2 = drawsubfigure(c0g, c1g, c2g; title = "classification with guidance")
drawLine(p2, (0,0), 1, (0,1), num_interval = 64, 
	color = :black, lw = 2, label = false, linestyle = :dash)
drawLine(p1, (0,0), 1, (0,1), num_interval = 64, 
	color = :black, lw = 2, label = false, linestyle = :dash)
p = Plots.plot(p1, p2, layout = (1, 2), size = (800, 400))
savefig(p, "decision_boundary.pdf")

# 显示图表
display(p, )
# compose_draw(inner_para[!, :parag], outer_para[!, :parag], :guid, info = "guid_cost")

#%% help functions
include("train/train.jl")

StepFunction(x::Real) = x > 0 ? 1 : 0

function compose_draw(inner_para1, inner_para2, outer_para1, outer_para2, guid::Symbol; info = "")
	center = (data_config["center_x"], data_config["center_y"])
	cnt = 0

	circuit1 = dispatch(circuit, inner_para1)
	circuit2 = dispatch(circuit, outer_para1)

	x_lst = 0:1/2^(NUM_BIT÷2):1
	y_lst = 0:1/2^(NUM_BIT÷2):1
	Plots.plot()

	class1 = []
	class2 = []
	class3 = []

	for x in x_lst, y in y_lst

		if guid == :raw
			expectations_in1 = quantum_nodes((x, y), binary_encode, circuit1, [observable1]; is_CUDA = train_config["is_CUDA"])
			dispatch!(circuit1, inner_para2)
			expectations_in2 = quantum_nodes((x, y), binary_encode, circuit1, [observable1]; is_CUDA = train_config["is_CUDA"])
			expectations_in = 1 / 2 .* (expectations_in1 .+ expectations_in2)

			expectations_out1 = quantum_nodes((x, y), binary_encode, circuit2, [observable1]; is_CUDA = train_config["is_CUDA"])
			dispatch!(circuit2, outer_para2)
			expectations_out2 = quantum_nodes((x, y), binary_encode, circuit2, [observable1]; is_CUDA = train_config["is_CUDA"])
			expectations_out = 1 / 2 .* (expectations_out1 .+ expectations_out2)
		else
			expectations_in1 = guided_point_node((x, y), circuit1, [observable1])
			dispatch!(circuit1, inner_para2)
			expectations_in2 = guided_point_node((x, y), circuit1, [observable1])
			expectations_in = 1 / 2 .* (expectations_in1 .+ expectations_in2)

			expectations_out1 = guided_point_node((x, y), circuit2, [observable1])
			dispatch!(circuit2, outer_para2)
			expectations_out2 = guided_point_node((x, y), circuit2, [observable1])
			expectations_out = 1 / 2 .* (expectations_out1 .+ expectations_out2)
			# expectations1 = guided_point_node((x, y), circuit1, [observable1])
			# expectations2 = guided_point_node((x, y), circuit2, [observable1])
		end

		label = sum(StepFunction, vcat(expectations_in, expectations_out))
		true_label = label_double_circle_data((x, y), RADIUS_INNER, RADIUS_OUTER, center)

		if label == true_label
			cnt += 1
		end

		if label == 0
			push!(class1, (x, y))
			mark, color = :circle, :red
		elseif label == 1
			push!(class2, (x, y))
			mark, color = :square, :blue
		else
			push!(class3, (x, y))
			mark, color = :x, :green
		end
		Plots.scatter!([x], [y], m = mark, mc = color, label = false, title = info)
	end

	accuracy = cnt / (length(x_lst) * length(y_lst))

	# draw_circle(center, RADIUS_INNER)
	# draw_circle(center, RADIUS_OUTER)
	# Plots.savefig("decision_boundary_$info.svg")

	return accuracy, class1, class2, class3
end

function draw_circle(p, (center_x, center_y), radius)
	th = 0:0.1:(2*pi)
	x = @. center_x + radius * cos(th)
	x = [x; x[1]]
	y = @. center_y + radius * sin(th)
	y = [y; y[1]]
	da = DataFrame(x = x, y = y)
	Plots.plot!(p, da[:, :x], da[:, :y], color = :black, lw = 2, label = false)
end
