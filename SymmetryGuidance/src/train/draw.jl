include("../sharedfunctions.jl")
StepFunction(x::Real) = x > 0 ? 1 : 0

function draw(param::Vector; info = "")
    center = (data_config["center_x"], data_config["center_y"])
    cnt = 0

    dispatch!(circuit, param)
    x_lst = 0:1/2^(NUM_BIT÷2):1
    y_lst = 0:1/2^(NUM_BIT÷2):1
    Plots.plot()
    for x in x_lst, y in y_lst
        expectations = quantum_nodes((x, y), binary_encode, circuit, [observable1]; is_CUDA = train_config["is_CUDA"])
        label = sum(StepFunction, expectations)
        true_label = label_circ_data((x, y), RADIUS_INNER, center)
        # true_label = label_circ_data((x, y), RADIUS_OUTER, center)
        
        if label == true_label
            cnt += 1
        end

        if label == 0
            mark, color = :circle, :red
        elseif label == 1
            mark, color = :square, :blue
        else
            mark, color = :x, :green
        end
        Plots.scatter!([x], [y], m=mark, mc=color, label=false, title=info)
    end

    accuracy = cnt / (length(x_lst) * length(y_lst))

    draw_circle(center, RADIUS_INNER)
    # draw_circle(center, RADIUS_OUTER)
    Plots.savefig(SAVE_PATH * "decision_boundary_$info.svg")

    return accuracy
end

function drawg(param::Vector; info = "")
    center = (data_config["center_x"], data_config["center_y"])
    cnt = 0

    dispatch!(circuit, param)
    x_lst = 0:1/2^(NUM_BIT÷2):1
    y_lst = 0:1/2^(NUM_BIT÷2):1
    Plots.plot()
    for x in x_lst, y in y_lst
        expectations = guided_point_node((x, y), circuit, [observable1])
        label = sum(StepFunction, expectations)
        true_label = label_circ_data((x, y), RADIUS_INNER, center)
        # true_label = label_circ_data((x, y), RADIUS_OUTER, center)
        
        if label == true_label
            cnt += 1
        end

        if label == 0
            mark, color = :circle, :red
        elseif label == 1
            mark, color = :square, :blue
        else
            mark, color = :x, :green
        end
        Plots.scatter!([x], [y], m=mark, mc=color, label=false, title=info)
    end

    accuracy = cnt / (length(x_lst) * length(y_lst))

    draw_circle(center, RADIUS_INNER)
    # draw_circle(center, RADIUS_OUTER)
    Plots.savefig(SAVE_PATH * "decision_boundary_$info.svg")

    return accuracy
end

function draw_circle((center_x, center_y), radius)
        th = 0:0.1:(2*pi)
        x = @. center_x + radius*cos(th)
        x = [x; x[1]]
        y = @. center_y + radius*sin(th)
        y = [y; y[1]]
        da = DataFrame(x=x, y=y)
        Plots.plot!(da[:,:x], da[:,:y], color=:black, lw = 2, label=false)
end

accuracy_raw = draw(param; info = "raw_cost")
accuracy_guid = drawg(paramg; info = "guid_cost")
Plots.plot(cst_lst, label = "raw_cost")
Plots.plot!(cstg_lst, label = "guid_cost")
savefig(SAVE_PATH * "costs.svg")

run_time = round(t2 - t1, Minute)


CSV.write(SAVE_PATH * "paras.csv", 
    DataFrame(para = param, parag = paramg))
CSV.write(SAVE_PATH * "costs.csv", 
    DataFrame(cost = cst_lst, costg = cstg_lst))


logger = FileLogger(SAVE_PATH * "logfile.log")
run_time = round(t2 - t1, Minute)
    with_logger(logger) do
    @info """
    qubits number: $NUM_BIT
    layer number: $(circuit_config["num_layer"])
    samples number: $(data_config["num_data"])
    learning rate: $(train_config["learning_rate"])
    epoch number: $NUM_EPOCH
    
    run time: $run_time 
    accuracy: $accuracy_raw
    guided accuracy: $accuracy_guid
    """
end