StepFunction(x::Real) = x > 0 ? 1 : 0

function draw(param::Vector; info = "")
    center = (data_config["center_x"], data_config["center_y"])
    radius_inner = data_config["radius_inner"]
    radius_outer = data_config["radius_outer"]

    cnt = 0

    dispatch!(circuit, param)
    x_lst = 0:1/2^(NUM_BIT÷2):1
    y_lst = 0:1/2^(NUM_BIT÷2):1
    Plots.plot()
    for x in x_lst, y in y_lst
        expectation1, expectation2 = quantum_nodes((x, y), binary_encode, circuit, [observable1, observable2]; is_CUDA = train_config["is_CUDA"])
        label = StepFunction(expectation1) + StepFunction(expectation2)
        true_label = label_data((x, y), radius_inner, radius_outer, center)
        
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

    draw_circle(center, radius_inner)
    draw_circle(center, radius_outer)
    Plots.savefig(RESULT_PATH * "decision_boundary_$info.png")

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
accuracy_guid = draw(paramg; info = "guid_cost")