using Plots

include("../sharedfunctions.jl")

function generate_datas(label_function::Function, num_data::Int, args...; bias::Bool = false, x_range = 1.0, y_range = 1.0)
	xs = []
	ys = []
	labels = []

	while length(xs) < num_data
		(x, y) = rand(2) .* [x_range, y_range]
		(!bias || x ≥ y) && begin
			push!(xs, x)
			push!(ys, y)
			push!(labels, label_function((x,y), args...))
		end
	end

	return DataFrame("x" => xs, "y" => ys, "labels" => labels)
end


function plot_data(df::DataFrame)
    colors = [:blue, :red, :green, :yellow, :purple, :orange, :cyan, :magenta, :brown, :pink, :gray, :olive]
    marks = [:circle, :square, :xcross, :diamond, :hexagon, :cross, :star5, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon]
    p = Plots.plot()
    for (idx, label) in Set(df[!, :labels]) |> enumerate
        category = subset(df, :labels => l -> l .== label )
        scatter!(p, category[!, :x], category[!, :y], label = label, color = colors[idx], m=marks[idx])
    end
    Plots.pdf(p, SAVE_PATH*"data.pdf")

end