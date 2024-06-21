using CSV, DataFrames
using Plots, DrawFuncs

df = CSV.read("./JuliaSupport/data/data_bias.csv", DataFrame) 


category0 = subset(df, :labels => l -> l .== 0 )
category1 = subset(df, :labels => l -> l .== 1 )
category2 = subset(df, :labels => l -> l .== 2 )

#%%
p = Plots.plot(aspect_ratio=:equal, xlims=(0, 1), ylims=(0, 1), title = "Training data")
scatter!(p, category0[1:130, :x], category0[1:130, :y], label = "class 0", markerstrokewidth=0)
scatter!(p, category1[1:380, :x], category1[1:380, :y], label = "class 1", markerstrokewidth=0)
scatter!(p, category2[1:500, :x], category2[1:500, :y], label = "class 2", markerstrokewidth=0)

drawLine(p, [0, 0], 1, (0,1);
        color = :black, lw = 2, label = false)

drawCircle(p, (0.5, 0.5), 0.2; color = :black, lw = 2, label = false)
drawCircle(p, (0.5, 0.5), 0.4; color = :black, lw = 2, label = false)


savefig(p, "./JuliaSupport/results/training_data.svg")