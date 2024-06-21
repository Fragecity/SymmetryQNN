# loading desired data
data = let 
    DATA_UNBIAS = CSV.read("./data/06-12_13:19/data_unbias.csv", DataFrame)
    DATA_BIAS   = CSV.read("./data/06-12_13:19/data_bias.csv", DataFrame)
    # DATA_UNBIAS = CSV.read("./data/06-12_19:35/data_unbias.csv", DataFrame)
    # DATA_BIAS   = CSV.read("./data/06-12_19:35/data_bias.csv", DataFrame)
    
    num_bias = round(train_config["sample_num"] * train_config["bias"]) |> Int
    num_unbias = train_config["sample_num"] - num_bias

    data_unbias = shuffle(DATA_UNBIAS)[1:num_unbias, :]
    data_bias = shuffle(DATA_BIAS)[1:num_bias, :]

    vcat(data_unbias, data_bias) |> shuffle
end