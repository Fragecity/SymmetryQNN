using Random: shuffle

config = JSON.parsefile("./JuliaSupport/config.json")
data_config = config["data_info"]
circuit_config = config["circuit_info"]
train_config = config["train_info"]

RESULT_PATH = "./JuliaSupport/results/" * Dates.format(now(), "mm-dd_HH:MM" ) * "/"
NUM_BIT = circuit_config["num_qubits"]
DATA_UNBIAS = CSV.read("./JuliaSupport/data/data_unbias.csv", DataFrame)
DATA_BIAS   = CSV.read("./JuliaSupport/data/data_bias.csv", DataFrame)



data = let 
    num_bias = round(train_config["sample_num"] * train_config["bias"]) |> Int
    num_unbias = train_config["sample_num"] - num_bias

    data_unbias = shuffle(DATA_UNBIAS)[1:num_unbias, :]
    data_bias = shuffle(DATA_BIAS)[1:num_bias, :]

    vcat(data_unbias, data_bias) |> shuffle
end
