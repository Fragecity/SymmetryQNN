using JSON, Dates, CSV, DataFrames, FilePaths
using Logging, LoggingExtras
using Yao, Yao.EasyBuild, CUDA
import Optimisers, Plots
include("./utils/Utils.jl")
using .Utils

#%% main
include("train/read_config.jl")
include("train/Training.jl")
mkpath(dirname(RESULT_PATH))
logger = FileLogger(RESULT_PATH * "logfile.log")

circuit = variational_circuit(NUM_BIT, circuit_config["num_layer"]; do_cache=train_config["do_cache"])
param = collect(rand(nparameters(circuit)))
paramg = deepcopy(param)
opt = Optimisers.setup(Optimisers.ADAM(train_config["learning_rate"]), param)
optg = Optimisers.setup(Optimisers.ADAM(train_config["learning_rate"]), param)

cst_lst = []
cstg_lst = []


@info "-----------Start training-------------"
num_epoch = train_config["num_epoch"]
t1 = now()
for i in 1:num_epoch
    @info "Epoch $i / $num_epoch"
    step!(param, cst_lst, circuit, opt, :raw)
    # push!(cst_lst, )
    # step!(paramg, cstg_lst, circuit, opt, :guid)
end
t2 = now()
@info " -----------Training finished-------------"

#  save and draw
include("train/draw.jl")
Plots.plot(cst_lst, label="raw" )
Plots.plot!(cstg_lst, label="guid" )
run_time = round(t2 - t1, Minute)

CSV.write(RESULT_PATH * "paras.csv", 
    DataFrame(para = param, parag = paramg))
# CSV.write(RESULT_PATH * "costs.csv", 
#     DataFrame(cost = cst_lst, costg = cstg_lst))
with_logger(logger) do
    @info """
    qubits number: $NUM_BIT
    layer number: $(circuit_config["num_layer"])
    samples number: $(data_config["num_data"])
    learning rate: $(train_config["learning_rate"])
    epoch number: $num_epoch
    
    run time: $run_time 
    accuracy: $accuracy_raw
    guided accuracy: $accuracy_guid
    """
end


