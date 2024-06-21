config = JSON.parsefile("./src/config.json")
data_config = config["data_info"]
circuit_config = config["circuit_info"]
train_config = config["train_info"]

# Frequently used constants
NUM_BIT = circuit_config["num_qubits"]
NUM_EPOCH = train_config["num_epoch"]
CENTER_X = data_config["center_x"]
CENTER_Y = data_config["center_y"]
RADIUS_INNER = data_config["radius_inner"]
RADIUS_OUTER = data_config["radius_outer"]

# creating circuits
circuit = variational_circuit(NUM_BIT, circuit_config["num_layer"]; do_cache=train_config["do_cache"])
param = collect(rand(nparameters(circuit)))
paramg = deepcopy(param)

opt = Optimisers.setup(Optimisers.ADAM(train_config["learning_rate"]), param)
optg = Optimisers.setup(Optimisers.ADAM(train_config["learning_rate"]), param)

cst_lst = []
cstg_lst = []