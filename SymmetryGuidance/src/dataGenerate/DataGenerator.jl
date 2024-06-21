using DataFrames, CSV, JSON, Dates
using Logging, LoggingExtras


include("utils.jl")

# 读取数据生成配置
data_config = JSON.parsefile("./src/config.json")["data_info"]
SAVE_PATH = "./data/" * Dates.format(now(), "mm-dd_HH:MM" ) * "/"
mkpath(dirname(SAVE_PATH))

# 生成数据

radius = data_config["radius_outer"]

df_unbias = generate_datas(label_circ_data, data_config["num_data"],  
	radius, (data_config["center_x"], data_config["center_y"]),
	bias = false
)

df_bias = generate_datas(label_circ_data, data_config["num_data"],  
	radius, (data_config["center_x"], data_config["center_y"]),
	bias = true
)


CSV.write(SAVE_PATH * "data_unbias.csv", df_unbias)
CSV.write(SAVE_PATH * "data_bias.csv", df_bias)

# 记录日志
logger = FileLogger(SAVE_PATH * "logfile.log")
with_logger(logger) do
	@info """
	The number of samples are $(data_config["num_data"])
	The range of data is [0, $(data_config["x_range"])] 

	The center of the circle is ($(data_config["center_x"]), $(data_config["center_y"]))
	The radius is $(radius)
	
	"""
end

#%% 可视化

plot_data(df_unbias)
# plot_data(df_bias)
