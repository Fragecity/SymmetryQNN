using JSON, Dates, CSV, DataFrames, FilePaths
using Logging, LoggingExtras
using Random: shuffle
import Optimisers, Plots


using Yao, Yao.EasyBuild, CUDA

SAVE_PATH = "results/" * Dates.format(now(), "mm-dd_HH:MM" ) * "/"
mkpath(dirname(SAVE_PATH))