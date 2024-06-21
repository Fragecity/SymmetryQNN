include("train/prepare.jl")

include("train/setup.jl")

# main
include("train/loadata.jl")
include("train/train.jl")

@info "-----------Start training-------------"
t1 = now()
for i in 1:NUM_EPOCH
    @info "Epoch $i / $NUM_EPOCH"
    step!(param, opt; cost_lst = cst_lst, guided= :raw)
    step!(paramg, opt; cost_lst = cstg_lst, guided= :guided)
end

t2 = now()
@info " -----------Training finished-------------"

include("train/draw.jl")