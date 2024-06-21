include("cost.jl")
include("symmetry_observable.jl") 
include("gradient.jl")

# best_cst = Inf
# best_para = deepcopy(param)
function step!(param::Vector, cost_lst::Vector, circuit::AbstractBlock, opt, guided::Symbol)
    data_batch = shuffle(data)[1:train_config["batch_size"], :]
    grad = let 
        func(paras::Vector) = begin
            dispatch!(circuit, paras)
            cost(data_batch, circuit, observable1, observable2, guided)
        end
        gradient(func, param, cost_lst)
    end
    Optimisers.update!(opt, param, grad)
    push!(cost_lst, cost(data_batch, circuit, observable1, observable2, guided))
    # if guided == :raw
    #     cst < best_cst
    #     best_cst = cst
    #     best_para = deepcopy(param)
    # end
end

# function stepg!(param::Vector, cost_lst::Vector, circuit::AbstractBlock, opt)
#     grad = let 
#         data_batch = shuffle(data)[1:train_config["batch_size"], :]
#         func(paras::Vector) = begin
#             dispatch!(circuit, paras)
#             cost(data_batch, circuit, observable1, observable2)
#         end
#         gradient(func, param, cost_lst)
#     end
#     Optimisers.update!(opt, param, grad)
# end