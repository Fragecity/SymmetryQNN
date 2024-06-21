function gradient(f::Function, x::Vector; callback::Function, ϵ = 1e-8)
    gradients = zero(x)
    for i in eachindex(x)
        dxᵢ = zero(x)
        dxᵢ[i] = ϵ
        fx = f(x+dxᵢ)
        df_dxᵢ = (fx - f(x-dxᵢ))/2ϵ
        gradients[i] = df_dxᵢ

        # callback(fx)
    end
    return gradients
end

gradient(f::Function, x::Vector, callback::Vector) = let 
    cb(val::Float64) = push!(callback, val)
    gradient(f, x; callback = cb)
end

# f(x) = sum(x.^2)

# gradient(f, [1.0, 2.0, 3.0], [])