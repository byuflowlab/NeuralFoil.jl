"""
    swish(x)

Swish activation function: x ./ (1.0 .+ exp.(-x))
"""
function swish(x)
    return x ./ (1.0 .+ exp.(-x))
end

"""
    net(x, net_cache)

Neural net

# Arguments
- `x::Matrix{Float}` : network inputs
- `net_cache::NetParameters` : network parameters

# Returns
- `y::Matrix{Float}` : network outputs
"""
function net(x, net_cache)

    # Rename for Convenience
    (; weights, biases) = net_cache

    # Neural Net
    for (i, (W, b)) in enumerate(zip(weights, biases))
        x = W * x
        x .+= b
        if i != length(weights)
            x = swish(x)
        end
    end

    return x
end
