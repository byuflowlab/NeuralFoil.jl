# - Functions to check if the input is a scalar - #
import Base.BroadcastStyle
"""
    isscalar(x::T) where {T} = isscalar(T)
    isscalar(::Type{T}) where {T} = BroadcastStyle(T) isa Broadcast.DefaultArrayStyle{0}

Determines if the input is a scalar. Note that `Base.BroadcastStyle` is imported.
"""
isscalar(x::T) where {T} = isscalar(T)
isscalar(::Type{T}) where {T} = BroadcastStyle(T) isa Broadcast.DefaultArrayStyle{0}

#---------------------------------#
#       Neural Net Utilities      #
#---------------------------------#

"""
    squared_mahalanobis_distance(x::AbstractMatrix)

Computes the squared Mahalanobis distance of a set of points from the training data.

# Arguments:
- `x::Matrix`: Query point in the input latent space. Shape: (N_cases, N_inputs) For non-vectorized queries, N_cases=1.
- `mean_inputs_scaled::Vector` : from scaled_input_distribution
- `inv_cov_inputs_scaled::Matrix` : from scaled_input_distribution

# Returns:
- `sqmd::Vector` : The squared Mahalanobis distance. Shape: (N_cases,)
"""
function squared_mahalanobis_distance(x, mean_inputs_scaled, inv_cov_inputs_scaled)
    x_minus_mean = (x .- mean_inputs_scaled)'
    return sum((x_minus_mean * inv_cov_inputs_scaled) .* x_minus_mean; dims=2)
end

"""
    sigmoid(x; ln_eps=log(10.0 / floatmax(Float64)))

Sigmoid function: 1.0 / (1.0 + exp(-x))
"""
function sigmoid(x; ln_eps=log(10.0 / floatmax(Float64)))
    x_clipped = clamp.(x, ln_eps, -ln_eps)
    return 1.0 ./ (1.0 .+ exp.(-x_clipped))
end

#---------------------------------#
#        Airfoil Utilities        #
#---------------------------------#

"""
    get_coordinates_from_file(filename)

Parse Selig style dat file into [x y] matrix of coordinates.
"""
function get_coordinates_from_file(filename)
    coordinates = Float64[]

    open(filename, "r") do io
        readline(io)  # Skip header
        for line in eachline(io)
            s = split(strip(line))
            if length(s) == 2
                append!(coordinates, parse.(Float64, s))
            end
        end
    end

    return permutedims(reshape(coordinates, 2, :))
end

"""
    split_cosine_spacing(N::Integer=160)

Returns cosine spaced x coordinates from 0 to 1.

# Arguments
- `N::Integer` : Number of points.

# Returns
- `x::AbstractArray{Float}` : cosine spaced x-coordinates, starting at 0.0 ending at 1.0.
"""
function split_cosine_spacing(N::Integer=80)
    return [0.5 * (1 - cos(pi * (i - 1) / (N - 1))) for i in 1:N]
end

"""
    normalize_coordinates!(coordinates)

Normalize airfoil to unit chord and shift leading edge to zero. Adjusts coordinates in place.

# Arguments:
- `coordinates::Array{Float}` : Array of [x y] coordinates
"""
function normalize_coordinates!(coordinates)
    x = @view(coordinates[:, 1])
    y = @view(coordinates[:, 2])

    # get current chord length
    chord = maximum(x) - minimum(x)

    # shift to zero
    x[:] .-= minimum(x)

    # normalize chord
    x[:] ./= chord

    # scale y coordinates to match
    y[:] ./= chord

    return coordinates
end

"""
    split_upper_lower(x, y; idx::Integer=nothing)

Split the upper and lower halves of the airfoil coordinates.

Assumes leading edge point is at first minimum x value if `idx` is not provided.
Returns the upper and lower coordinates each with the leading edge point.
Assumes airfoil is defined clockwise starting at the trailing edge.

# Arguments:
 - `x::AbstractArray{Float}` : Vector of x coordinates
 - `y::AbstractArray{Float}` : Vector of y coordinates

# Keyword Arguments:
 - `idx::Integer` : optional index at which to split the coordinates

# Returns:
 - `xl::AbstractArray{Float}` : Vector of lower half of x coordinates
 - `xu::AbstractArray{Float}` : Vector of upper half of x coordinates
 - `yl::AbstractArray{Float}` : Vector of lower half of y coordinates
 - `yu::AbstractArray{Float}` : Vector of upper half of y coordinates

"""
function split_upper_lower(x, y; idx=nothing)

    # get half length of geometry coordinates
    if isnothing(idx)
        _, idx = findmin(x)
    end

    return x[1:idx], x[idx:end], y[1:idx], y[idx:end]
end
