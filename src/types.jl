"""
    KulfanParameters

# Fields

- `upper_weights::Vector{Float}` : upper side weights (length 8)
- `lower_weights::Vector{Float}` : lower side weights (length 8)
- `leading_edge_weight::Vector{Float}` : weight for leading edge thickness (length 1)
- `TE_thickness::Vector{Float}` : trailing edge thickness (length 1)

Note: for optimization purposes in Julia, it is convenient to have all fields as arrays.
"""
@kwdef struct KulfanParameters{TUW,TLW,TLE,TTE}
    upper_weights::AbstractVector{TUW}
    lower_weights::AbstractVector{TLW}
    leading_edge_weight::AbstractVector{TLE}
    TE_thickness::AbstractVector{TTE}
end

function KulfanParameters(uw, lw, lew, tet)
    return KulfanParameters(
        uw, lw, isscalar(lew) ? [lew] : lew, isscalar(tet) ? [tet] : tet
    )
end

struct NetParameters{Tb,TM,TV,TW}
    mean_inputs_scaled::TV
    cov_inputs_scaled::TM
    inv_cov_inputs_scaled::TM
    weights::TW
    biases::Tb
end

"""
    NetParameters(; model_size="xlarge")

    Constructor for NetParameters type.

# Keyword Arguments
- `model_size::String="xlarge"` : NeuralFoil model size, choose from:
  - "xxsmall"
  - "xsmall"
  - "small"
  - "medium"
  - "large"
  - "xlarge"
  - "xxlarge"
  - "xxxlarge"

# Returns
- `net_cache::NetParameters` : NetParameters method object with fields:

## NetParameters Object Fields:
- `mean_inputs_scaled::Vector{Float}` : from saved model values
- `cov_inputs_scaled::Matrix{Float}`: from saved model values
- `inv_cov_inputs_scaled::Matrix{Float}`: from saved model values
- `weights::Vector{Vector{Float}}}`: from saved model values
- `biases::Vector{Vector{Float}}}`: from saved model values
"""
function NetParameters(; model_size="xlarge")
    scaled_input_distribution = NPZ.npzread(
        joinpath(@__DIR__, "..", "data", "scaled_input_distribution.npz")
    )
    nn_params = NPZ.npzread(joinpath(@__DIR__, "..", "data", "nn-" * model_size * ".npz"))

    unique_keys = unique(sort([parse(Int, split(key, ".")[2]) for key in keys(nn_params)]))

    return NetParameters(
        scaled_input_distribution["mean_inputs_scaled"],
        scaled_input_distribution["cov_inputs_scaled"],
        scaled_input_distribution["inv_cov_inputs_scaled"],
        [nn_params["net.$(id).weight"] for id in unique_keys],
        [nn_params["net.$(id).bias"] for id in unique_keys],
    )
end

"""
    NeuralOutputs

# Fields
- `analysis_confidence::Vector` : confidence factor reported by NeuralFoil
- `cl::Vector` : lift coefficients
- `cd::Vector` : drag coefficients
- `cm::Vector` : moment coefficients
- `top_xtr::Vector` : laminar to turbulent transition location of top surface
- `bot_xtr::Vector` : laminar to turbulent transition location of bottom surface
- `upper_bl_ue_over_vinf::Matrix` : upper boundary layer normalized surface velocity
- `upper_theta::Matrix` : upper boundary layer momentum thickness
- `upper_H::Matrix` : upper boundary layer shape factor
- `lower_bl_ue_over_vinf::Matrix` : lower boundary layer normalized surface velocity
- `lower_theta::Matrix` : lower boundary layer momentum thickness
- `lower_H::Matrix` : lower boundary layer shape factor
"""
@kwdef struct NeuralOutputs{TM,TV}
    analysis_confidence::TV
    cl::TV
    cd::TV
    cm::TV
    top_xtr::TV
    bot_xtr::TV
    upper_bl_ue_over_vinf::TM
    upper_theta::TM
    upper_H::TM
    lower_bl_ue_over_vinf::TM
    lower_theta::TM
    lower_H::TM
end
