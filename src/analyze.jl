"""
    get_aero_from_coordinates(
        kulfan_parameters,
        alpha,
        Re;
        n_crit=9.0,
        xtr_upper=1.0,
        xtr_lower=1.0,
        model_size="xlarge",
        net_cache=nothing,
    )

Determine aerodynamic properties of an airfoil given Kulfan parameters.

# Arguments
- `kulfan_parameters::KulfanParameters` : KulfanParameters object containig upper and lower weights, leading edge weight, and trailing edge thickness inputs to the neural net.
- `alpha::Vector` : angles of attack in degrees
- `Re::Vector` : Reynolds numbers

# Keyword Arguments
- `n_crit::Float=9.0` : n for e^n model
- `xtr_upper::Float=1.0` : upper side laminar to turbulent forced transition location
- `xtr_lower::Float=1.0` : lower side laminar to turbulent forced transition location
- `model_size::String="xlarge"` : NeuralFoil model size. Choose from
  - "xxsmall"
  - "xsmall"
  - "small"
  - "medium"
  - "large"
  - "xlarge"
  - "xxlarge"
  - "xxxlarge"
- `net_cache::NetParameters=nothing` : Neural net parameters, if not provided, is generated based on the `model_size`.

# Returns
- `outputs::NeuralOutputs` : NeuralOutputs object containing the outputs of the neural net.
"""
function get_aero_from_kulfan_parameters(
    kulfan_parameters,
    alpha,
    Re;
    n_crit=9.0,
    xtr_upper=1.0,
    xtr_lower=1.0,
    model_size="xlarge",
    net_cache=nothing,
)
    if isnothing(net_cache)
        net_cache = NetParameters(; model_size=model_size)
    end

    # Assemble inputs to run everything all at once
    re_vec = stack(reshape([[reynolds] for _ in alpha, reynolds in Re], :))
    x = stack(
        reshape(
            [
                [
                    kulfan_parameters.upper_weights
                    kulfan_parameters.lower_weights
                    kulfan_parameters.leading_edge_weight[]
                    kulfan_parameters.TE_thickness[] * 50.0
                    sind(2.0 * aoa)
                    cosd(aoa)
                    1.0 - cosd(aoa)^2
                    (log(reynolds) - 12.5) / 3.5
                    (n_crit .- 9.0) / 4.5
                    xtr_upper
                    xtr_lower
                ] for aoa in alpha, reynolds in Re
            ],
            :,
        ),
    )

    x_flipped = stack(
        reshape(
            [
                [
                    -kulfan_parameters.lower_weights
                    -kulfan_parameters.upper_weights
                    -kulfan_parameters.leading_edge_weight[]
                    kulfan_parameters.TE_thickness[] * 50.0
                    -sind(2.0 * aoa)
                    cosd(aoa)
                    1.0 - cosd(aoa)^2
                    (log(reynolds) - 12.5) / 3.5
                    (n_crit .- 9.0) / 4.5
                    xtr_lower
                    xtr_upper
                ] for aoa in alpha, reynolds in Re
            ],
            :,
        ),
    )

    # rename for convenience
    Wb = (; weights=net_cache.weights, biases=net_cache.biases)

    # - Call network (outputs num cases x num outputs) - #
    y = net(x, Wb)
    y_flipped = net(x_flipped, Wb)

    # - Compute confidence values - #
    y[1, :] .-=
        squared_mahalanobis_distance(
            x, net_cache.mean_inputs_scaled, net_cache.inv_cov_inputs_scaled
        ) ./ (2.0 * size(x, 1))
    y_flipped[1, :] .-=
        squared_mahalanobis_distance(
            x_flipped, net_cache.mean_inputs_scaled, net_cache.inv_cov_inputs_scaled
        ) ./ (2.0 * size(x_flipped, 1))

    # - Unflip flipped output - #
    y_unflipped = copy(y_flipped)
    y_unflipped[2, :] .*= -1.0  # CL
    y_unflipped[4, :] .*= -1.0  # CM
    y_unflipped[5, :] .= y_flipped[6, :]   # Top_Xtr
    y_unflipped[6, :] .= y_flipped[5, :]   # Bot_Xtr

    # Switch upper and lower Ret, H
    y_unflipped[7:(7 + 32 * 2 - 1), :] .= y_flipped[(7 + 32 * 3):(7 + 32 * 5 - 1), :]
    y_unflipped[(7 + 32 * 3):(7 + 32 * 5 - 1), :] .= y_flipped[7:(7 + 32 * 2 - 1), :]

    # Switch upper_bl_ue/vinf with lower_bl_ue/vinf
    y_unflipped[(7 + 32 * 2):(7 + 32 * 3 - 1), :] .=
        -y_flipped[(7 + 32 * 5):(7 + 32 * 6 - 1), :]
    y_unflipped[(7 + 32 * 5):(7 + 32 * 6 - 1), :] .=
        -y_flipped[(7 + 32 * 2):(7 + 32 * 3 - 1), :]

    # - Average outputs - #
    y_fused = (y .+ y_unflipped) ./ 2.0
    y_fused[1, :] .= sigmoid.(y_fused[1, :])
    y_fused[5, :] .= clamp.(y_fused[5, :], 0, 1)
    y_fused[6, :] .= clamp.(y_fused[6, :], 0, 1)

    # Set up outputs for return
    N = 32 #hard coded in neuralfoil
    confidence = y_fused[1, :]
    cl = y_fused[2, :] ./ 2.0
    cd = exp.((y_fused[3, :] .- 2.0) .* 2)
    cm = y_fused[4, :] ./ 20.0
    top_xtr = y_fused[5, :]
    bot_xtr = y_fused[6, :]
    upper_bl_ue_over_vinf = y_fused[(7 + N * 2):(7 + N * 3 - 1), :]
    lower_bl_ue_over_vinf = y_fused[(7 + N * 5):(7 + N * 6 - 1), :]
    upper_theta =
        ((10.0 .^ y_fused[7:(7 + N - 1), :]) .- 0.1) ./
        (abs.(upper_bl_ue_over_vinf) .* re_vec)
    upper_H = 2.6 .* exp.(y_fused[(7 + N):(7 + N * 2 - 1), :])
    lower_theta =
        ((10.0 .^ y_fused[(7 + N * 3):(7 + N * 4 - 1), :]) .- 0.1) ./
        (abs.(lower_bl_ue_over_vinf) .* re_vec)
    lower_H = 2.6 .* exp.(y_fused[(7 + N * 4):(7 + N * 5 - 1), :])

    # - Return Outputs - #
    vec_out_size = (length(alpha), length(Re))
    matu_out_size = (length(alpha), size(upper_H, 1), length(Re))
    matl_out_size = (length(alpha), size(lower_H, 1), length(Re))
    return NeuralOutputs(
        reshape(confidence, vec_out_size),
        reshape(cl, vec_out_size),
        reshape(cd, vec_out_size),
        reshape(cm, vec_out_size),
        reshape(top_xtr, vec_out_size),
        reshape(bot_xtr, vec_out_size),
        reshape(upper_bl_ue_over_vinf, matu_out_size),
        reshape(upper_theta, matu_out_size),
        reshape(upper_H, matu_out_size),
        reshape(lower_bl_ue_over_vinf, matl_out_size),
        reshape(lower_theta, matl_out_size),
        reshape(lower_H, matl_out_size),
    )
end

"""
    get_aero_from_coordinates(
        coordinates,
        alpha,
        Re;
        n_crit=9.0,
        xtr_upper=1.0,
        xtr_lower=1.0,
        model_size="xlarge",
        net_cache=nothing,
    )

Determine aerodynamic properties of an airfoil given airfoil coordinates.

Determines Kulfan parameters from coordinates and calls `get_aero_from_kulfan_parameters`.

# Arguments
- `coordinates::Matrix{Float}` : [x y] coordintes (counter-clockwise starting at upper side trailing edge)
- `alpha::Vector` : angles of attack in degrees
- `Re::Vector` : Reynolds numbers

# Keyword Arguments
- `n_crit::Float=9.0` : n for e^n model
- `xtr_upper::Float=1.0` : upper side laminar to turbulent forced transition location
- `xtr_lower::Float=1.0` : lower side laminar to turbulent forced transition location
- `model_size::String="xlarge"` : NeuralFoil model size. Choose from
  - "xxsmall"
  - "xsmall"
  - "small"
  - "medium"
  - "large"
  - "xlarge"
  - "xxlarge"
  - "xxxlarge"
- `net_cache::NetParameters=nothing` : Neural net parameters, if not provided, is generated based on the `model_size`.

# Returns
- `outputs::NeuralOutputs` : NeuralOutputs object containing the outputs of the neural net.
"""
function get_aero_from_coordinates(
    coordinates,
    alpha,
    Re;
    n_crit=9.0,
    xtr_upper=1.0,
    xtr_lower=1.0,
    model_size="xlarge",
    net_cache=nothing,
)
    return get_aero_from_kulfan_parameters(
        get_kulfan_parameters(coordinates),
        alpha,
        Re;
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size,
        net_cache=net_cache,
    )
end

"""
    get_aero_from_dat_file(
        filename,
        alpha,
        Re;
        n_crit=9.0,
        xtr_upper=1.0,
        xtr_lower=1.0,
        model_size="xlarge",
        net_cache=nothing,
    )

Determine aerodynamic properties of an airfoil given in a .dat file.

Reads in coordinates and calls `get_aero_from_coordinates`.

# Arguments
- `filename::String` : name (including path) of Selig style coordinate file
- `alpha::Vector` : angles of attack in degrees
- `Re::Vector` : Reynolds numbers

# Keyword Arguments
- `n_crit::Float=9.0` : n for e^n model
- `xtr_upper::Float=1.0` : upper side laminar to turbulent forced transition location
- `xtr_lower::Float=1.0` : lower side laminar to turbulent forced transition location
- `model_size::String="xlarge"` : NeuralFoil model size. Choose from
  - "xxsmall"
  - "xsmall"
  - "small"
  - "medium"
  - "large"
  - "xlarge"
  - "xxlarge"
  - "xxxlarge"
- `net_cache::NetParameters=nothing` : Neural net parameters, if not provided, is generated based on the `model_size`.

# Returns
- `outputs::NeuralOutputs` : NeuralOutputs object containing the outputs of the neural net.
"""
function get_aero_from_dat_file(
    filename,
    alpha,
    Re;
    n_crit=9.0,
    xtr_upper=1.0,
    xtr_lower=1.0,
    model_size="xlarge",
    net_cache=nothing,
)
    return get_aero_from_coordinates(
        get_coordinates_from_file(filename),
        alpha,
        Re;
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size,
        net_cache=net_cache,
    )
end
