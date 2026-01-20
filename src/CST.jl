"""
    bernstein(r, n, x)

Bernstein Basis Function: `binomial(n, r) .* x .^ r .* (1 .- x) .^ (n .- r)`
"""
function bernstein(r, n, x)
    return binomial(n, r) .* x .^ r .* (1 .- x) .^ (n .- r)
end

"""
    half_cst(coefficients, x, dz, leading_edge_weight; N1=0.5, N2=1.0)

Determine y-coordinates of one side of an airfoil give coeffiecients and x coordinates.

# Arguments
- `coefficients::Vector{Float}` : Kulfan parameters
- `x::Vector{Float}` : x-coordinates (front to back)
- `dz::Float` : Trailing edge gap
- `leading_edge_weight::Float` : Kulfan leading edge modification weight

# Keyword Arguments
- `N1::Float=0.5` : Class function parameter for leading edge
- `N2::Float=1.0` : Class function parameter for trailing edge

# Returns
- `y::Vector{Float}` : y-coordinates
"""
function half_cst(coefficients, x, dz, leading_edge_weight; N1=0.5, N2=1.0)
    nb = length(coefficients) - 1

    # Get class values
    C = @. x^N1 * (1.0 - x)^N2

    # Initialize shape functions
    S = similar(x) .= 0

    # Populate shape functions
    for (i, c) in enumerate(coefficients)
        S += c * bernstein(i - 1, nb, x)
    end

    # determine nominal y-values
    y = @. C * S + x * dz

    # Kulfan leading edge modification
    y .+= leading_edge_weight .* x .* max.(1.0 .- x, 0) .^ (length(coefficients) + 0.5)

    return y
end

"""
    cst(x, p, x_split_id; N1=0.5, N2=1.0)

Determine y-coordinates of one side of an airfoil give coeffiecients and x coordinates.

# Arguments
- `x::Vector{Float}` : x-coordinates (concatenated top and bottom)
- `p::Vector{Float}` : parameters including Kulfan parameters, leading edge weight, and trailing edge gap.
- `x_split_id::Int`  : id for splitting the upper and lower coordinates

# Keyword Arguments
- `N1::Float=0.5` : Class function parameter for leading edge
- `N2::Float=1.0` : Class function parameter for trailing edge

# Returns
- `y::Vector{Float}` : y-coordinates associated with the x-coordinates
"""
function cst(x, p, x_split_id; N1=0.5, N2=1.0)
    weights..., leading_edge_weight, dz = p

    N = convert(Int, length(weights) / 2)
    weights_upper = weights[1:N]
    weights_lower = weights[(N + 1):end]

    x_upper = x[1:x_split_id]
    x_lower = x[(x_split_id + 1):end]

    y_upper = half_cst(
        weights_upper, reverse(x_upper), dz / 2, leading_edge_weight; N1, N2
    )
    y_lower = half_cst(
        weights_lower, x_lower, -dz / 2, leading_edge_weight; N1, N2
    )

    return [reverse(y_upper); y_lower]
end

"""
    cst_te0(x, p, x_split_id; N1=0.5, N2=1.0)

Determine y-coordinates of one side of an airfoil give coeffiecients and x coordinates. Require a zero gap trailing edge

# Arguments
- `x::Vector{Float}` : x-coordinates (concatenated top and bottom)
- `p::Vector{Float}` : parameters including Kulfan parameters, leading edge weight, and trailing edge gap.
- `x_split_id::Int`  : id for splitting the upper and lower coordinates

# Keyword Arguments
- `N1::Float=0.5` : Class function parameter for leading edge
- `N2::Float=1.0` : Class function parameter for trailing edge

# Returns
- `y::Vector{Float}` : y-coordinates associated with the x-coordinates
"""
function cst_te0(x, p, x_split_id; N1=0.5, N2=1.0)
    return cst(x, [p; 0], x_split_id; N1, N2)
end

"""
    get_kulfan_parameters(coordinates; n_coefficients=8, N1=0.5, N2=1.0)

Use least squares to approximate kulfan parameters generating the input coordinates.

# Arguments
- `coordinates::Matrix{Float}` : [x y] coordinates for which to find the Kulfan paramters

# Keyword Arguments
- `n_coefficients::Int=8` : Number of coefficients to use per side
- `N1::Float=0.5` : Class function parameter for leading edge
- `N2::Float=1.0` : Class function parameter for trailing edge

# Returns
- `kulfan_parameters::KulfanParameters` : a KulfanParameters object containing the Kulfan parameters.
"""
function get_kulfan_parameters(coordinates; n_coefficients=8, N1=0.5, N2=1.0)
    # Split
    coords_upper, coords_lower = split_upper_lower(coordinates)

    xu = @view coords_upper[:, 1]
    yu = @view coords_upper[:, 2]
    xl = @view coords_lower[:, 1]
    yl = @view coords_lower[:, 2]

    # Get trailing edge gap
    te_z = yu[1] - yl[end]

    # Fit coordintes
    fit = LsqFit.curve_fit(
        (x, p) -> cst(x, p, length(xu); N1, N2),
        [xu; xl],
        [yu; yl],
        [ones(2 * n_coefficients + 1); te_z];
        autodiff=:forwarddiff,
    )

    # If you get a negative trailing edge gap, solve again requiring a zero gap.
    if fit.param[end] < 0.0
        # Fit coordintes
        fit = LsqFit.curve_fit(
            (x, p) -> cst_te0(x, p, length(xu); N1, N2),
            [xu; xl],
            [yu; yl],
            ones(2 * n_coefficients + 1);
            autodiff=:forwarddiff,
        )

        # Organize Outputs
        cst_upper = fit.param[1:n_coefficients]
        cst_lower = fit.param[(n_coefficients + 1):(2 * n_coefficients)]
        cst_LE = fit.param[end]
        cst_TE = 0.0
    else
        # Organize Outputs
        cst_upper = fit.param[1:n_coefficients]
        cst_lower = fit.param[(n_coefficients + 1):(2 * n_coefficients)]
        cst_LE = fit.param[end - 1]
        cst_TE = fit.param[end]
    end

    # Return
    return KulfanParameters(cst_upper, cst_lower, [cst_LE], [cst_TE])
end
