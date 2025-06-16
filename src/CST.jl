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
    y .+= leading_edge_weight .* x .* (1.0 .- x) .^ (length(coefficients) + 0.5)

    return y
end

"""
    cst(x, p; N1=0.5, N2=1.0)

Determine y-coordinates of one side of an airfoil give coeffiecients and x coordinates.

# Arguments
- `x::Vector{Float}` : x-coordinates (concatenated top and bottom)
- `p::Vector{Float}` : parameters including Kulfan parameters, leading edge weight, and trailing edge gap.

# Keyword Arguments
- `N1::Float=0.5` : Class function parameter for leading edge
- `N2::Float=1.0` : Class function parameter for trailing edge

# Returns
- `y::Vector{Float}` : y-coordinates associated with the x-coordinates
"""
function cst(x, p; N1=0.5, N2=1.0)
    # Extract Parameters
    np = Int((length(p) - 2) / 2)
    pu = p[1:np]
    pl = p[(np + 1):(np * 2)]
    leading_edge_weight = p[end - 1]
    dz = p[end]

    # Split halves
    nx = Int(length(x) / 2)
    xu = x[1:nx]
    xl = x[(nx + 1):end]

    # Get y-values
    yu = half_cst(pu, xu, dz / 2.0, leading_edge_weight)
    yl = half_cst(pl, xl, -dz / 2.0, leading_edge_weight)

    # Return
    return [yu; yl]
end

"""
    cst_te0(x, p; N1=0.5, N2=1.0)

Determine y-coordinates of one side of an airfoil give coeffiecients and x coordinates. Require a zero gap trailing edge

# Arguments
- `x::Vector{Float}` : x-coordinates (concatenated top and bottom)
- `p::Vector{Float}` : parameters including Kulfan parameters, leading edge weight, and trailing edge gap.

# Keyword Arguments
- `N1::Float=0.5` : Class function parameter for leading edge
- `N2::Float=1.0` : Class function parameter for trailing edge

# Returns
- `y::Vector{Float}` : y-coordinates associated with the x-coordinates
"""
function cst_te0(x, p; N1=0.5, N2=1.0)
    # Extract Parameters
    np = Int((length(p) - 1) / 2)
    pu = p[1:np]
    pl = p[(np + 1):(np * 2)]
    leading_edge_weight = p[end]

    # Split halves
    nx = Int(length(x) / 2)
    xu = x[1:nx]
    xl = x[(nx + 1):end]

    # Get y-values
    yu = half_cst(pu, xu, 0.0, leading_edge_weight)
    yl = half_cst(pl, xl, 0.0, leading_edge_weight)

    # Return
    return [yu; yl]
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

    # Normalize
    normalize_coordinates!(coordinates)

    # Split
    xu, xl, yu, yl = NeuralFoil.split_upper_lower(
        coordinates[:, 1], coordinates[:, 2]; idx=Int((size(coordinates, 1) + 1) / 2)
    )

    # Get trailing edge gap
    te_z = yu[end] - yl[end]

    # Fit coordintes
    fit = LsqFit.curve_fit(
        NeuralFoil.cst,
        [xu; reverse(xl)],
        [yu; reverse(yl)],
        [0.1 * ones(n_coefficients); -0.1 * ones(n_coefficients); 0.1; te_z];
        autodiff=:forwarddiff,
    )

    # If you get a negative trailing edge gap, solve again requiring a zero gap.
    if fit.param[end] < 0.0
        # Fit coordintes
        fit = LsqFit.curve_fit(
            NeuralFoil.cst_te0,
            [xu; reverse(xl)],
            [yu; reverse(yl)],
            [0.1 * ones(n_coefficients); -0.1 * ones(n_coefficients); 0.1];
            autodiff=:forwarddiff,
        )

        # Organize Outputs
        cst_TE = 0.0
        nc = Int((length(fit.param) - 1) / 2)
        cst_upper = fit.param[1:nc]
        cst_lower = fit.param[(nc + 1):(end - 1)]
        cst_LE = fit.param[end - 1]

    else
        # Organize Outputs
        cst_TE = fit.param[end]
        nc = Int((length(fit.param) - 2) / 2)
        cst_upper = fit.param[1:nc]
        cst_lower = fit.param[(nc + 1):(end - 2)]
        cst_LE = fit.param[end - 1]
    end

    # Return
    return KulfanParameters(cst_upper, cst_lower, [cst_LE], [cst_TE])
end
