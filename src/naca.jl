"""
    naca4_thickness(x, maxthick; blunt_te=false)

Compute thickness at a given chord-normalized x-position by NACA 4-series thickness equations.

# Arguments
- `x::Float` : x position along chordlin, markersize=3, markershape=:squaree
- `maxthick::Float` : Maximum thickness value

# Keyword Arguments
- `blunt_te::Bool` : Flag whether trailing edge is blunt or not
"""
function naca4_thickness(x, maxthick; blunt_te=false)

    # change c2 coefficient base on whether trailing edge is to be blunt or not.
    c2 = blunt_te ? 0.3516 : 0.3537

    # return naca4_thickness value
    return 10.0 *
           maxthick *
           (0.2969 * sqrt(x) - 0.1260 * x - c2 * x^2 + 0.2843 * x^3 - 0.1015 * x^4)
end

"""
    naca4_camber(x, max_camber, max_camber_pos)

Compute camber at a given chord-normalized x-position by NACA 4-series camber equations.

# Arguments
- `x::Float` : x position along chordline
- `max_camber::Float64` : Maximum camber value
- `max_camber_pos::Float64` : Position of maximum camber
"""
function naca4_camber(x, max_camber, max_camber_pos)
    if real(max_camber) != 0.0 && real(max_camber_pos) != 0.0
        if x <= real(max_camber_pos)
            zbar = max_camber * (2 * max_camber_pos * x - x^2) / max_camber_pos^2
        else
            zbar =
                max_camber * (1 - 2 * max_camber_pos + 2 * max_camber_pos * x - x^2) /
                (1 - max_camber_pos)^2
        end
    else
        zbar = 0.0
    end
    return zbar
end

"""
    naca4(c=2.0, p=4.0, t=12.0; N=161, x=nothing, blunt_te=false, split=false)

Compute x, z airfoil coordinates for N nodes, based on NACA 4-Series Parameterization.

# Arguments
- `c::Float` : Maximum camber value (percent of chord)
- `p::Float` : Position along chord (in 10ths of chord) where maximum naca4_camber lies
- `t::Float` : Maximum thickness of airfoil in percent chord

# Keyword Arguments
- `N::Int` : Total number of coordinates to use.  This values should be odd, but if not, the number of points returned will be N-1.
- `x::AbstractArray{Float}` : x-coordinates (cosine spaced coordinates used by default)
- `blunt_te::Bool` : Flag whether trailing edge is blunt or not
- `split::Bool` : Flag wheter to split into upper and lower halves.

# Returns
If `split` == false:
 - `x::AbstractArray{Float}` : Vector of x coordinates, clockwise from trailing edge.
 - `z::AbstractArray{Float}` : Vector of z coordinates, clockwise from trailing edge.
If `split` == true:
 - `xl::AbstractArray{Float}` : Vector of lower half of x coordinates from trailing edge to leading edge.
 - `xu::AbstractArray{Float}` : Vector of upper half of x coordinates from leading edge to trailing edge.
 - `zl::AbstractArray{Float}` : Vector of lower half of z coordinates from trailing edge to leading edge.
 - `zu::AbstractArray{Float}` : Vector of upper half of z coordinates from leading edge to trailing edge.
"""
function naca4(c=2.0, p=4.0, t=12.0; N=161, x=nothing, blunt_te=false, split=false)

    # get x coordinates
    N = ceil(Int, N / 2)
    if isnothing(x)
        x = split_cosine_spacing(N)
    end

    #naca digits
    max_camber = c / 100.0
    max_camber_pos = p / 10.0
    maxthick = t / 100.0

    #initialize arrays
    TF = promote_type(typeof(c), eltype(x))
    zu = zeros(TF, N) #upper z values
    zl = zeros(TF, N) #lower z values

    #--Calculate z-values--#
    #naca4_thickness distribution
    T = naca4_thickness.(x, Ref(maxthick); blunt_te=blunt_te)

    #naca4_camber distribution
    zbar = naca4_camber.(x, Ref(max_camber), Ref(max_camber_pos))

    #z-positions at chordwise stations
    zl = @. zbar - T / 2
    zu = @. zbar + T / 2

    if split
        return reverse(x), x, reverse(zu), zl
    else
        return [reverse(x); x[2:end]], [reverse(zu); zl[2:end]]
    end
end
