module NeuralFoil

using NPZ
using LsqFit

include("types.jl")
include("utilities.jl")
include("CST.jl")
include("net.jl")
include("analyze.jl")
include("naca.jl")

export NetParameters, KulfanParameters
export get_aero_from_dat_file, get_aero_from_coordinates, get_aero_from_kulfan_parameters

end
