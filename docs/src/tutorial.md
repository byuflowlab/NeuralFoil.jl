# Quick Start

## Analyze From Coordinates

```@setup coords
using NeuralFoil
```

```@example coords
x, y = NeuralFoil.naca4()
alpha = 5.0
Re = 1e6

ouputs = get_aero_from_coordinates([x y], alpha, Re; model_size="xlarge")
```

## Analyze From .dat File

```@julia
filename = "naca_2412.dat"
alpha = 5.0
Re = 1e6

ouputs = get_aero_from_file(filename, alpha, Re; model_size="xlarge")
```

