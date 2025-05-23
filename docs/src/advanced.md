# Advanced Usage

## Caching Neural Net Parameters

Caching the neural net parameters beforehand skips the process of reading in the parameter files when calling one of the analysis functions. This is particularly helpful in optmization settings where the analysis will be called many times.

```@setup cache
using NeuralFoil
```

```@example cache
using BenchmarkTools

x, y = NeuralFoil.naca4()
coordinates = [x y]
alpha = range(-5,10,step=1.0)
Re = 1e6
model_size = "xlarge"

# Without Caching
t = @benchmark get_aero_from_coordinates(coordinates, alpha, Re; model_size=model_size)
```

```@example cache
# With Caching
net_cache = NetParameters(;model_size=model_size)

t = @benchmark get_aero_from_coordinates(coordinates, alpha, Re; net_cache=net_cache)
```

As we can see, caching the network parameters is helpful if the analysis is to be called many times.
