# Public

## Types

```@docs
NeuralFoil.NetParameters
NeuralFoil.KulfanParameters
```

## Methods
```@docs
NeuralFoil.get_aero_from_kulfan_parameters
NeuralFoil.get_aero_from_coordinates
NeuralFoil.get_aero_from_dat_file
```

## Outputs
```@docs
NeuralFoil.NeuralOutputs
```

# Private

## Neural Net Functions

```@docs
NeuralFoil.net
NeuralFoil.swish
```

## CST Functions

```@docs
NeuralFoil.get_kulfan_parameters
NeuralFoil.cst
NeuralFoil.cst_te0
NeuralFoil.half_cst
NeuralFoil.bernstein
```

## NACA 4-Series Functions

```@docs
NeuralFoil.naca4
NeuralFoil.naca4_camber
NeuralFoil.naca4_thickness
```

## Utility Functions

### Neural Net Post-process Utilities

```@docs
NeuralFoil.sigmoid
NeuralFoil.squared_mahalanobis_distance
```

### Airfoil Utilities

```@docs
NeuralFoil.get_coordinates_from_file
NeuralFoil.split_cosine_spacing
NeuralFoil.normalize_coordinates!
NeuralFoil.split_upper_lower
```

# Index

```@index
Modules = [NeuralFoil]
```
