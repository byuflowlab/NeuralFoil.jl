using Documenter
using NeuralFoil
using BenchmarkTools

# DocMeta.setdocmeta!(NeuralFoil, :DocTestSetup, :(using NeuralFoil); recursive=true)

makedocs(;
    modules=[NeuralFoil],
    authors="Judd Mehr",
    sitename="NeuralFoil.jl",
    format=Documenter.HTML(;
        repolink="https://github.com/byuflowlab/NeuralFoil.jl/blob/{commit}{path}#L{line}",
        edit_link="main",
    ),
    pages=[
        "Intro" => "index.md",
        "Quick Start" => "tutorial.md",
        "Advanced Usage" => "advanced.md",
        "API Reference" => "reference.md",
    ],
    warnonly=Documenter.except(:linkcheck, :footnote),
)

deploydocs(; repo="https://github.com/byuflowlab/NeuralFoil.jl", devbranch="main")
