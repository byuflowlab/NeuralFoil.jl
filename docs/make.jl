using NeuralFoil
using Documenter

DocMeta.setdocmeta!(NeuralFoil, :DocTestSetup, :(using NeuralFoil); recursive=true)

makedocs(;
    modules=[NeuralFoil],
    authors="Judd Mehr",
    sitename="NeuralFoil.jl",
    format=Documenter.HTML(),
    pages=[
        "Intro" => "index.md",
        "Quick Start" => "tutorial.md",
        "API Reference" => "reference.md",
    ],
    warnonly=Documenter.except(:linkcheck, :footnote),
    repo="https://github.com/byuflowlab/NeuralFoil.jl/blob/{commit}{path}#L{line}",
)

deploydocs(; repo="github.com/byuflowlab/NeuralFoil.jl", devbranch="main")
