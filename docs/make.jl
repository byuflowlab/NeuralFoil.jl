using NeuralFoil
using Documenter

DocMeta.setdocmeta!(NeuralFoil, :DocTestSetup, :(using NeuralFoil); recursive=true)

makedocs(;
    modules=[NeuralFoil],
    authors="Judd Mehr",
    sitename="NeuralFoil.jl",
    format=Documenter.HTML(;
        canonical="https://byuflowlab.github.io/NeuralFoil.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/byuflowlab/NeuralFoil.jl",
    devbranch="main",
)
