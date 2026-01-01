using T4AMPOContractions
using Documenter

DocMeta.setdocmeta!(T4AMPOContractions, :DocTestSetup, :(using T4AMPOContractions); recursive=true)

makedocs(;
    modules=[T4AMPOContractions],
    authors="Ritter.Marc <Ritter.Marc@physik.uni-muenchen.de> and contributors",
    sitename="T4AMPOContractions.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/T4AMPOContractions.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md",
        "API Reference" => "documentation.md",
        "Examples" => "extensions.md",
    ],
    checkdocs=:none,
    linkcheck=false,
    warnonly=[:cross_references]
)

deploydocs(;
    repo="github.com/tensor4all/T4AMPOContractions.jl.git",
    devbranch="main",
)
