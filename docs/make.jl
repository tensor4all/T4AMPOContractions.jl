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
        "Documentation" => "documentation.md",
        "Extensions" => "extensions.md",
    ]
)

deploydocs(;
    repo="github.com/tensor4all/T4AMPOContractions.jl.git",
    devbranch="main",
)
