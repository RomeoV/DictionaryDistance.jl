using DictionaryDistance
using Documenter

DocMeta.setdocmeta!(DictionaryDistance, :DocTestSetup, :(using DictionaryDistance); recursive=true)

makedocs(;
    modules=[DictionaryDistance],
    authors="Romeo Valentin <romeov@stanford.edu> and contributors",
    sitename="DictionaryDistance.jl",
    format=Documenter.HTML(;
        canonical="https://RomeoV.github.io/DictionaryDistance.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RomeoV/DictionaryDistance.jl",
    devbranch="master",
)
