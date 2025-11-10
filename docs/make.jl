using Documenter, DEMetropolis, DocumenterInterLinks

links = InterLinks(
    "MCMCDiagnosticTools" => "https://turinglang.org/MCMCDiagnosticTools.jl/stable/objects.inv"
);

makedocs(
    sitename = "Differential Evolution Metropolis",
    plugins = [links],
    pages = [
        "index.md",
        "tutorial.md",
        "custom.md",
    ],
    modules = [DEMetropolis]
)

deploydocs(
    repo = "github.com/GBarnsley/DEMetropolis.jl.git",
)
