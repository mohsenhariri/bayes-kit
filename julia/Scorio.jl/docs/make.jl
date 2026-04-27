using Documenter
using Scorio

const Remotes = Documenter.Remotes

makedocs(
    sitename = "Scorio.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://mohsenhariri.github.io/scorio/julia/",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/scorio.css"],
        sidebar_sitename = false,
    ),
    modules = [Scorio],
    pages = [
        "Home" => "index.md",
        "API Reference" => [
            "Overview" => "api.md",
            "Evaluation (Scorio.Eval)" => "api/eval.md",
            "Ranking (Scorio.Rank)" => "api/rank.md",
            "Utilities (Scorio.Utils)" => "api/utils.md"
        ],
        "Examples" => "examples.md",
        "Citation" => "citation.md"
    ],
    repo = Remotes.GitHub("mohsenhariri", "scorio"),
)

# Note: We don't use deploydocs() here because deployment is handled
# by the unified GitHub Actions workflow that combines Python and Julia docs.
# The workflow uploads the built docs from docs/build/ to gh-pages branch.
