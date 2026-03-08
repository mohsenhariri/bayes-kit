using Dates
using Printf
using Random
using Statistics
using Scorio

const ROOT_DIR = normpath(joinpath(@__DIR__))
const RESULTS_DIR = joinpath(ROOT_DIR, "results")
const TABLES_DIR = joinpath(ROOT_DIR, "tables")
const FIGURES_DIR = joinpath(ROOT_DIR, "figures")
const BINARY_WEIGHTS = [0.0, 1.0]
const TRUE_THETA = Float64[-1.5, -1.2, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.2, 1.2, 1.6]

struct ExperimentProfile
    name::String
    rank_recovery_seeds::Vector{Int}
    rank_recovery_trials::Vector{Int}
    stability_seeds::Vector{Int}
    stability_trials::Vector{Int}
    stability_n_max::Int
    runtime_ls::Vector{Int}
    runtime_ms::Vector{Int}
    runtime_ns::Vector{Int}
    runtime_replicates::Int
    runtime_include_kemeny::Bool
end

function profile_by_name(name::AbstractString)::ExperimentProfile
    key = lowercase(String(name))
    if key == "full"
        return ExperimentProfile(
            "full",
            collect(1:30),
            [1, 2, 4, 8, 16, 32],
            collect(1:100),
            [1, 2, 4, 8, 16, 32],
            64,
            [4, 8, 16, 32],
            [100, 500, 1000, 5000],
            [1, 4, 16],
            5,
            true,
        )
    elseif key == "pilot"
        return ExperimentProfile(
            "pilot",
            collect(1:4),
            [1, 2, 4, 8, 16, 32],
            collect(1:10),
            [1, 2, 4, 8, 16, 32],
            64,
            [4, 8, 16],
            [100, 500, 1000],
            [1, 4],
            2,
            true,
        )
    end

    error("Unknown profile '$name'. Use 'pilot' or 'full'.")
end

function suffix_for_profile(profile::ExperimentProfile)::String
    return profile.name == "full" ? "" : "_" * profile.name
end

function profiled_filename(base::AbstractString, profile::ExperimentProfile)::String
    stem, ext = splitext(base)
    return stem * suffix_for_profile(profile) * ext
end

function output_path(dir::AbstractString, base::AbstractString, profile::ExperimentProfile)::String
    return joinpath(dir, profiled_filename(base, profile))
end

function manifest_path(profile::ExperimentProfile)::String
    return output_path(RESULTS_DIR, "experiment_manifest.txt", profile)
end

function error_log_path(profile::ExperimentProfile)::String
    return output_path(RESULTS_DIR, "experiment_errors.txt", profile)
end

function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

function synthetic_tensor(
    theta::AbstractVector{<:Real},
    m_questions::Integer,
    n_trials::Integer,
    seed::Integer,
)::Array{Int,3}
    rng = MersenneTwister(seed)
    theta_f = Float64.(collect(theta))
    l_models = length(theta_f)
    b = randn(rng, Int(m_questions))
    probs = zeros(Float64, l_models, Int(m_questions))
    for l in 1:l_models
        for m in 1:Int(m_questions)
            probs[l, m] = sigmoid(theta_f[l] - b[m])
        end
    end

    draws = rand(rng, l_models, Int(m_questions), Int(n_trials))
    return Int.(draws .< reshape(probs, l_models, Int(m_questions), 1))
end

function truth_ranking_from_theta(theta::AbstractVector{<:Real})::Vector{Float64}
    return Scorio.rank_scores(Float64.(collect(theta)))["competition"]
end

function topk_set_from_ranking(ranking::AbstractVector{<:Real}, k::Integer)::Set{Int}
    threshold = Float64(k)
    return Set(findall(x -> Float64(x) <= threshold + 1e-12, ranking))
end

function mean_abs_rank_error(predicted::AbstractVector{<:Real}, truth::AbstractVector{<:Real})::Float64
    return mean(abs.(Float64.(predicted) .- Float64.(truth)))
end

function exact_true_tie_recovered(
    ranking::AbstractVector{<:Real},
    left_idx::Integer,
    right_idx::Integer,
)::Bool
    left_rank = Float64(ranking[Int(left_idx)])
    right_rank = Float64(ranking[Int(right_idx)])
    if left_rank != right_rank
        return false
    end
    return count(x -> Float64(x) == left_rank, ranking) == 2
end

function count_adjacent_ties(
    ranking::AbstractVector{<:Real},
    scores::AbstractVector{<:Real},
)::Int
    order = sortperm(Float64.(scores); rev=true)
    ranked = Float64.(ranking)[order]
    ties = 0
    for i in 1:(length(ranked) - 1)
        ties += ranked[i] == ranked[i + 1] ? 1 : 0
    end
    return ties
end

function mean_adjacent_score_gap(scores::AbstractVector{<:Real})::Float64
    ordered = sort(Float64.(scores); rev=true)
    if length(ordered) < 2
        return NaN
    end
    return mean(abs.(diff(ordered)))
end

function safe_statistic(values::AbstractVector{<:Real}, reducer::Function)::Float64
    filtered = [Float64(x) for x in values if isfinite(Float64(x))]
    return isempty(filtered) ? NaN : reducer(filtered)
end

function safe_mean(values::AbstractVector{<:Real})::Float64
    return safe_statistic(values, mean)
end

function safe_std(values::AbstractVector{<:Real})::Float64
    return safe_statistic(values, x -> std(x; corrected=false))
end

function csv_escape(value)::String
    if value isa AbstractString
        text = replace(String(value), '"' => "\"\"")
        if occursin(',', text) || occursin('"', text) || occursin('\n', text)
            return "\"" * text * "\""
        end
        return text
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractFloat
        x = Float64(value)
        if isnan(x)
            return "NaN"
        elseif isinf(x)
            return x > 0 ? "Inf" : "-Inf"
        end
        return @sprintf("%.10g", x)
    end

    return string(value)
end

function write_csv(path::AbstractString, headers::Vector{Symbol}, rows::Vector{<:NamedTuple})
    open(path, "w") do io
        println(io, join(String.(headers), ","))
        for row in rows
            values = [csv_escape(getproperty(row, header)) for header in headers]
            println(io, join(values, ","))
        end
    end
end

function write_text(path::AbstractString, lines::Vector{String})
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
end

function tex_escape(text::AbstractString)::String
    escaped = String(text)
    escaped = replace(escaped, "\\" => "\\textbackslash{}")
    escaped = replace(escaped, "_" => "\\_")
    escaped = replace(escaped, "%" => "\\%")
    escaped = replace(escaped, "&" => "\\&")
    escaped = replace(escaped, "#" => "\\#")
    return escaped
end

function coordinate_list(points::Vector{Tuple{Float64,Float64}})::String
    return join(["(" * @sprintf("%.10g", x) * "," * @sprintf("%.10g", y) * ")" for (x, y) in points], " ")
end

const PLOT_COLORS = [
    "black",
    "blue!70!black",
    "teal!70!black",
    "red!70!black",
    "orange!90!black",
    "purple!70!black",
    "brown!80!black",
    "green!50!black",
    "magenta!70!black",
    "cyan!80!black",
]

function pdflatex_cmd(tex_path::AbstractString)
    return `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(FIGURES_DIR) $(tex_path)`
end

function compile_tex_to_pdf(tex_path::AbstractString)::Bool
    if isnothing(Sys.which("pdflatex"))
        return false
    end
    try
        run(pdflatex_cmd(tex_path))
        run(pdflatex_cmd(tex_path))
        return true
    catch err
        @warn "Failed to compile figure" tex_path exception=(err, catch_backtrace())
        return false
    end
end

function write_single_axis_plot(
    tex_path::AbstractString;
    title::AbstractString,
    xlabel::AbstractString,
    ylabel::AbstractString,
    series::Vector{NamedTuple},
    xmode::AbstractString="normal",
    ymode::AbstractString="normal",
)
    lines = String[
        "\\documentclass[tikz,border=4pt]{standalone}",
        "\\usepackage{pgfplots}",
        "\\pgfplotsset{compat=1.18}",
        "\\begin{document}",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "width=13cm,",
        "height=8cm,",
        "title={" * tex_escape(title) * "},",
        "xlabel={" * tex_escape(xlabel) * "},",
        "ylabel={" * tex_escape(ylabel) * "},",
        "grid=both,",
        "legend pos=south east,",
        "legend cell align=left,",
        xmode == "log2" ? "xmode=log, log basis x=2," : "",
        ymode == "log" ? "ymode=log," : "",
        "]",
    ]

    filter!(!isempty, lines)

    for (idx, entry) in enumerate(series)
        color = PLOT_COLORS[mod1(idx, length(PLOT_COLORS))]
        points = coordinate_list(entry.points)
        push!(lines, "\\addplot+[mark=*, thick, color=$color] coordinates {$points};")
        push!(lines, "\\addlegendentry{" * tex_escape(entry.label) * "}")
    end

    append!(lines, ["\\end{axis}", "\\end{tikzpicture}", "\\end{document}"])
    write_text(tex_path, lines)
end

function write_group_plot(
    tex_path::AbstractString;
    title::AbstractString,
    xlabel::AbstractString,
    ylabel::AbstractString,
    panels::Vector{NamedTuple},
    xmode::AbstractString="normal",
    ymode::AbstractString="normal",
)
    n_panels = length(panels)
    n_cols = min(3, max(1, n_panels))
    n_rows = cld(n_panels, n_cols)

    lines = String[
        "\\documentclass[tikz,border=4pt]{standalone}",
        "\\usepackage{pgfplots}",
        "\\usepgfplotslibrary{groupplots}",
        "\\pgfplotsset{compat=1.18}",
        "\\begin{document}",
        "\\begin{tikzpicture}",
        "\\begin{groupplot}[",
        "group style={group size=$n_cols by $n_rows, horizontal sep=1.5cm, vertical sep=2.0cm},",
        "width=6.1cm,",
        "height=4.8cm,",
        "grid=both,",
        "legend to name=combinedlegend,",
        "legend columns=5,",
        xmode == "log2" ? "xmode=log, log basis x=2," : "",
        ymode == "log" ? "ymode=log," : "",
        "]",
    ]

    filter!(!isempty, lines)

    xlabel_escaped = tex_escape(xlabel)
    ylabel_escaped = tex_escape(ylabel)

    for (panel_idx, panel) in enumerate(panels)
        is_left_col = mod1(panel_idx, n_cols) == 1
        is_bottom_row = panel_idx > n_panels - n_cols
        per_plot_opts = "title={" * tex_escape(panel.title) * "}"
        if is_left_col
            per_plot_opts *= ", ylabel={" * ylabel_escaped * "}"
        end
        if is_bottom_row
            per_plot_opts *= ", xlabel={" * xlabel_escaped * "}"
        end
        push!(lines, "\\nextgroupplot[$per_plot_opts]")
        for (idx, entry) in enumerate(panel.series)
            color = PLOT_COLORS[mod1(idx, length(PLOT_COLORS))]
            points = coordinate_list(entry.points)
            push!(lines, "\\addplot+[mark=*, thick, color=$color] coordinates {$points};")
            if panel_idx == 1
                push!(lines, "\\addlegendentry{" * tex_escape(entry.label) * "}")
            end
        end
    end

    append!(
        lines,
        [
            "\\end{groupplot}",
            "\\path (current bounding box.south) node[below=1.2cm] {\\pgfplotslegendfromname{combinedlegend}};",
            "\\path (current bounding box.north) node[above=0.7cm,font=\\bfseries] {" * tex_escape(title) * "};",
            "\\end{tikzpicture}",
            "\\end{document}",
        ],
    )
    write_text(tex_path, lines)
end

function add_error!(
    errors::Vector{String},
    experiment::AbstractString,
    context::AbstractString,
    err,
)
    push!(errors, "[$(experiment)] $(context): $(sprint(showerror, err))")
end

function compare_kendall_tau(predicted::AbstractVector{<:Real}, truth::AbstractVector{<:Real})::Float64
    tau, _ = Scorio.compare_rankings(predicted, truth; method="kendall")
    return tau
end

function compare_spearman(predicted::AbstractVector{<:Real}, truth::AbstractVector{<:Real})::Float64
    rho, _ = Scorio.compare_rankings(predicted, truth; method="spearman")
    return rho
end

function recovery_method_specs(seed::Integer)
    [
        (name="avg", runner=R -> Scorio.Rank.avg(R; return_scores=true)),
        (name="bayes", runner=R -> Scorio.Rank.bayes(R, BINARY_WEIGHTS; return_scores=true)),
        (name="elo", runner=R -> Scorio.Rank.elo(R; return_scores=true)),
        (name="glicko", runner=R -> Scorio.Rank.glicko(R; return_scores=true)),
        (
            name="bradley_terry_davidson",
            runner=R -> Scorio.Rank.bradley_terry_davidson(R; max_iter=100, return_scores=true),
        ),
        (
            name="thompson",
            runner=R -> Scorio.Rank.thompson(R; n_samples=2500, seed=seed, return_scores=true),
        ),
        (name="rasch", runner=R -> Scorio.Rank.rasch(R; max_iter=8, return_scores=true)),
        (name="borda", runner=R -> Scorio.Rank.borda(R; return_scores=true)),
        (name="pagerank", runner=R -> Scorio.Rank.pagerank(R; return_scores=true)),
        (
            name="plackett_luce",
            runner=R -> Scorio.Rank.plackett_luce(R; max_iter=200, return_scores=true),
        ),
    ]
end

function stability_method_specs(seed::Integer, n::Integer)
    capped_k = min(4, Int(n))
    [
        (name="avg", runner=R -> Scorio.Rank.avg(R; return_scores=true)),
        (name="bayes", runner=R -> Scorio.Rank.bayes(R, BINARY_WEIGHTS; return_scores=true)),
        (
            name="pass_at_k_4capn",
            runner=R -> Scorio.Rank.pass_at_k(R, capped_k; return_scores=true),
        ),
        (
            name="g_pass_at_k_tau_4capn_0.75",
            runner=R -> Scorio.Rank.g_pass_at_k_tau(R, capped_k, 0.75; return_scores=true),
        ),
        (
            name="mg_pass_at_k_4capn",
            runner=R -> Scorio.Rank.mg_pass_at_k(R, capped_k; return_scores=true),
        ),
        (name="elo", runner=R -> Scorio.Rank.elo(R; return_scores=true)),
        (
            name="bradley_terry_davidson",
            runner=R -> Scorio.Rank.bradley_terry_davidson(R; max_iter=100, return_scores=true),
        ),
        (name="rasch", runner=R -> Scorio.Rank.rasch(R; max_iter=8, return_scores=true)),
        (name="borda", runner=R -> Scorio.Rank.borda(R; return_scores=true)),
        (name="pagerank", runner=R -> Scorio.Rank.pagerank(R; return_scores=true)),
    ]
end

function runtime_method_specs(l_models::Integer)
    specs = [
        (name="avg", runner=R -> Scorio.Rank.avg(R; return_scores=true)),
        (name="bayes", runner=R -> Scorio.Rank.bayes(R, BINARY_WEIGHTS; return_scores=true)),
        (name="elo", runner=R -> Scorio.Rank.elo(R; return_scores=true)),
        (
            name="bradley_terry",
            runner=R -> Scorio.Rank.bradley_terry(R; max_iter=100, return_scores=true),
        ),
        (name="rasch", runner=R -> Scorio.Rank.rasch(R; max_iter=8, return_scores=true)),
        (name="borda", runner=R -> Scorio.Rank.borda(R; return_scores=true)),
        (name="pagerank", runner=R -> Scorio.Rank.pagerank(R; return_scores=true)),
        (
            name="alpharank",
            runner=R -> Scorio.Rank.alpharank(
                R;
                population_size=20,
                max_iter=20_000,
                return_scores=true,
            ),
        ),
        (
            name="plackett_luce",
            runner=R -> Scorio.Rank.plackett_luce(R; max_iter=200, return_scores=true),
        ),
    ]

    if Int(l_models) <= 12
        push!(
            specs,
            (
                name="kemeny_young",
                runner=R -> Scorio.Rank.kemeny_young(
                    R;
                    time_limit=10.0,
                    return_scores=true,
                ),
            ),
        )
    end

    return specs
end

function summarize_metric_rows(rows, group_headers::Vector{Symbol}, metric_keys::Vector{Symbol}, metric_names::Dict{Symbol,String})
    grouped = Dict{Tuple,Dict{Symbol,Vector{Float64}}}()

    for row in rows
        group_key = tuple((getproperty(row, h) for h in group_headers)...)
        bucket = get!(grouped, group_key, Dict{Symbol,Vector{Float64}}())
        for key in metric_keys
            values = get!(bucket, key, Float64[])
            value = Float64(getproperty(row, key))
            if isfinite(value)
                push!(values, value)
            end
        end
    end

    summary_rows = NamedTuple[]
    for (group_key, metrics) in sort(collect(grouped); by=first)
        for metric_key in metric_keys
            values = get(metrics, metric_key, Float64[])
            push!(
                summary_rows,
                NamedTuple{Tuple(vcat(group_headers, [:metric, :mean, :std]))}(
                    tuple(
                        group_key...,
                        metric_names[metric_key],
                        safe_mean(values),
                        safe_std(values),
                    ),
                ),
            )
        end
    end

    return summary_rows
end

function series_from_summary(
    rows::Vector{<:NamedTuple},
    x_key::Symbol,
    y_key::Symbol,
    label_key::Symbol,
)
    grouped = Dict{String,Vector{Tuple{Float64,Float64}}}()
    for row in rows
        x = Float64(getproperty(row, x_key))
        y = Float64(getproperty(row, y_key))
        if !isfinite(y)
            continue
        end
        label = String(getproperty(row, label_key))
        push!(get!(grouped, label, Tuple{Float64,Float64}[]), (x, y))
    end

    output = NamedTuple[]
    for (label, points) in sort(collect(grouped); by=first)
        sorted_points = sort(points; by=first)
        push!(output, (label=label, points=sorted_points))
    end
    return output
end

function run_synthetic_rank_recovery(profile::ExperimentProfile, errors::Vector{String}; compile_figures::Bool=true)
    println("Running synthetic rank recovery ($(profile.name))")

    truth_ranking = truth_ranking_from_theta(TRUE_THETA)
    truth_top1 = topk_set_from_ranking(truth_ranking, 1)
    truth_top3 = topk_set_from_ranking(truth_ranking, 3)
    max_trials = maximum(profile.rank_recovery_trials)
    rows = NamedTuple[]

    for seed in profile.rank_recovery_seeds
        println("  seed $seed")
        tensor = synthetic_tensor(TRUE_THETA, 500, max_trials, seed)
        for n in profile.rank_recovery_trials
            subset = @view tensor[:, :, 1:n]
            for spec in recovery_method_specs(seed)
                try
                    ranking, scores = spec.runner(subset)
                    push!(
                        rows,
                        (
                            seed=seed,
                            N=n,
                            method=spec.name,
                            kendall_tau_b=compare_kendall_tau(ranking, truth_ranking),
                            spearman_rho=compare_spearman(ranking, truth_ranking),
                            top1_correct=topk_set_from_ranking(ranking, 1) == truth_top1 ? 1.0 : 0.0,
                            top3_set_correct=topk_set_from_ranking(ranking, 3) == truth_top3 ? 1.0 : 0.0,
                            mean_abs_rank_error=mean_abs_rank_error(ranking, truth_ranking),
                            recovered_true_tie=exact_true_tie_recovered(ranking, 9, 10) ? 1.0 : 0.0,
                        ),
                    )
                catch err
                    add_error!(errors, "synthetic_rank_recovery", "seed=$seed N=$n method=$(spec.name)", err)
                end
            end
        end
    end

    result_headers = [
        :seed,
        :N,
        :method,
        :kendall_tau_b,
        :spearman_rho,
        :top1_correct,
        :top3_set_correct,
        :mean_abs_rank_error,
        :recovered_true_tie,
    ]
    result_path = output_path(RESULTS_DIR, "synthetic_rank_recovery.csv", profile)
    write_csv(result_path, result_headers, rows)

    summary_rows = summarize_metric_rows(
        rows,
        [:method, :N],
        [
            :kendall_tau_b,
            :spearman_rho,
            :top1_correct,
            :top3_set_correct,
            :mean_abs_rank_error,
            :recovered_true_tie,
        ],
        Dict(
            :kendall_tau_b => "kendall_tau_b",
            :spearman_rho => "spearman_rho",
            :top1_correct => "top1_correct",
            :top3_set_correct => "top3_set_correct",
            :mean_abs_rank_error => "mean_abs_rank_error",
            :recovered_true_tie => "recovered_true_tie",
        ),
    )
    summary_path = output_path(TABLES_DIR, "table_synthetic_rank_recovery_summary.csv", profile)
    write_csv(summary_path, [:method, :N, :metric, :mean, :std], summary_rows)

    tau_rows = [row for row in summary_rows if row.metric == "kendall_tau_b"]
    tau_series = series_from_summary(tau_rows, :N, :mean, :method)
    tau_tex = output_path(FIGURES_DIR, "fig_synthetic_rank_recovery_tau.tex", profile)
    write_single_axis_plot(
        tau_tex;
        title="Synthetic Rank Recovery: Kendall tau-b",
        xlabel="Trials per question (N)",
        ylabel="Mean Kendall tau-b",
        series=tau_series,
        xmode="log2",
    )
    compile_figures && compile_tex_to_pdf(tau_tex)

    mae_rows = [row for row in summary_rows if row.metric == "mean_abs_rank_error"]
    mae_series = series_from_summary(mae_rows, :N, :mean, :method)
    mae_tex = output_path(FIGURES_DIR, "fig_synthetic_rank_recovery_mae.tex", profile)
    write_single_axis_plot(
        mae_tex;
        title="Synthetic Rank Recovery: Mean Absolute Rank Error",
        xlabel="Trials per question (N)",
        ylabel="Mean absolute rank error",
        series=mae_series,
        xmode="log2",
    )
    compile_figures && compile_tex_to_pdf(mae_tex)

    return (
        result_path=result_path,
        summary_path=summary_path,
        tau_figure=replace(tau_tex, ".tex" => ".pdf"),
        mae_figure=replace(mae_tex, ".tex" => ".pdf"),
        n_rows=length(rows),
    )
end

function run_stability_vs_trials(profile::ExperimentProfile, errors::Vector{String}; compile_figures::Bool=true)
    println("Running stability vs trials ($(profile.name))")

    rows = NamedTuple[]
    for seed in profile.stability_seeds
        println("  seed $seed")
        tensor = synthetic_tensor(TRUE_THETA, 500, profile.stability_n_max, 10_000 + seed)
        reference_ranking, _ = Scorio.Rank.bayes(tensor, BINARY_WEIGHTS; return_scores=true)
        reference_top1 = topk_set_from_ranking(reference_ranking, 1)
        reference_top3 = topk_set_from_ranking(reference_ranking, 3)

        for n in profile.stability_trials
            subset = @view tensor[:, :, 1:n]
            for spec in stability_method_specs(seed, n)
                try
                    ranking, scores = spec.runner(subset)
                    push!(
                        rows,
                        (
                            seed=seed,
                            n=n,
                            method=spec.name,
                            kendall_tau_b_to_reference=compare_kendall_tau(ranking, reference_ranking),
                            top1_match=topk_set_from_ranking(ranking, 1) == reference_top1 ? 1.0 : 0.0,
                            top3_set_match=topk_set_from_ranking(ranking, 3) == reference_top3 ? 1.0 : 0.0,
                            n_adjacent_ties=count_adjacent_ties(ranking, scores),
                            mean_adjacent_score_gap=mean_adjacent_score_gap(scores),
                        ),
                    )
                catch err
                    add_error!(errors, "stability_vs_trials", "seed=$seed n=$n method=$(spec.name)", err)
                end
            end
        end
    end

    result_headers = [
        :seed,
        :n,
        :method,
        :kendall_tau_b_to_reference,
        :top1_match,
        :top3_set_match,
        :n_adjacent_ties,
        :mean_adjacent_score_gap,
    ]
    result_path = output_path(RESULTS_DIR, "stability_vs_trials.csv", profile)
    write_csv(result_path, result_headers, rows)

    summary_rows = summarize_metric_rows(
        rows,
        [:method, :n],
        [
            :kendall_tau_b_to_reference,
            :top1_match,
            :top3_set_match,
            :n_adjacent_ties,
            :mean_adjacent_score_gap,
        ],
        Dict(
            :kendall_tau_b_to_reference => "kendall_tau_b_to_reference",
            :top1_match => "top1_match",
            :top3_set_match => "top3_set_match",
            :n_adjacent_ties => "n_adjacent_ties",
            :mean_adjacent_score_gap => "mean_adjacent_score_gap",
        ),
    )
    summary_path = output_path(TABLES_DIR, "table_stability_summary.csv", profile)
    write_csv(summary_path, [:method, :n, :metric, :mean, :std], summary_rows)

    tau_rows = [row for row in summary_rows if row.metric == "kendall_tau_b_to_reference"]
    tau_series = series_from_summary(tau_rows, :n, :mean, :method)
    tau_tex = output_path(FIGURES_DIR, "fig_stability_vs_trials.tex", profile)
    write_single_axis_plot(
        tau_tex;
        title="Ranking Stability vs Trials",
        xlabel="Trials kept (n)",
        ylabel="Mean Kendall tau-b to reference",
        series=tau_series,
        xmode="log2",
    )
    compile_figures && compile_tex_to_pdf(tau_tex)

    top1_rows = [row for row in summary_rows if row.metric == "top1_match"]
    top1_series = series_from_summary(top1_rows, :n, :mean, :method)
    top1_tex = output_path(FIGURES_DIR, "fig_top1_stability_vs_trials.tex", profile)
    write_single_axis_plot(
        top1_tex;
        title="Top-1 Stability vs Trials",
        xlabel="Trials kept (n)",
        ylabel="Top-1 agreement probability",
        series=top1_series,
        xmode="log2",
    )
    compile_figures && compile_tex_to_pdf(top1_tex)

    return (
        result_path=result_path,
        summary_path=summary_path,
        tau_figure=replace(tau_tex, ".tex" => ".pdf"),
        top1_figure=replace(top1_tex, ".tex" => ".pdf"),
        n_rows=length(rows),
    )
end

function summarize_runtime_rows(rows)
    grouped = Dict{Tuple,Vector{NamedTuple}}()
    for row in rows
        key = (row.method, row.L, row.M, row.N)
        push!(get!(grouped, key, NamedTuple[]), row)
    end

    summary_rows = NamedTuple[]
    for (key, bucket) in sort(collect(grouped); by=first)
        wall_times = [Float64(row.wall_time_s) for row in bucket if row.success]
        memory_values = [Float64(row.memory_bytes) for row in bucket if row.success]
        success_rate = mean([row.success ? 1.0 : 0.0 for row in bucket])
        push!(
            summary_rows,
            (
                method=key[1],
                L=key[2],
                M=key[3],
                N=key[4],
                wall_time_mean_s=safe_mean(wall_times),
                wall_time_std_s=safe_std(wall_times),
                memory_mean_bytes=safe_mean(memory_values),
                success_rate=success_rate,
            ),
        )
    end

    return summary_rows
end

function runtime_panel_series(
    rows::Vector{<:NamedTuple},
    fixed_key::Function,
    x_key::Symbol,
)
    grouped = Dict{Any,Dict{String,Vector{Tuple{Float64,Float64}}}}()
    for row in rows
        if !isfinite(Float64(row.wall_time_mean_s))
            continue
        end
        panel_key = fixed_key(row)
        panel = get!(grouped, panel_key, Dict{String,Vector{Tuple{Float64,Float64}}}())
        label = String(row.method)
        push!(
            get!(panel, label, Tuple{Float64,Float64}[]),
            (Float64(getproperty(row, x_key)), max(Float64(row.wall_time_mean_s), 1e-9)),
        )
    end

    panels = NamedTuple[]
    for (panel_key, series_dict) in sort(collect(grouped); by=first)
        series = NamedTuple[]
        for (label, points) in sort(collect(series_dict); by=first)
            push!(series, (label=label, points=sort(points; by=first)))
        end
        push!(panels, (title=String(panel_key), series=series))
    end
    return panels
end

function benchmark_runtime(spec, tensor)
    GC.gc()
    result = @timed spec.runner(tensor)
    return result.time, Float64(result.bytes)
end

function run_runtime_scaling(profile::ExperimentProfile, errors::Vector{String}; compile_figures::Bool=true)
    println("Running runtime scaling ($(profile.name))")

    rows = NamedTuple[]
    for l_models in profile.runtime_ls
        for m_questions in profile.runtime_ms
            for n_trials in profile.runtime_ns
                println("  L=$l_models M=$m_questions N=$n_trials")
                specs = runtime_method_specs(l_models)
                for spec in specs
                    warmup_tensor = synthetic_tensor(
                        collect(range(-1.2, 1.2; length=l_models)),
                        m_questions,
                        n_trials,
                        hash((profile.name, "warmup", spec.name, l_models, m_questions, n_trials)),
                    )
                    try
                        spec.runner(warmup_tensor)
                    catch err
                        add_error!(
                            errors,
                            "runtime_scaling",
                            "warmup L=$l_models M=$m_questions N=$n_trials method=$(spec.name)",
                            err,
                        )
                    end

                    for replicate in 1:profile.runtime_replicates
                        tensor = synthetic_tensor(
                            collect(range(-1.2, 1.2; length=l_models)),
                            m_questions,
                            n_trials,
                            hash((profile.name, replicate, spec.name, l_models, m_questions, n_trials)),
                        )
                        success = true
                        wall_time_s = NaN
                        memory_bytes = NaN
                        try
                            wall_time_s, memory_bytes = benchmark_runtime(spec, tensor)
                        catch err
                            success = false
                            add_error!(
                                errors,
                                "runtime_scaling",
                                "replicate=$replicate L=$l_models M=$m_questions N=$n_trials method=$(spec.name)",
                                err,
                            )
                        end

                        push!(
                            rows,
                            (
                                replicate=replicate,
                                L=l_models,
                                M=m_questions,
                                N=n_trials,
                                method=spec.name,
                                wall_time_s=wall_time_s,
                                memory_bytes=memory_bytes,
                                success=success,
                            ),
                        )
                    end
                end
            end
        end
    end

    result_headers = [
        :replicate,
        :L,
        :M,
        :N,
        :method,
        :wall_time_s,
        :memory_bytes,
        :success,
    ]
    result_path = output_path(RESULTS_DIR, "runtime_scaling.csv", profile)
    write_csv(result_path, result_headers, rows)

    summary_rows = summarize_runtime_rows(rows)
    summary_path = output_path(TABLES_DIR, "table_runtime_summary.csv", profile)
    write_csv(
        summary_path,
        [:method, :L, :M, :N, :wall_time_mean_s, :wall_time_std_s, :memory_mean_bytes, :success_rate],
        summary_rows,
    )

    runtime_m_panels = runtime_panel_series(
        summary_rows,
        row -> "L=$(row.L), N=$(row.N)",
        :M,
    )
    runtime_m_tex = output_path(FIGURES_DIR, "fig_runtime_vs_M.tex", profile)
    write_group_plot(
        runtime_m_tex;
        title="Runtime vs M",
        xlabel="Questions (M)",
        ylabel="Wall time (s)",
        panels=runtime_m_panels,
        ymode="log",
    )
    compile_figures && compile_tex_to_pdf(runtime_m_tex)

    runtime_l_panels = runtime_panel_series(
        summary_rows,
        row -> "M=$(row.M), N=$(row.N)",
        :L,
    )
    runtime_l_tex = output_path(FIGURES_DIR, "fig_runtime_vs_L.tex", profile)
    write_group_plot(
        runtime_l_tex;
        title="Runtime vs L",
        xlabel="Models (L)",
        ylabel="Wall time (s)",
        panels=runtime_l_panels,
        ymode="log",
    )
    compile_figures && compile_tex_to_pdf(runtime_l_tex)

    return (
        result_path=result_path,
        summary_path=summary_path,
        runtime_vs_m_figure=replace(runtime_m_tex, ".tex" => ".pdf"),
        runtime_vs_l_figure=replace(runtime_l_tex, ".tex" => ".pdf"),
        n_rows=length(rows),
    )
end

function usage()
    println(
        """
        Usage:
          julia --project=Scorio.jl experiments/run_paper_experiments.jl [all|recovery|stability|runtime] [--profile=pilot|full] [--no-figures]
        """,
    )
end

function parse_args(args::Vector{String})
    selected = String[]
    profile_name = "pilot"
    compile_figures = true

    for arg in args
        if startswith(arg, "--profile=")
            profile_name = split(arg, "=", limit=2)[2]
        elseif arg == "--no-figures"
            compile_figures = false
        elseif arg in ("all", "recovery", "stability", "runtime")
            push!(selected, arg)
        elseif arg in ("-h", "--help")
            usage()
            exit(0)
        else
            error("Unknown argument '$arg'")
        end
    end

    if isempty(selected)
        selected = ["all"]
    end

    if "all" in selected
        selected = ["recovery", "stability", "runtime"]
    else
        selected = unique(selected)
    end

    return selected, profile_by_name(profile_name), compile_figures
end

function write_manifest(
    profile::ExperimentProfile,
    selected::Vector{String},
    outputs::Vector{Pair{String,Any}},
    errors::Vector{String},
)
    generated_at = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    selected_joined = join(selected, ",")
    rank_recovery_seeds = join(profile.rank_recovery_seeds, ",")
    rank_recovery_trials = join(profile.rank_recovery_trials, ",")
    stability_seeds = join(profile.stability_seeds, ",")
    stability_trials = join(profile.stability_trials, ",")
    runtime_ls = join(profile.runtime_ls, ",")
    runtime_ms = join(profile.runtime_ms, ",")
    runtime_ns = join(profile.runtime_ns, ",")

    lines = String[
        "profile=" * profile.name,
        "generated_at=" * generated_at,
        "selected=" * selected_joined,
        "rank_recovery_seeds=" * rank_recovery_seeds,
        "rank_recovery_trials=" * rank_recovery_trials,
        "stability_seeds=" * stability_seeds,
        "stability_trials=" * stability_trials,
        "stability_n_max=$(profile.stability_n_max)",
        "runtime_ls=" * runtime_ls,
        "runtime_ms=" * runtime_ms,
        "runtime_ns=" * runtime_ns,
        "runtime_replicates=$(profile.runtime_replicates)",
        "notes=recovery/stability use nested Rasch-style synthetic tensors; pass-family stability metrics use k=min(4,n); rasch uses max_iter=8; runtime kemeny_young included only for L<=12.",
    ]

    for (name, result) in outputs
        push!(lines, "[$name]")
        for field in propertynames(result)
            push!(lines, "$(field)=$(getproperty(result, field))")
        end
    end

    push!(lines, "error_count=$(length(errors))")
    write_text(manifest_path(profile), lines)
    write_text(error_log_path(profile), isempty(errors) ? ["No errors recorded."] : errors)
end

function main()
    selected, profile, compile_figures = parse_args(ARGS)
    errors = String[]
    outputs = Pair{String,Any}[]

    if "recovery" in selected
        push!(outputs, "recovery" => run_synthetic_rank_recovery(profile, errors; compile_figures=compile_figures))
    end
    if "stability" in selected
        push!(outputs, "stability" => run_stability_vs_trials(profile, errors; compile_figures=compile_figures))
    end
    if "runtime" in selected
        push!(outputs, "runtime" => run_runtime_scaling(profile, errors; compile_figures=compile_figures))
    end

    write_manifest(profile, selected, outputs, errors)

    println()
    println("Completed experiments for profile=$(profile.name)")
    println("Manifest: $(manifest_path(profile))")
    println("Error log: $(error_log_path(profile))")
end

main()
