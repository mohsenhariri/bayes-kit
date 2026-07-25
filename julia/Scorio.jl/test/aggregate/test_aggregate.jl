using Test
using JSON
using Scorio

const AggregateAPI = Scorio.Aggregate
const AGGREGATE_FIXTURE = JSON.parsefile(joinpath(@__DIR__, "..", "fixtures", "aggregate.json"))

function _check_python_selection(actual, expected; with_score::Bool=true)
    @test actual[1] == expected[1]
    @test actual[2] == Int(expected[2])
    with_score && @test actual[3] ≈ Float64(expected[3]) atol = 1e-14
end

@testset "aggregate/test_aggregate.jl" begin
    @testset "public namespace" begin
        @test Scorio.Agg === Scorio.Aggregate
        @test Scorio.agg === Scorio.Aggregate
        @test AggregateAPI.mob === AggregateAPI.majority_of_the_bests
        for name in (
            :mean_logprob,
            :sequence_logprob,
            :perplexity,
            :self_certainty,
            :token_confidence,
            :deepconf_confidence,
            :token_entropy,
            :varentropy,
            :max_softmax_probability,
            :logprob_margin,
            :picsar,
            :prm_aggregate,
            :best_of_n,
            :majority_of_the_bests,
            :mob,
            :best_of_majority,
            :majority_vote,
            :weighted_majority_vote,
            :softmax_weighted_vote,
            :rank_weighted_vote,
            :logit_weighted_vote,
            :filtered_vote,
            :KDEVoteCalibration,
            :fit_kde_vote_calibration,
            :kde_weighted_vote,
            :CGES_OTHER,
            :cges_vote,
            :cges_stop,
            :adaptive_consistency_stop,
            :adaptive_consistency_dirichlet_stop,
            :adaptive_consistency_crp_stop,
            :esc_stop,
            :deepconf_stop_threshold,
            :deepconf_online_stop,
        )
            @test isdefined(AggregateAPI, name)
        end
        # `names` also includes the module's own `:Aggregate` binding.
        @test length(names(AggregateAPI)) == 35
    end

    @testset "Python golden confidence signals" begin
        input = AGGREGATE_FIXTURE["inputs"]
        expected = AGGREGATE_FIXTURE["confidence"]
        logprobs = input["logprobs"]
        topk = input["topk"]

        @test AggregateAPI.mean_logprob(logprobs) ≈ expected["mean_logprob"] atol = 1e-14
        @test AggregateAPI.sequence_logprob(logprobs) ≈
              expected["sequence_logprob"] atol = 1e-14
        @test AggregateAPI.perplexity(logprobs) ≈ expected["perplexity"] atol = 1e-14

        picsar = expected["picsar"]
        @test AggregateAPI.picsar(logprobs) ≈ picsar["whole"] atol = 1e-14
        @test AggregateAPI.picsar(logprobs; answer_start=3) ≈
              picsar["split"] atol = 1e-14
        @test AggregateAPI.picsar(
            logprobs;
            answer_start=3,
            normalize_reasoning=true,
        ) ≈ picsar["normalized"] atol = 1e-14
        @test AggregateAPI.picsar(logprobs; answer_start=0) ≈
              picsar["split_zero"] atol = 1e-14
        @test AggregateAPI.picsar(logprobs; answer_start=length(logprobs)) ≈
              picsar["split_end"] atol = 1e-14

        for (name, fn) in (
            ("self_certainty", AggregateAPI.self_certainty),
            ("token_entropy", AggregateAPI.token_entropy),
            ("varentropy", AggregateAPI.varentropy),
            ("max_softmax_probability", AggregateAPI.max_softmax_probability),
        )
            for how in ("mean", "min", "max")
                @test fn(topk; aggregate=how) ≈ expected[name][how] atol = 2e-14
            end
        end

        margin = expected["logprob_margin"]
        @test AggregateAPI.logprob_margin(topk) ≈ margin["log"] atol = 1e-14
        @test AggregateAPI.logprob_margin(topk; use_prob=true) ≈
              margin["prob"] atol = 1e-14
        @test AggregateAPI.logprob_margin(topk; aggregate="min") ≈
              margin["min"] atol = 1e-14
        @test AggregateAPI.logprob_margin(input["ragged_topk"]) ≈
              margin["ragged"] atol = 1e-14

        @test AggregateAPI.token_confidence(topk) ≈
              Float64.(expected["token_confidence"]) atol = 1e-14
        deepconf = expected["deepconf_confidence"]
        @test AggregateAPI.deepconf_confidence(topk) ≈ deepconf["mean"] atol = 1e-14
        @test AggregateAPI.deepconf_confidence(topk; mode="tail", tail_tokens=2) ≈
              deepconf["tail"] atol = 1e-14
        @test AggregateAPI.deepconf_confidence(topk; mode="lowest_group", window=2) ≈
              deepconf["lowest_group"] atol = 1e-14
        @test AggregateAPI.deepconf_confidence(
            topk;
            mode="bottom_group",
            window=2,
            bottom_quantile=0.5,
        ) ≈ deepconf["bottom_group"] atol = 1e-14
    end

    @testset "confidence input and boundary behavior" begin
        @test AggregateAPI._flatten_numeric([1.0 2.0; 3.0 4.0], "x") ==
              [1.0, 2.0, 3.0, 4.0]
        @test AggregateAPI.picsar(
            [1.0 2.0; 3.0 4.0];
            answer_start=2,
            normalize_reasoning=true,
        ) == 8.5
        @test AggregateAPI.self_certainty([0.0, -1.0, -2.0]) ≈
              AggregateAPI.self_certainty([[0.0, -1.0, -2.0]])
        @test AggregateAPI.logprob_margin([[0.0]]) == 0.0
        @test AggregateAPI.token_entropy([[-log(3.0), -log(3.0), -log(3.0)]]) ≈
              log(3.0) atol = 1e-12
        @test abs(
            AggregateAPI.self_certainty([[-log(3.0), -log(3.0), -log(3.0)]]),
        ) < 1e-12
        # Python golden values for an extreme but finite top-k row. Keeping
        # normalized values in log space prevents `log(0)`/NaN underflow.
        extreme = [[0.0, -1000.0]]
        @test AggregateAPI.token_entropy(extreme) ≈ 0.0 atol = 1e-300
        @test AggregateAPI.varentropy(extreme) ≈ 0.0 atol = 1e-300
        @test AggregateAPI.self_certainty(extreme) ≈
              500.0 - log(2.0) atol = 1e-12

        # NumPy treats the first dimension as tokens and C-flattens remaining
        # dimensions in its ragged fallback.
        topk_3d = reshape([0.0, -1.0, -2.0, -3.0], 1, 2, 2)
        @test AggregateAPI.token_confidence(topk_3d) == [1.5]

        @test_throws ErrorException AggregateAPI.mean_logprob([])
        @test_throws ErrorException AggregateAPI.mean_logprob([0.0, Inf])
        @test_throws ErrorException AggregateAPI.mean_logprob([[1.0], [2.0, 3.0]])
        @test_throws ErrorException AggregateAPI.self_certainty([])
        @test_throws ErrorException AggregateAPI.token_entropy([[0.0, NaN]])
        @test_throws ErrorException AggregateAPI.self_certainty(
            [[0.0, -1.0]];
            aggregate="median",
        )
        @test_throws ErrorException AggregateAPI.picsar([-0.1]; answer_start=2)
        @test_throws ErrorException AggregateAPI.deepconf_confidence(
            [[0.0, -1.0]];
            mode="bad",
        )
        @test_throws ErrorException AggregateAPI.deepconf_confidence(
            [[0.0, -1.0]];
            mode="bottom_group",
            bottom_quantile=0.0,
        )
        @test_throws ErrorException AggregateAPI.deepconf_confidence(
            [[0.0, -1.0]];
            mode="lowest_group",
            window=0,
        )
    end

    @testset "Python golden PRM reductions" begin
        expected = AGGREGATE_FIXTURE["prm_aggregate"]
        for method in ("last", "min", "mean", "prod", "max")
            @test AggregateAPI.prm_aggregate(
                [0.9, 0.4, 0.95];
                method=method,
            ) ≈ expected[method] atol = 1e-14
        end
        @test AggregateAPI.prm_aggregate(0.7) == 0.7
        @test AggregateAPI.prm_aggregate([0.9 0.4; 0.8 0.95]; method="last") == 0.95
        @test_throws ErrorException AggregateAPI.prm_aggregate([])
        @test_throws ErrorException AggregateAPI.prm_aggregate([0.2, Inf])
        @test_throws ErrorException AggregateAPI.prm_aggregate([[0.2], [0.3, 0.4]])
        @test_throws ErrorException AggregateAPI.prm_aggregate([0.2]; method="median")
    end

    @testset "Python golden selection rules" begin
        input = AGGREGATE_FIXTURE["inputs"]
        expected = AGGREGATE_FIXTURE["selection"]
        answers, scores = input["answers"], input["scores"]

        majority = AggregateAPI.majority_vote(answers; return_index=true)
        _check_python_selection(majority, expected["majority_vote"]; with_score=false)
        _check_python_selection(
            AggregateAPI.best_of_n(
                answers,
                scores;
                return_index=true,
                return_score=true,
            ),
            expected["best_of_n"],
        )
        _check_python_selection(
            AggregateAPI.majority_of_the_bests(
                answers,
                scores;
                return_index=true,
                return_score=true,
            ),
            expected["majority_of_the_bests"],
        )
        _check_python_selection(
            AggregateAPI.mob(
                answers,
                scores;
                return_index=true,
                return_score=true,
            ),
            expected["mob"],
        )
        _check_python_selection(
            AggregateAPI.majority_of_the_bests(
                answers,
                scores;
                m=1,
                return_index=true,
                return_score=true,
            ),
            expected["majority_of_the_bests_m1"],
        )
        _check_python_selection(
            AggregateAPI.best_of_majority(
                answers,
                scores;
                alpha=0.4,
                aggregate="mean",
                return_index=true,
                return_score=true,
            ),
            expected["best_of_majority"],
        )
        for aggregate in ("sum", "mean")
            _check_python_selection(
                AggregateAPI.weighted_majority_vote(
                    answers,
                    scores;
                    aggregate=aggregate,
                    return_index=true,
                    return_score=true,
                ),
                expected["weighted_majority_vote_$aggregate"],
            )
        end
        _check_python_selection(
            AggregateAPI.softmax_weighted_vote(
                answers,
                scores;
                temperature=0.7,
                return_index=true,
                return_score=true,
            ),
            expected["softmax_weighted_vote"],
        )
        _check_python_selection(
            AggregateAPI.rank_weighted_vote(
                answers,
                scores;
                p=1.3,
                return_index=true,
                return_score=true,
            ),
            expected["rank_weighted_vote"],
        )
        _check_python_selection(
            AggregateAPI.logit_weighted_vote(
                answers,
                scores;
                threshold=0.5,
                return_index=true,
                return_score=true,
            ),
            expected["logit_weighted_vote"],
        )
        _check_python_selection(
            AggregateAPI.logit_weighted_vote(
                answers,
                scores;
                threshold=0.2,
                transform="linear",
                return_index=true,
                return_score=true,
            ),
            expected["logit_weighted_vote_linear"],
        )
        _check_python_selection(
            AggregateAPI.filtered_vote(
                answers,
                scores;
                keep=0.5,
                return_index=true,
                return_score=true,
            ),
            expected["filtered_vote_weighted"],
        )
        _check_python_selection(
            AggregateAPI.filtered_vote(
                answers,
                scores;
                keep=3,
                weighted=false,
                return_index=true,
                return_score=true,
            ),
            expected["filtered_vote_unweighted"],
        )
    end

    @testset "batched return contract and shape" begin
        input = AGGREGATE_FIXTURE["inputs"]
        expected = AGGREGATE_FIXTURE["batch"]
        answers, scores = input["batch_answers"], input["batch_scores"]

        selected, indices = AggregateAPI.majority_vote(answers; return_index=true)
        @test selected == expected["majority_vote"][1]
        @test indices == Int.(expected["majority_vote"][2])

        for (name, fn) in (
            ("best_of_n", AggregateAPI.best_of_n),
            ("weighted_majority_vote", AggregateAPI.weighted_majority_vote),
        )
            selected, indices, selected_scores =
                fn(answers, scores; return_index=true, return_score=true)
            @test selected == expected[name][1]
            @test indices == Int.(expected[name][2])
            @test selected_scores ≈ Float64.(expected[name][3]) atol = 1e-14
        end

        matrix_answers = ["A" "B" "A"; "X" "X" "Y"]
        matrix_scores = [0.1 0.9 0.2; 0.4 0.3 0.8]
        @test AggregateAPI.best_of_n(matrix_answers, matrix_scores) == ["B", "Y"]
    end

    @testset "selection ties, validity, and exact arithmetic" begin
        @test AggregateAPI.majority_vote(["B", "A", "A", "B"]) == "B"
        @test AggregateAPI.best_of_n(["A", "B"], [0.5, 0.5]; return_index=true) ==
              ("A", 0)
        @test AggregateAPI.filtered_vote(
            ["A", "B"],
            [0.1, 0.9];
            keep=1.0,
            weighted=false,
        ) == "A" # tie by original appearance, not retained score order
        @test AggregateAPI.filtered_vote(
            ["A", "B", "B", "A"],
            [0.1, 0.3, 0.8, 0.9];
            keep=1.0,
            weighted=false,
        ) == "B" # Python tracks each group's first member within score-sorted survivors
        @test AggregateAPI.weighted_majority_vote(
            ["A", "A", "B"],
            [0.1, 0.9, 0.5];
            return_index=true,
        ) == ("A", 1)

        answers = Any["A", nothing, "", NaN, "B"]
        scores = [0.2, 99.0, 98.0, 97.0, 0.3]
        @test AggregateAPI.best_of_n(answers, scores) == "B"
        @test AggregateAPI.majority_vote(answers) == "A"
        @test AggregateAPI.logit_weighted_vote(
            Any["A", nothing, "B"],
            [0.4, 9.0, 0.9],
        ) == "B" # invalid candidate's out-of-range score is ignored

        selection, index, score = AggregateAPI.best_of_n(
            Any[nothing, "", NaN],
            [1.0, 2.0, 3.0];
            return_index=true,
            return_score=true,
        )
        @test selection === nothing
        @test index == -1
        @test isnan(score)
        @test AggregateAPI.majority_vote(
            Any[nothing, "", NaN];
            return_index=true,
        ) == (nothing, -1)

        tie_answers = ["D", "B", "D", "B", "A", "B", "A"]
        tie_scores = [0.6667, 0.3333, 0.8333, 0.1667, 0.5, 0.0, 1.0]
        @test AggregateAPI.majority_of_the_bests(tie_answers, tie_scores) == "D"
        @test AggregateAPI.majority_of_the_bests(
            Any["D", nothing, "B", "D", "B", "A", "B", "A"],
            [0.6667, 99.0, 0.3333, 0.8333, 0.1667, 0.5, 0.0, 1.0],
        ) == "D" # default m uses valid n
        @test AggregateAPI.majority_of_the_bests(tie_answers, tie_scores; m=99) ==
              AggregateAPI.majority_of_the_bests(tie_answers, tie_scores; m=7)

        many_answers = [i == 200 ? "B" : "A" for i in 1:200]
        many_scores = Float64.(1:200)
        @test AggregateAPI.rank_weighted_vote(many_answers, many_scores; p=300.0) ==
              AggregateAPI.best_of_n(many_answers, many_scores)
        @test AggregateAPI.rank_weighted_vote(many_answers, many_scores; p=300.5) ==
              AggregateAPI.best_of_n(many_answers, many_scores)
    end

    @testset "selection parameter validation and identities" begin
        answers = ["A", "A", "B", "C"]
        scores = [0.1, 0.2, 0.9, 0.3]
        @test AggregateAPI.softmax_weighted_vote(
            answers,
            scores;
            temperature=Inf,
        ) == AggregateAPI.majority_vote(answers)
        @test AggregateAPI.rank_weighted_vote(answers, scores; p=0.0) ==
              AggregateAPI.majority_vote(answers)
        @test AggregateAPI.filtered_vote(answers, scores; keep=1) ==
              AggregateAPI.best_of_n(answers, scores)
        @test AggregateAPI.logit_weighted_vote(
            answers,
            scores;
            transform="linear",
            threshold=0.0,
        ) == AggregateAPI.weighted_majority_vote(answers, scores)
        @test AggregateAPI.logit_weighted_vote(
            ["A", "B"],
            [5.0, -3.0];
            transform="linear",
            threshold=0.0,
        ) == "A"
        @test AggregateAPI.best_of_majority(
            ["A", "A", "B"],
            [0.2, 0.3, 0.9];
            alpha=1.0,
            aggregate="max",
        ) == "B" # an empty gate relaxes to all groups

        @test AggregateAPI._keep_count(0.07, 100) == 7
        @test AggregateAPI._keep_count(0.29, 100) == 29
        @test AggregateAPI._keep_count(0.1, 4) == 1
        @test AggregateAPI._keep_count(3, 2) == 2

        @test_throws ErrorException AggregateAPI.weighted_majority_vote(
            ["A"],
            [0.5];
            aggregate="median",
        )
        @test_throws ErrorException AggregateAPI.softmax_weighted_vote(
            ["A"],
            [0.5];
            temperature=0.0,
        )
        @test_throws ErrorException AggregateAPI.rank_weighted_vote(
            ["A"],
            [0.5];
            p=Inf,
        )
        @test_throws ErrorException AggregateAPI.logit_weighted_vote(
            ["A"],
            [1.0],
        )
        @test_throws ErrorException AggregateAPI.logit_weighted_vote(
            ["A"],
            [0.5];
            transform="sqrt",
        )
        @test_throws ErrorException AggregateAPI.filtered_vote(["A"], [0.5]; keep=0.0)
        @test_throws ErrorException AggregateAPI.filtered_vote(["A"], [0.5]; keep=true)
        @test_throws ErrorException AggregateAPI.majority_of_the_bests(
            ["A"],
            [0.5];
            m=0,
        )
        @test_throws ErrorException AggregateAPI.best_of_majority(
            ["A"],
            [0.5];
            alpha=1.1,
        )
    end

    @testset "shared input validation" begin
        @test_throws ErrorException AggregateAPI.majority_vote(String[])
        @test_throws ErrorException AggregateAPI.majority_vote(fill("A", 2, 2, 2))
        @test_throws ErrorException AggregateAPI.best_of_n(
            ["A" "B" "C"],
            [0.1 0.2],
        )
        @test_throws ErrorException AggregateAPI.weighted_majority_vote(
            ["A", "B"],
            nothing,
        )
        @test_throws ErrorException AggregateAPI.majority_vote([String[], String[]])
    end

    @testset "Python golden CGES" begin
        @test sprint(show, AggregateAPI.CGES_OTHER) == "CGES_OTHER"
        posterior = AggregateAPI._row_cges_posterior(["A", "B"], [0.8, 0.6])
        @test posterior["A"] ≈ 2.0 / 3.0 atol = 1e-14
        @test posterior["B"] ≈ 1.0 / 4.0 atol = 1e-14
        @test posterior[AggregateAPI.CGES_OTHER] ≈ 1.0 / 12.0 atol = 1e-14
        @test sum(values(posterior)) ≈ 1.0 atol = 1e-14

        @test AggregateAPI.cges_vote(
            ["A", "A", "B"],
            [0.7, 0.9, 0.6];
            return_index=true,
            return_score=true,
        ) == ("A", 1, 0.9)
        other, other_index, other_score = AggregateAPI.cges_vote(
            ["A"],
            [0.1];
            allow_other=true,
            return_index=true,
            return_score=true,
        )
        @test other === AggregateAPI.CGES_OTHER
        @test other_index == -1 && isnan(other_score)

        selected, indices, selected_scores = AggregateAPI.cges_vote(
            [["A", "B"], ["X", "X"]],
            [[0.8, 0.6], [0.7, 0.8]];
            return_index=true,
            return_score=true,
        )
        @test selected == ["A", "X"]
        @test indices == [0, 1]
        @test selected_scores == [0.8, 0.8]

        stopped, probability = AggregateAPI.cges_stop(
            ["A"],
            [0.9];
            threshold=0.8,
            return_prob=true,
        )
        @test stopped
        @test probability ≈ 0.9 atol = 1e-14
        stopped, probability = AggregateAPI.cges_stop(
            ["A"],
            [0.1];
            threshold=0.8,
            include_other=true,
            return_prob=true,
        )
        @test stopped
        @test probability ≈ 0.9 atol = 1e-14
        @test !AggregateAPI.cges_stop(
            Any[nothing, "A"],
            [0.0, 0.9];
            threshold=0.8,
            min_samples=2,
        )
        @test_throws ErrorException AggregateAPI.cges_vote(
            ["A"],
            [0.8];
            allow_other=1,
        )
        @test_throws ErrorException AggregateAPI.cges_stop(
            [["A"], ["B"]],
            [[0.8], [0.7]],
        )
        @test_throws ErrorException AggregateAPI.cges_stop(
            ["A"],
            [0.8];
            min_samples=true,
        )
    end

    @testset "Python golden KDE calibration and voting" begin
        calibration = AggregateAPI.fit_kde_vote_calibration(
            [0.8, 0.9, 0.1, 0.2],
            [1, 1, 0, 0];
            n_bins=2,
            bandwidth=0.5,
        )
        @test calibration.correct_logits ≈
              [1.3862943611198908, 2.1972245773362196] atol = 1e-14
        @test calibration.incorrect_logits ≈
              [-2.197224577336219, -1.3862943611198906] atol = 1e-14
        @test calibration.correct_bandwidth == 0.5
        @test calibration.incorrect_bandwidth == 0.5
        @test calibration.bin_edges == [-Inf, 0.8, Inf]
        @test calibration.bin_probability == [0.0, 1.0]
        @test calibration.n_bins == 2
        @test calibration.calibrated_probability([0.2, 0.7, 0.8, 0.95]) ==
              [0.0, 0.0, 1.0, 1.0]
        @test calibration.calibrated_probability(0.5) isa Float64
        @test calibration.log_density_ratio(0.5) isa Float64
        @test calibration.log_density_ratio([0.65])[1] ≈
              6.887000313265823 atol = 1e-12

        original_copy = calibration.correct_logits
        original_copy[1] = 99.0
        @test calibration.correct_logits[1] ≈ 1.3862943611198908 atol = 1e-14

        samples = AggregateAPI._logit.([0.3, 0.7])
        constant = AggregateAPI.KDEVoteCalibration(
            samples,
            samples,
            0.5,
            0.5,
            [-Inf, Inf],
            [0.6],
        )
        @test constant.weights([0.4, 0.7]; n_answers=3) ≈
              fill(log(3.0), 2) atol = 1e-14

        @test AggregateAPI.kde_weighted_vote(
            ["A", "A", "B"],
            [0.2, 0.2, 0.8],
            calibration;
            return_index=true,
            return_score=true,
        ) == ("B", 2, 0.8)
        selected, indices, selected_scores = AggregateAPI.kde_weighted_vote(
            [["A", "A", "B"], ["X", "Y", "Y"]],
            [[0.4, 0.7, 0.6], [0.8, 0.5, 0.6]],
            constant;
            return_index=true,
            return_score=true,
        )
        @test selected == ["A", "Y"]
        @test indices == [1, 2]
        @test selected_scores == [0.7, 0.6]

        @test_throws ErrorException AggregateAPI.fit_kde_vote_calibration(
            [0.8, 0.9, 0.1, 0.2],
            [1, 1, 0, 0];
            n_bins=true,
            bandwidth=0.5,
        )
        @test_throws ErrorException AggregateAPI.fit_kde_vote_calibration(
            [0.8, 0.8, 0.2, 0.3],
            [1, 1, 0, 0],
        )
        @test_throws ErrorException AggregateAPI.kde_weighted_vote(
            ["A", "B"],
            [0.5, 1.0],
            constant,
        )
        @test_throws ErrorException constant.weights([0.4]; n_answers=true)
    end

    @testset "Python golden online rules" begin
        expected = AGGREGATE_FIXTURE["online"]
        stop, probability = AggregateAPI.adaptive_consistency_stop(
            vcat(fill("A", 8), fill("B", 2));
            return_prob=true,
        )
        @test stop == expected["adaptive_consistency_stop"][1]
        @test probability ≈ expected["adaptive_consistency_stop"][2] atol = 1e-14
        stop, probability = AggregateAPI.adaptive_consistency_stop(
            ["A", "A", "B", "B"];
            return_prob=true,
        )
        @test stop == expected["adaptive_tie"][1]
        @test probability ≈ expected["adaptive_tie"][2] atol = 1e-14
        @test AggregateAPI.esc_stop(["A", "A", "A"]) == expected["esc_stop_true"]
        @test AggregateAPI.esc_stop(["A", "B", "A"]) == expected["esc_stop_false"]
        @test AggregateAPI.deepconf_stop_threshold(
            [1.0, 2.0, 3.0, 4.0, 5.0];
            keep=0.2,
        ) ≈ expected["deepconf_stop_threshold"] atol = 1e-14
        @test AggregateAPI.deepconf_online_stop(
            vcat(
                [Float64[0.0, -2.0] for _ in 1:3],
                [Float64[-4.0, -6.0] for _ in 1:3],
            ),
            2.0;
            window=3,
        ) == Int(expected["deepconf_online_stop"])

        dirichlet_stop, dirichlet_probability =
            AggregateAPI.adaptive_consistency_dirichlet_stop(
                vcat(fill("A", 5), fill("B", 2), ["C"]);
                return_prob=true,
            )
        @test !dirichlet_stop
        @test dirichlet_probability ≈ 0.8179649396053817 atol = 1e-10
        _, large_probability = AggregateAPI.adaptive_consistency_dirichlet_stop(
            vcat(fill("A", 1000), fill("B", 900), ["C"]);
            return_prob=true,
        )
        @test large_probability ≈ 0.9891042503731576 atol = 2e-9
        @test AggregateAPI._dirichlet_leader_probability([1000, 1000, 1000]) ==
              1.0 / 3.0

        # A dominant leader creates a very narrow integration boundary layer
        # near zero. Python/SciPy resolves this as probability 1.0; insufficient
        # adaptive depth used to return about 0.999999981 and reverse this stop.
        extreme_dirichlet = AggregateAPI.adaptive_consistency_dirichlet_stop(
            vcat(fill("A", 60), ["B", "C"]);
            threshold=0.99999999,
            return_prob=true,
        )
        @test extreme_dirichlet[1]
        @test extreme_dirichlet[2] ≈ 1.0 atol = 1e-14

        # Python/NumPy's exact seeded Monte Carlo result for this one-step model.
        crp_result = AggregateAPI.adaptive_consistency_crp_stop(
            ["A", "B"];
            horizon=3,
            n_alpha=100,
            n_simulations=1000,
            seed=0,
            return_prob=true,
        )
        @test crp_result == (false, 0.63924)
        @test crp_result == AggregateAPI.adaptive_consistency_crp_stop(
            ["A", "B"];
            horizon=3,
            n_alpha=100,
            n_simulations=1000,
            seed=0,
            return_prob=true,
        )
        @test AggregateAPI.adaptive_consistency_crp_stop(
            ["A", "B", "A"];
            horizon=3,
            return_prob=true,
        ) == (true, 1.0)
    end

    @testset "NumPy binomial and multinomial golden vectors" begin
        parameters = [
            (0, 0.2),
            (10, 0.0),
            (10, 0.2),
            (100, 0.2),
            (1000, 0.2),
            (1000, 0.8),
            (100_000, 0.5),
            (100, 0.7),
            (10, 0.8),
            (1000, 0.2),
        ]
        scalar_rng = Scorio._NumpyRNG(42)
        sequence_rng = Scorio._NumpyRNG((1, 2, 3))
        @test [
            Scorio._numpy_binomial!(scalar_rng, n, probability) for
            (n, probability) in parameters
        ] == [0, 0, 3, 19, 180, 806, 49_850, 74, 8, 182]
        @test [
            Scorio._numpy_binomial!(sequence_rng, n, probability) for
            (n, probability) in parameters
        ] == [0, 0, 2, 15, 211, 812, 50_194, 71, 9, 195]

        probabilities = [
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4],
            [0.01, 0.49, 0.5],
            [0.4, 0.6, 0.0],
        ]
        scalar_rng = Scorio._NumpyRNG(42)
        sequence_rng = Scorio._NumpyRNG((1, 2, 3))
        @test [Scorio._numpy_multinomial!(scalar_rng, 100, p) for p in probabilities] ==
              [[12, 19, 69], [35, 30, 35], [0, 58, 42], [46, 54, 0]]
        @test [
            Scorio._numpy_multinomial!(sequence_rng, 100, p) for p in probabilities
        ] == [[11, 15, 74], [30, 32, 38], [1, 51, 48], [39, 61, 0]]
        @test [Scorio._numpy_next_u64!(scalar_rng) for _ in 1:4] == UInt64[
            0x5eec9e8b80171029,
            0xed4078662ebb92af,
            0xa4d45831c814ce06,
            0xd2a0814d6704a0b6,
        ]
        @test [Scorio._numpy_next_u64!(sequence_rng) for _ in 1:4] == UInt64[
            0xbef02ff1e90b751c,
            0x53c43b10917fe517,
            0x256a1649d36fae45,
            0xdb048d380b885e24,
        ]
    end

    @testset "Python-generated CRP chunking goldens" begin
        @test AggregateAPI._crp_leader_probability(
            [7, 1];
            horizon=12,
            n_alpha=20,
            n_simulations=200,
            seed=7,
        ) == 1.0
        @test AggregateAPI._crp_leader_probability(
            [2, 2];
            horizon=12,
            n_alpha=20,
            n_simulations=200,
            seed=7,
        ) == 0.5665
        @test AggregateAPI._crp_leader_probability(
            [4, 1];
            horizon=10,
            n_alpha=12,
            n_simulations=100,
            seed=23,
        ) == 0.9933333333333333
        @test AggregateAPI._crp_leader_probability(
            [3, 2, 1];
            horizon=15,
            n_alpha=7,
            n_simulations=19,
            seed=42,
        ) == 0.7969924812030075

        # This profile has chunk_size=8192. Exercise both the exact boundary
        # and the first run in the next chunk, whose interleaved draw order is
        # observable in the golden probability.
        @test AggregateAPI._crp_leader_probability(
            [3, 2, 1];
            horizon=15,
            n_alpha=1,
            n_simulations=8192,
            seed=314,
        ) == 0.743408203125
        @test AggregateAPI.adaptive_consistency_crp_stop(
            vcat(fill("A", 3), fill("B", 2), ["C"]);
            horizon=15,
            n_alpha=1,
            n_simulations=8193,
            seed=314,
            return_prob=true,
        ) == (false, 0.7433174661296228)

        # A large horizon reduces the memory-bounded chunk size to 250.
        @test AggregateAPI._crp_leader_probability(
            [4, 3];
            horizon=1000,
            n_alpha=1,
            n_simulations=250,
            seed=2718,
        ) == 0.62
        @test AggregateAPI._crp_leader_probability(
            [4, 3];
            horizon=1000,
            n_alpha=1,
            n_simulations=251,
            seed=2718,
        ) == 0.6215139442231076
    end

    @testset "online boundaries and validation" begin
        stop, probability = AggregateAPI.adaptive_consistency_stop(
            Any[nothing, "", NaN];
            return_prob=true,
        )
        @test !stop && probability == 0.0
        @test !AggregateAPI.adaptive_consistency_stop(Any[])
        @test_throws ErrorException AggregateAPI.adaptive_consistency_stop(
            ["A"];
            threshold=1.0,
        )

        @test AggregateAPI.esc_stop([3, 3, 3])
        @test !AggregateAPI.esc_stop([3, 3, 7])
        @test !AggregateAPI.esc_stop(Any["A", nothing, "A"])
        @test !AggregateAPI.esc_stop(Any[])

        all_one = fill([0.0, -2.0], 4) # token confidence exactly 1
        @test AggregateAPI.deepconf_online_stop(all_one, 1.0; window=2) === nothing
        later_crossing = vcat(fill([-4.0, -6.0], 2), fill([0.0, -2.0], 2))
        @test AggregateAPI.deepconf_online_stop(later_crossing, 2.0; window=2) == 3
        @test AggregateAPI.deepconf_online_stop(all_one, 2.0; window=99) == 3

        @test AggregateAPI.deepconf_stop_threshold([1.0, 5.0, 3.0]; keep=1.0) == 1.0
        @test_throws ErrorException AggregateAPI.deepconf_stop_threshold([])
        @test_throws ErrorException AggregateAPI.deepconf_stop_threshold([1.0, Inf])
        @test_throws ErrorException AggregateAPI.deepconf_stop_threshold(
            [[1.0], [2.0, 3.0]],
        )
        @test_throws ErrorException AggregateAPI.deepconf_stop_threshold(
            [1.0];
            keep=0.0,
        )
        @test_throws ErrorException AggregateAPI.deepconf_online_stop(
            all_one,
            1.0;
            window=0,
        )
    end
end
