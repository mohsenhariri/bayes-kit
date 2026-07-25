"""Python-compatible positional-or-keyword forwarding for Eval optionals."""

function _eval_forward_arg_name(argument)
    if argument isa Symbol
        return argument
    elseif argument isa Expr && argument.head == :(::)
        return argument.args[1]
    end
    error("unsupported forwarding argument: $argument")
end

"""
    @_eval_positional_or_keyword f(required...; optional=default, ...)

Generate one forwarding method for every proper positional prefix of the
optional parameters.  Consequently each optional can be supplied in its
Python position, by its Python keyword name, or through a positional prefix
followed by keywords.  The existing full-positional method remains the single
implementation of the calculation.
"""
macro _eval_positional_or_keyword(signature)
    signature isa Expr && signature.head == :call ||
        error("expected a function-call signature")

    function_name = signature.args[1]
    parameters_index = findfirst(
        argument -> argument isa Expr && argument.head == :parameters,
        signature.args,
    )
    isnothing(parameters_index) && error("expected optional keyword parameters")

    optional_parameters = signature.args[parameters_index].args
    required_parameters = Any[
        signature.args[index] for index in 2:length(signature.args)
        if index != parameters_index
    ]
    required_names = _eval_forward_arg_name.(required_parameters)
    optional_arguments = Any[parameter.args[1] for parameter in optional_parameters]
    optional_names = _eval_forward_arg_name.(optional_arguments)

    definitions = Any[]
    for prefix_length in 0:(length(optional_parameters) - 1)
        remaining_keywords = optional_parameters[(prefix_length + 1):end]
        positional_prefix = optional_arguments[1:prefix_length]

        call_arguments = Any[function_name]
        push!(call_arguments, Expr(:parameters, remaining_keywords...))
        append!(call_arguments, required_parameters)
        append!(call_arguments, positional_prefix)

        forwarding_signature = Expr(:call, call_arguments...)
        forwarding_call = Expr(
            :call,
            function_name,
            required_names...,
            optional_names...,
        )
        push!(definitions, Expr(:(=), forwarding_signature, forwarding_call))
    end

    return esc(Expr(:block, definitions...))
end

@_eval_positional_or_keyword avg(R; w=nothing)
@_eval_positional_or_keyword avg_ci(R; w=nothing, confidence::Real=0.95, bounds=nothing)

@_eval_positional_or_keyword bayes(R; w=nothing, R0=nothing)
@_eval_positional_or_keyword bayes_ci(
    R;
    w=nothing,
    R0=nothing,
    confidence::Real=0.95,
    bounds=nothing,
)

@_eval_positional_or_keyword pass_at_k_ci(
    R,
    k::Integer;
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
@_eval_positional_or_keyword pass_hat_k_ci(
    R,
    k::Integer;
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
@_eval_positional_or_keyword g_pass_at_k_ci(
    R,
    k::Integer;
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
@_eval_positional_or_keyword g_pass_at_k_tau_ci(
    R,
    k::Integer,
    tau::Real;
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
@_eval_positional_or_keyword mg_pass_at_k_ci(
    R,
    k::Integer;
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)

@_eval_positional_or_keyword auc_at_k_ci(
    R,
    k::Integer;
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
@_eval_positional_or_keyword maj_at_k_ci(
    R,
    k::Integer;
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)

@_eval_positional_or_keyword max_at_k(R, k::Integer; w=nothing)
@_eval_positional_or_keyword max_at_k_ci(
    R,
    k::Integer;
    w=nothing,
    R0=nothing,
    confidence::Real=0.95,
    bounds=nothing,
)

@_eval_positional_or_keyword geom_at_k(
    R,
    k::Integer;
    pass_power::Real=0.5,
    unanimous_power::Real=0.5,
)
@_eval_positional_or_keyword geom_ds_at_k(
    R,
    k::Integer;
    pass_power::Real=0.5,
    unanimous_power::Real=0.5,
)
@_eval_positional_or_keyword geo_spectrum_at_k(
    R,
    k::Integer;
    lam::Real=0.5,
    weights=nothing,
    lambda_=nothing,
)

@_eval_positional_or_keyword threshold_spectrum_at_k_ci(
    R,
    k::Integer,
    weights;
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
@_eval_positional_or_keyword geom_at_k_ci(
    R,
    k::Integer;
    pass_power::Real=0.5,
    unanimous_power::Real=0.5,
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
@_eval_positional_or_keyword geom_ds_at_k_ci(
    R,
    k::Integer;
    pass_power::Real=0.5,
    unanimous_power::Real=0.5,
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
@_eval_positional_or_keyword geo_spectrum_at_k_ci(
    R,
    k::Integer;
    lam::Real=0.5,
    weights=nothing,
    lambda_=nothing,
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
@_eval_positional_or_keyword geo_spectrum_star_at_k_ci(
    R,
    k::Integer;
    confidence::Real=0.95,
    bounds=(0.0, 1.0),
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
