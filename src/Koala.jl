#__precompile__()
module Koala

# new: 
export @more, keys_ordered_by_values, bootstrap_resample_of_mean, params
export load_boston, load_ames, datanow
export fit!, predict, rms, rmsl, rmslp1, err, transform, inverse_transform
export SupervisedMachine, ConstantRegressor
export TransformerMachine, IdentityTransformer, FeatureSelector
export default_transformer_X, default_transformer_y, clean!
export Machine
export learning_curve, cv, @colon, @curve, @pcurve

# for use in this module:
import DataFrames: DataFrame, AbstractDataFrame, names
import CSV
import StatsBase: sample

# extended in this module:
import Base: show, showall, isempty, split

# constants:
const COLUMN_WIDTH = 24 # for displaying dictionaries with `showall`
const srcdir = dirname(@__FILE__) # the full path for this file

# functions to be extended (provided methods) in dependent packages:
function fit end
function predict end
function setup end
function transform end
function inverse_transform end
function default_transformer_X end
function default_transformer_y end

## Some general assorted helpers:

""" macro shortcut for showing all of last REPL expression"""
macro more()
    esc(quote
        showall(Main.ans)
    end)
end

""" a version of warn that only warns given a non-empty string."""
function softwarn(str)
    if !isempty(str)
        warn(str)
    end
end
    
"""Load a well-known public regression dataset with nominal features."""
function load_boston()
    df = CSV.read(joinpath(srcdir, "data", "Boston.csv"))
    features = filter(names(df)) do f
        f != :MedV
    end
    X = df[features] 
    y = df[:MedV]
    return X, y 
end

"""Load a reduced version of the well-known Ames Housing dataset,
having six numerical and six categorical features."""
function load_ames()
    df = CSV.read(joinpath(srcdir, "data", "reduced_ames.csv"))
    features = filter(names(df)) do f
        f != :target
    end
    X = df[features] 
    y = exp.(df[:target])
    return X, y 
end
datanow=load_ames

""" `showall` method for dictionaries with markdown format"""
function Base.showall(stream::IO, d::Dict)
    print(stream, "\n")
    println(stream, "key                     | value")
    println(stream, "-"^COLUMN_WIDTH * "|" * "-"^COLUMN_WIDTH)
    kys = keys(d) |> collect |> sort
    for k in kys
        key_string = string(k)*" "^(max(0,COLUMN_WIDTH - length(string(k))))
        println(stream, key_string * "|" * string(d[k]))
    end
end

function keys_ordered_by_values(d::Dict{T,S}) where {T, S<:Real}

    items = collect(d) # 1d array containing the (key, value) pairs
    sort!(items, by=pair->pair[2], alg=QuickSort)

    return T[pair[1] for pair in items]

end

"""
## `bootstrap_resample_of_mean(v; n=10^6)`

Returns a vector of `n` estimates of the mean of the distribution
generating the samples in `v` from `n` bootstrap resamples of `v`.

"""
function bootstrap_resample_of_mean(v; n=10^6)

    n_samples = length(v)
    mu = mean(v)

    simulated_means = Array{Float64}(n)

    for i in 1:n
        pseudo_sample = sample(v, n_samples, replace=true)
        simulated_means[i]=mean(pseudo_sample)
    end
    return simulated_means
end


## `BaseType` - base type for external `Koala` structs in dependent packages.  

abstract type BaseType end

""" Return a dictionary of values keyed on the fields of specified object."""
function params(object::BaseType)
    value_given_parameter  = Dict{Symbol,Any}()
    for name in fieldnames(object)
        if isdefined(object, name)
            value_given_parameter[name] = getfield(object,name)
        else
            value_given_parameter[name] = "undefined"
        end 
    end
    return value_given_parameter
end

""" Extract type parameters of the type of an object."""
function type_parameters(object)
    params = typeof(object).parameters
    ret =[]
    for p in params
        if isa(p, Type)
            push!(ret, p.name.name)
        else
            push!(ret, p)
        end
    end
    return ret
end

""" Output plain/text representation to specified stream. """
function Base.show(stream::IO, object::BaseType)
    abbreviated(n) = "…"*string(n)[end-2:end]
    type_params = type_parameters(object)
    if isempty(type_params)
        type_string = ""
    else
        type_string = string("{", ["$T," for T in type_params]..., "}")
    end
    print(stream, string(typeof(object).name.name,
                         type_string,
                         "@", abbreviated(hash(object))))
end

""" 
Output detailed plain/text representation to specified stream. If
`dic` is unspecified then the parameter dictionary (ie dictionary
keyed on `object`'s fields) is displayed .To display an altered
dictionary for some subtype of `BaseType` overload the two-argument
version of this method and call *this* method with the altered
dictionary.

"""
function Base.showall(stream::IO, object::BaseType;
                      dic::Dict{Symbol,Any}=Dict{Symbol,Any}())
    if isempty(dic)
        dic = params(object)
    end
    show(stream, object)
    println(stream)
    showall(stream, dic)
end


## Abstract model types

# Here *models* are simply small data structures storing parameters
# describing *what* is to be done with some data (rather than *how*
# this is to be done). In the simplest case, the *type* of the model
# itself, may suffice to deptermine what is to be done (see the model
# type `UnivariateStandardizer` defined below). As a more complex
# example, consider the learning task requiring one-hot encoding the
# categorical features of some input patterns, standardizing the
# corresonding target values, and training a tree gradient booster on
# that data. The model parameters (ie its fields) will then involve
# flags describing the transformations required, and parameters such
# as the degree of L2 regularization of the booster weights.
abstract type Model <: BaseType end

# An abstract type for transformation models:
abstract type Transformer <: Model end

# Supervised model types are collected together according to their
# corresponding predictor type, `P`. For example, the model type
# `ConstantRegressor` (for predicting the historical target average on
# all input patterns) has predictor type `Float64`. One reason for
# this is that ensemble methods involve ensembles of predictors whose
# type should be known ahead of time.
abstract type SupervisedModel{P} <: Model end
abstract type Regressor{P} <: SupervisedModel{P} end
abstract type Classifier{P} <: SupervisedModel{P} end


## Loss and low-level error functions

function rms(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in rows
        ret += (y[i] - yhat[i])^2
    end
    return sqrt(ret/length(rows))
end

function rms(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        ret += (y[i] - yhat[i])^2
    end
    return sqrt(ret/length(y))
end

function rmsl(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in rows
        ret += (log(y[i]) - log(yhat[i]))^2
    end
    return sqrt(ret/length(rows))
end

function rmsl(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        ret += (log(y[i]) - log(yhat[i]))^2
    end
    return sqrt(ret/length(y))
end

function rmslp1(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in rows
        ret += (log(y[i] + 1) - log(yhat[i] + 1))^2
    end
    return sqrt(ret/length(rows))
end

function rmslp1(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        ret += (log(y[i] + 1) - log(yhat[i] + 1))^2
    end
    return sqrt(ret/length(y))
end

function err(rgs::Regressor, predictor, X, y, rows,
             parallel, verbosity, loss::Function=rms)
    return loss(y[rows], predict(rgs, predictor, X, rows, parallel, verbosity))
end

function err(rgs::Regressor, predictor, X, y,
             parallel, verbosity, loss::Function=rms)
    return loss(y, predict(rgs, predictor, X, parallel, verbosity))
end

# TODO: Classifier loss and error functions


## Machines

# A *machine* is a larger structure wrapping around the model extra
# data needed to carry out, at various stages, a data processing (eg
# learning) task. Certain parameters of the model will have to be
# tuned during training and we call these *hyperparameters*. If a
# preliminary part of the task (eg one-hot encoding inputs) does not
# depend on the values of hyperparameters, then these tasks will be
# performed already during the machine's construction (instantiation).
abstract type Machine <: BaseType end


## Machines for transforming

struct TransformerMachine{T <: Transformer} <: Machine

    transformer::T
    scheme

    function TransformerMachine{T}(transformer::T, X;
                                   parallel::Bool=true,
                                   verbosity::Int=1, args...) where T <: Transformer
        scheme = fit(transformer, X, parallel, verbosity; args...)
        return new{T}(transformer, scheme)
    end

end

TransformerMachine(transformer::T, X; args...) where T <: Transformer =
    TransformerMachine{T}(transformer, X; args...)

Machine(transformer::Transformer, X; args...) =
    TransformerMachine(transformer, X; args...)

function Base.showall(stream::IO, mach::TransformerMachine)
    dict = params(mach)
    showall(stream, mach, dic=dict)
    println(stream, "\n## Transformer detail:")
    showall(stream, mach.transformer)
    println(stream, "\n##Scheme detail:")
    showall(stream, mach.scheme)
end

function transform(mach::TransformerMachine, X; parallel=true, verbosity=1)
    return transform(mach.transformer, mach.scheme, X)
end

function inverse_transform(mach::TransformerMachine, X; parallel=true, verbosity=1)
    return inverse_transform(mach.transformer, mach.scheme, X)
end


## Commonly used transformers

# a null transformer, useful as default keyword argument
# (has no fit, transform or inverse_transform methods):
struct EmptyTransformer <: Transformer end

# no transformer is empty except the null transformer:
Base.isempty(transformer::Transformer) = false
Base.isempty(transformer::EmptyTransformer) = true

# for (a) remembering the features used in `fit` (calibration) and
# selecting only those on tranforming new data frames; or (b) select only
# feature labels specified in the tranformer:
mutable struct FeatureSelector <: Transformer
    features::Vector{Symbol} # empty means do (a) above
end
FeatureSelector(;features=Symbol[]) = FeatureSelector(features)
function fit(transformer::FeatureSelector, X, parallel, verbosity)
    if isempty(transformer.features)
        return names(X)
    else
        return transformer.features
    end
end
function transform(transformer::FeatureSelector, scheme, X)
    issubset(Set(scheme), Set(names(X))) || throw(DomainError)
    return X[scheme]
end 

# identity transformations:
struct IdentityTransformer <: Transformer end
fit(transformer::IdentityTransformer, y, parallel, verbosity) = nothing
transform(transformer::IdentityTransformer, scheme, y) = y
inverse_transform(transformer::IdentityTransformer, scheme, y) = y


## Machines for supervised learning

mutable struct SupervisedMachine{P, M <: SupervisedModel{P}} <: Machine

    model::M
    transformer_X::Transformer
    transformer_y::Transformer
    scheme_X
    scheme_y
    n_iter::Int
    Xt
    yt
    predictor::P
    report
    cache

    function SupervisedMachine{P, M}(
        model::M,
        X::AbstractDataFrame,
        y::AbstractVector,
        train_rows::AbstractVector{Int}; # for defining data transformations
        features = Symbol[],
        transformer_X=EmptyTransformer(),
        transformer_y=EmptyTransformer(),
        verbosity::Int=1) where {P, M <: SupervisedModel{P}}

        # check dimension match:
        size(X,1) == length(y) || throw(DimensionMismatch())

        # check valid `features`; if none provided, take to be all
        if isempty(features)
            features = names(X)
        end
        allunique(features) || error("Duplicate features.")
        issubset(Set(features), Set(names(X))) || error("Invalid feature vector.")

        # assign transformers if not provided:
        if isempty(transformer_X)
            transformer_X = default_transformer_X(model)
        end
        if isempty(transformer_y)
            transformer_y = default_transformer_y(model)
        end            

        mach = new{P, M}(model::M)
        mach.transformer_X = transformer_X
        mach.transformer_y = transformer_y
        mach.scheme_X = fit(mach.transformer_X, X[train_rows, features],
                            true, verbosity - 1)
        mach.scheme_y = fit(mach.transformer_y, y[train_rows], verbosity - 1, 1)
        mach.Xt = transform(mach.transformer_X, mach.scheme_X, X)
        mach.yt = transform(mach.transformer_y, mach.scheme_y, y)
        mach.n_iter = 0
        mach.report = Dict{Symbol,Any}()

        return mach
    end

end

function SupervisedMachine(model::M, X, y,
                           train_rows; args...) where {P, M <: SupervisedModel{P}}
    return SupervisedMachine{P, M}(model, X, y, train_rows; args...)
end

Machine(model::SupervisedModel, X, y, train_rows; args...) =
    SupervisedMachine(model, X, y, train_rows; args...)

function Base.show(stream::IO, mach::SupervisedMachine)
    abbreviated(n) = "..."*string(n)[end-2:end]
    type_string = string("SupervisedMachine{", typeof(mach.model).name.name, "}")
    print(stream, type_string, "@", abbreviated(hash(mach)))
end

function Base.showall(stream::IO, mach::SupervisedMachine)
    dic = params(mach)
    report_items = sort(collect(keys(dic[:report])))
    dic[:report] = "Dic with keys: $report_items"
    dic[:Xt] = string(typeof(mach.Xt), " of shape ", size(mach.Xt))
    dic[:yt] = string(typeof(mach.yt), " of shape ", size(mach.yt))
    delete!(dic, :cache)
    showall(stream, mach, dic=dic)
    println(stream, "\n## Model detail:")
    showall(stream, mach.model)
end

function fit!(mach::SupervisedMachine, rows;
              add=false, verbosity=1, parallel=true, args...)
    verbosity < 0 || softwarn(clean!(mach.model))
    if !add
        mach.n_iter = 0
    end
    if  mach.n_iter == 0 
        mach.cache = setup(mach.model, mach.Xt, mach.yt, rows, mach.scheme_X,
                           parallel, verbosity)
    end
    mach.predictor, report, mach.cache =
        fit(mach.model, mach.cache, add, parallel, verbosity; args...)
    merge!(mach.report, report)
    if isdefined(mach.model, :n)
        mach.n_iter += mach.model.n
    else
        mach.n_iter = 1
    end
    return mach
end

# When rows are not specified, the cache is not recalculated. Ie,
# `setup` is skipped:
function fit!(mach::SupervisedMachine;
              add=false, verbosity=1, parallel=true, args...)
    verbosity < 0 || softwarn(clean!(mach.model))
    if !isdefined(mach, :cache)
        error("You must specify training rows in the first call to fit!\n"*
              "E.g., fit!(mach, train_rows).")
    end
    if !add
        mach.n_iter = 0
    end
    mach.predictor, report, mach.cache =
        fit(mach.model, mach.cache, add, parallel, verbosity; args...)
    merge!(mach.report, report)
    if isdefined(mach.model, :n)
        mach.n_iter += mach.model.n
    else
        mach.n_iter = 1
    end
    return mach
end

function predict(mach::SupervisedMachine, X, rows; parallel=true, verbosity=1)
    mach.n_iter > 0 || error(string(mach, " has not been fitted."))
    Xt = transform(mach.transformer_X, mach.scheme_X, X[rows,:])
    yt = predict(mach.model, mach.predictor, Xt, parallel, verbosity)
    return inverse_transform(mach.transformer_y, mach.scheme_y, yt)
end

function predict(mach::SupervisedMachine, X; parallel=true, verbosity=1)
    mach.n_iter > 0 || error(string(mach, " has not been fitted."))
    Xt = transform(mach.transformer_X, mach.scheme_X, X)
    yt = predict(mach.model, mach.predictor, Xt, parallel, verbosity)
    return inverse_transform(mach.transformer_y, mach.scheme_y, yt)
end

function err(mach::SupervisedMachine, test_rows;
             loss=rms, parallel=false, verbosity=0, raw=false)

    mach.n_iter > 0 || error("Attempting to predict using untrained machine.")

    !raw || verbosity < 0 || warn("Reporting errors for *transformed* target. "*
                                    "Use `raw=false` to report true errors.")

    # transformed version of target predictions:
    raw_predictions = predict(mach.model, mach.predictor, mach.Xt, test_rows,
                            parallel, verbosity) 

    if raw # return error on *transformed* target, which is faster
        return loss(raw_predictions, mach.yt[test_rows])
    else  # return error for untransformed target
        return loss(inverse_transform(mach.transformer_y, mach.scheme_y, raw_predictions),
           inverse_transform(mach.transformer_y, mach.scheme_y, mach.yt[test_rows]))
    end
end


## `SupervisedModel`  fall-back methods

# default default_transformers:
default_transformer_X(model::SupervisedModel) = FeatureSelector()
default_transformer_y(model::SupervisedModel) = IdentityTransformer()

# for enforcing model parameter invariants:
clean!(model::SupervisedModel) = "" # no checks, emtpy message

# to allow for an extra `rows` argument (depreceate?):
setup(model::SupervisedModel, Xt, yt, rows, scheme_X, parallel, verbosity) =
    setup(model, Xt[rows,:], yt[rows], scheme_X, parallel, verbosity) 
predict(model::SupervisedModel, predictor, Xt, rows, parallel, verbosity) =
    predict(model, predictor, Xt[rows,:], parallel, verbosity)


## Constant-predicting regressor (a `Regressor` example)

# Note: To test iterative methods, we give the following simple regressor
# model a "bogus" field `n` for counting the number of iterations (which
# make no difference to predictions):

mutable struct ConstantRegressor <: Regressor{Float64}
    n::Int 
end

ConstantRegressor() = ConstantRegressor(1)

function setup(rgs::ConstantRegressor, X, y, scheme_X, parallel, verbosity)
    return mean(y)
end
    
function fit(rgs::ConstantRegressor, cache, add, parallel, verbosity)
    predictor = cache
    report = Dict{Symbol, Any}()
    report[:mean] = predictor 
    return predictor, report, cache
end

function predict(rgs::ConstantRegressor, predictor, X, parallel, verbosity)
    return  Float64[predictor for i in 1:size(X,1)]
end

    
## Validation tools

"""
## split(rows::AbstractVector{Int}, fractions...)

Then splits the vector `rows` into a tuple of `Vector{Int}` objects
whose lengths are given by the corresponding `fractions` of
`length(rows)`. The last fraction is not provided, as it
is inferred from the preceding ones. So, for example,

    julia> split(1:1000, 0.2, 0.7)
    (1:200, 201:900, 901:1000)

"""
function Base.split(rows::AbstractVector{Int}, fractions...)
    rows = collect(rows)
    rowss = []
    if sum(fractions) >= 1
        throw(DomainError)
    end
    n_patterns = length(rows)
    first = 1
    for p in fractions
        n = round(Int, p*n_patterns)
        n == 0 ? (Base.warn("A split has only one element"); n = 1) : nothing
        push!(rowss, rows[first:(first + n - 1)])
        first = first + n
    end
    if first > n_patterns
        Base.warn("Last vector in the split has only one element.")
        first = n_patterns
    end
    push!(rowss, rows[first:n_patterns])
    return tuple(rowss...)
end

"""
## `function learning_curve(mach::SupervisedMachine, train_rows, test_rows,
##                      range; restart=true, loss=rms, raw=false, parallel=true,
##                      verbosity=1, fit_args...)`

    u,v = learning_curve(mach, test_rows, 1:10:200)
    plot(u, v)

Assming, say, `Plots` is installed, the above produces a plot of the
RMS error for the machine `mach`, on the test data with rows
`test_rows`, against the number of iterations `n` of the algorithm it
implements (assumed to be iterative). Here `n` ranges over `1:10:200`
and training is performed using `train_rows`. For parallization, the
value of the optional keyword `parallel` is passed to each call to
`fit`, along with any other keyword arguments `fit_args` that `fit`
supports.

"""
function learning_curve(mach::SupervisedMachine, train_rows, test_rows,
                        range; restart=true, loss=rms, raw=false, parallel=true,
                        verbosity=1, fit_args...) 

    isdefined(mach.model, :n) || error("$(mach.model) does not support iteration.")

    # save to be reset at end:
    old_n = mach.model.n
    
    !raw || verbosity < 0 ||
        warn("Reporting errors for *transformed* target. Use `raw=false` "*
             " to report true errors.")
    
    range = collect(range)
    sort!(range)
    
    if restart
        mach.n_iter = 0
        mach.cache = setup(mach.model, mach.Xt, mach.yt, train_rows, mach.scheme_X,
                           parallel, verbosity - 1) 
    end

    n_iter_list = Float64[]
    errors = Float64[]

    filter!(range) do x
        x > mach.n_iter
    end

    while !isempty(range)
        verbosity < 1 || print("\rNext iteration number: ", range[1]) 
        # set number of iterations for `fit` call:
        mach.model.n = range[1] - mach.n_iter
        if mach.n_iter == 0 # then use add=false
            mach.predictor, report, mach.cache =
                fit(mach.model, mach.cache, false, parallel, verbosity - 1; fit_args...)
        else # use add=true
            mach.predictor, report, mach.cache =
                fit(mach.model, mach.cache, true, parallel, verbosity - 1; fit_args...)
        end
        mach.n_iter += mach.model.n
        push!(n_iter_list, mach.n_iter)
        push!(errors, err(mach, test_rows, raw=raw, loss=loss))
        filter!(range) do x
            x > mach.n_iter
        end
    end

    verbosity < 1 || println("\nLearning curve added to machine report.")
    
    mach.report[:learning_curve] = (n_iter_list, errors)
    
    mach.model.n = old_n
    
    return n_iter_list, errors

end

""" 
## `cv(mach::SupervisedMachine, rows; n_folds=9, loss=rms, parallel=true, verbosity=1, raw=false, randomize=false)`

Return a list of cross-validation root-mean-squared errors for
patterns with row indices in `rows`, an iterator that is initially
randomized when an optional parameter `randomize` is set to `true`.

"""
function cv(mach::SupervisedMachine, rows; n_folds=9, loss=rms,
             parallel=true, verbosity=1, raw=false, randomize=false)

    !raw || verbosity < 0 ||
        warn("Reporting errors for *transformed* target. Use `raw=false` "*
             " to report true errors.")

    n_samples = length(rows)
    if randomize
         rows = sample(collect(rows), n_samples, replace=false)
    end
    k = floor(Int,n_samples/n_folds)

    # function to return the error for the fold `rows[f:s]`:
    function get_error(f, s)
        test_rows = rows[f:s]
        train_rows = vcat(rows[1:(f - 1)], rows[(s + 1):end])
        fit!(mach, train_rows; parallel=false, verbosity=0)
        return err(mach, test_rows;
                   parallel=false, verbosity=verbosity - 1,
                   raw=raw)
    end

    firsts = 1:k:((n_folds - 1)*k + 1) # itr of first test_rows index
    seconds = k:k:(n_folds*k)          # itr of ending test_rows index

    if parallel && nworkers() > 1
        if verbosity > 0
            println("Distributing cross-validation computation "*
                    "among $(nworkers()) workers.")
        end
        return @parallel vcat for n in 1:n_folds
            Float64[get_error(firsts[n], seconds[n])]
        end
    else
        errors = Array{Float64}(n_folds)
        for n in 1:n_folds
            verbosity < 1 || print("\rfold: $n")
            errors[n] = get_error(firsts[n], seconds[n])
        end
        verbosity < 1 || println()
        return errors
    end

end

macro colon(p)
    Expr(:quote, p)
end

"""
## `@curve`

The code, 
 
    @curve var range code 

evaluates `code`, replacing appearances of `var` therein with each
value in `range`. The range and corresponding evaluations are returned
as a tuple of arrays. For example,

    @curve  x 1:3 (x^2 + 1)

evaluates to 

    ([1,2,3], [2, 5, 10])

This is convenient for plotting functions using, eg, the `Plots` package:

    plot(@curve x 1:3 (x^2 + 1))

A macro `@pcurve` parallelizes the same behaviour.  A two-variable
implementation is also available, operating as in the following
example:

    julia> @curve x [1,2,3] y [7,8] (x + y)
    ([1,2,3],[7 8],[8.0 9.0; 9.0 10.0; 10.0 11.0])

    julia> ans[3]
    3×2 Array{Float64,2}:
      8.0   9.0
      9.0  10.0
     10.0  11.0

N.B. The second range is returned as a *row* vector for consistency
with the output matrix. This is also helpful when plotting, as in:

    julia> u1, u2, A = @curve x linspace(0,1,100) α [1,2,3] x^α
    julia> u2 = map(u2) do α "α = "*string(α) end
    julia> plot(u1, A, label=u2)

which generates three superimposed plots - of the functions x, x^2 and x^3 - each
labels with the exponents α = 1, 2, 3 in the legend.

"""
macro curve(var1, range, code)
    quote
        output = []
        N = length($(esc(range)))
        for i in eachindex($(esc(range)))
            $(esc(var1)) = $(esc(range))[i]
            print((@colon $(esc(var1))), "=", $(esc(var1)), "                    \r")
            flush(STDOUT)
            # print(i,"\r"); flush(STDOUT) 
            push!(output, $(esc(code)))
        end
        collect($(esc(range))), [x for x in output]
    end
end

macro curve(var1, range1, var2, range2, code)
    quote
        output = Array{Float64}(length($(esc(range1))), length($(esc(range2))))
        for i1 in eachindex($(esc(range1)))
            $(esc(var1)) = $(esc(range1))[i1]
            for i2 in eachindex($(esc(range2)))
                $(esc(var2)) = $(esc(range2))[i2]
                # @dbg $(esc(var1)) $(esc(var2))
                print((@colon $(esc(var1))), "=", $(esc(var1)), " ")
                print((@colon $(esc(var2))), "=", $(esc(var2)), "                    \r")
                flush(STDOUT)
                output[i1,i2] = $(esc(code))
            end
        end
        collect($(esc(range1))), collect($(esc(range2)))', output
    end
end

macro pcurve(var1, range, code)
    quote
        N = length($(esc(range)))
        pairs = @parallel vcat for i in eachindex($(esc(range)))
            $(esc(var1)) = $(esc(range))[i]
            print((@colon $(esc(var1))), "=", $(esc(var1)), "                    \r")
            flush(STDOUT)
            print(i,"\r"); flush(STDOUT) 
            [( $(esc(range))[i], $(esc(code)) )]
        end
        sort!(pairs, by=first)
        collect(map(first,pairs)), collect(map(last, pairs))
    end
end

    

end # module
