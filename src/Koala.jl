module Koala

# new: 
export @more, @dbg, @colon, keys_ordered_by_values, params, partition
export bootstrap_resample_of_mean, bootstrap_resample_of_median
export load_boston, load_ames, load_iris, datanow
export hasmissing, countmissing, ismissingtype, purify
export fit!, predict, rms, rmsl, rmslp1, rmsp, err, transform, inverse_transform
export ConstantRegressor
export IdentityTransformer, FeatureSelector
export default_transformer_X, default_transformer_y, clean!
export Machine
export learning_curve, cv, compete, @colon, @curve, @pcurve
export split_seen_unseen
export bootstrap_histogram_of_mean, bootstrap_histogram_of_mean!, PlotableDict
export bootstrap_histogram_of_median, bootstrap_histogram_of_median!

# for use in this module:
using Statistics
import DataFrames: DataFrame, AbstractDataFrame, names, eltypes, describe
import CSV
import StatsBase: sample, countmap
import HypothesisTests: OneSampleTTest, pvalue
# import ROC
using  RecipesBase # for plotting recipes
import Distributed: @distributed, nworkers

# reexport:
export describe, countmap, names, eltypes

# extended in this module:
import Base: show, isempty, split

# constants:
const COLUMN_WIDTH = 24           # for displaying dictionaries with `show`
const DEFAULT_SHOW_DEPTH = 1      # how deep to display fields of `BaseType` objects
const LINE_SEPARATOR = ""         # "````" what to put on show separation lines
const srcdir = dirname(@__FILE__) # the full path for this file

# functions to be extended (provided methods) in dependent packages:
function fit end
function predict end
function setup end
function transform end
function inverse_transform end
function default_transformer_X end
function default_transformer_y end

## ON TERMINOLOGY AND THE MAIN TYPES

# 1. *Transformers and schemes*. A subtype of `Transformer` is a
# `struct` identifying a type of data transformation whose objects
# (*transformers*) store associated hyperparemeters. For example, a
# `ToIntTransformer` object (defined in KoalaTransformer.jl) relabels
# non-numeric vectors with integer vectors, the first integer being
# given by the value of the hyperparameter `initial_label` (0 or 1,
# say). A transformer does *not* store the actual data encoding the
# transformation (the relabelling in our example). This information
# (called a *scheme*) is part of a transformer *machine* (see below).

# 2. *Models and predictors.* A subtype of `Model` is a `struct`
# associated with some learning algorithm whose objects (*models*)
# store associated hyperparameters. For example, a `TreeRegressor`
# object (defined in KoalaTrees.jl) has fields describing the growth
# of a decision tree, such as the minimim number of training patterns
# triggering a node split. Such parameters are distinct from variables
# that are *learned* by the model, such as a decision tree's actual
# structure, or the coefficients (weights) in a linear model. So a
# model, in our sense, is distinct from a *predictor*, which is part
# of a superivised learning *machine* (see below).

# 3. A *transformer machine* is an object used for actually
# transforming data. It is fit once (to some possibly different data,
# usually the training data) during instantiation. This fitting wraps
# the provided transformer with the scheme encoding the
# transformation.  

# 4. A *supervised learning machine* is an object eventually used to
# make predictions but is not pre-trained. Rather, during
# instantiation, it pre-transforms (once) the data to be used for
# training *and* testing, into a form suitable for the algorithm
# associated with model supplied (eg, by standardizing or one-hot
# encoding inputs). The hyperparmaters stored in the model may be
# changed after instantiation and retraining the machine retrains
# using the new hyperparameters..

# 5. *Summary*. Tranformers and models just store
# hyperparemters. Machines do the actual transforming and
# learning/predicting. So, a supervised learning machine wraps a model
# in data and, after training, also a predictor; a transformer machine
# wraps a transformer with the scheme encoding the actual
# transformation.


# All Koala types and types created in its associated libraries
# subtype the following:

abstract type BaseType end


## HELPERS AND CONVENIENCE FUNCTIONS

# extension of `show` method for `BaseType` objects:
include("show.jl")

ismissingtype(T::Type) = isa(T, Union) && T.a == Missing

principaltype(T::Union) = T.b::Type # for extacting pure type from missing type

hasmissing(v::AbstractVector) = findfirst(ismissing, v) != nothing

countmissing(v::AbstractVector) = count(ismissing, v)

""" convert a vector of eltype `Union{T, Missing}` to one of eltype `T`"""
function purify(v::AbstractVector)
    T = eltype(v)
    if ismissingtype(T) 
        !hasmissing(v) || @error "Can't purify a vector with missing values."
        return convert(Array{principaltype(T)}, v)
    else
        return v
    end
end
    
# """convenience macro for printing variable values (eg for degugging)"""
# macro dbg(v)
#     esc(quote
#         print((@colon $v), "=", $v, " ")
#     end)
# end

# macro dbg(v1, v2)
#     esc(quote
#         print((@colon $v1), "=", $v1, " ")
#         println((@colon $v2), "=", $v2, " ")
#     end)
# end

# macro dbg(v1, v2, v3)
#     esc(quote
#         print((@colon $v1), "=", $v1, " ")
#         print((@colon $v2), "=", $v2, " ")
#         println((@colon $v3), "=", $v3, " ")
#     end)
# end

""" a version of warn that only warns given a *non-empty* string."""
function softwarn(str)
    if !isempty(str)
        @warn str
    end
end
    
"""Load a well-known public regression dataset with nominal features."""
function load_boston()
    df = CSV.read(joinpath(srcdir, "data", "Boston.csv"),
                  categorical=false, allowmissing=:none)
    features = filter(names(df)) do f
        f != :MedV
    end
    X = df[features]
    for j in 1:size(X, 2)
        X[j] = purify(X[j])
    end
    y = purify(df[:MedV])
    return X, y 
end

"""Load a reduced version of the well-known Ames Housing dataset,
having six numerical and six categorical features."""
function load_ames()
    df = CSV.read(joinpath(srcdir, "data", "reduced_ames.csv"),
                  categorical=false, allowmissing=:none)
    features = filter(names(df)) do f
        f != :target
    end
    X = df[features]
    for j in 1:size(X,2)
        X[j] = purify(X[j])
    end
    y = purify(exp.(df[:target]))
    return X, y 
end
datanow=load_ames

"""Load a well-known public classification dataset with nominal features."""
function load_iris()
    df = CSV.read(joinpath(srcdir, "data", "iris.csv"),
                  categorical=false, allowmissing=:none)
    features = filter(names(df)) do f
        f != :target
    end
    X = df[features]
    for j in 1:size(X, 2)
        X[j] = purify(X[j])
    end
    y = purify(df[:target])
    return X, y 
end

function keys_ordered_by_values(d::Dict{T,S}) where {T, S<:Real}

    items = collect(d) # 1d array containing the (key, value) pairs
    sort!(items, by=pair->pair[2], alg=QuickSort)

    return T[pair[1] for pair in items]

end


"""
    bootstrap_resample_of_mean(v; n=10^6)

Returns a vector of `n` estimates of the of the distribution
generating the samples in `v` from `n` bootstrap resamples of `v`.

"""
function bootstrap_resample_of_mean(v; n=10^6)

    n_samples = length(v)
    mu = mean(v)

    simulated_means = Array{Float64}(undef, n)

    for i in 1:n
        pseudo_sample = sample(v, n_samples, replace=true)
        simulated_means[i]=mean(pseudo_sample)
    end
    return simulated_means
end

"""
    bootstrap_resample_of_median(v; n=10^6)

Returns a vector of `n` estimates of the median of the distribution
generating the samples in `v` from `n` bootstrap resamples of `v`.

"""
function bootstrap_resample_of_median(v; n=10^6)

    n_samples = length(v)
    mu = median(v)

    simulated_medians = Array{Float64}(n)

    for i in 1:n
        pseudo_sample = sample(v, n_samples, replace=true)
        simulated_medians[i]=median(pseudo_sample)
    end
    return simulated_medians
end

# """ 
#     get_meta(df)

# Return very short description of an `AbstractDataFrame`, `df`.

# """
# function get_meta(df::AbstractDataFrame)
#     return DataFrame(feature=names(df),
#                      eltype=eltypes(df),
#                      n_missing=[countmissing(df[j]) for j in 1:size(df, 2)])
# end

""" Return a dictionary of values keyed on the fields of specified
object, suitable for display"""
function params(object::BaseType)
    value_given_parameter  = Dict{Symbol,Any}()
    for name in fieldnames(typeof(object))
        if isdefined(object, name)
            value_given_parameter[name] = getfield(object,name)
        else
            value_given_parameter[name] = "<undefined>"
        end 
    end
    return value_given_parameter
end

# function name(T::Type)
#     if isa(T, Union)
#         types = [getfield(T, f) for f in fieldnames(T)]
#         if isempty(types)
#             return ""
#         else
#             return string("Union{", types[1], types[2:end]..., "}")
#         end
#     else
#         return T.name.name
#     end
# end
            
# """ Extract type parameters of the type of an object."""
# type_parameters(object) = typeof(object).parameters


## ABSTRACT MODEL TYPES

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


## LOSS AND LOW-LEVEL ERROR FUNCTIONS

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

""" Root mean squared percentage loss """
function rmsp(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    count = 0
    for i in rows
        if y[i] != 0.0
            ret += ((y[i] - yhat[i])/y[i])^2
            count += 1
        end
    end
    return sqrt(ret/count)
end

function rmsp(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    count = 0
    for i in eachindex(y)
        if y[i] != 0.0
            ret += ((y[i] - yhat[i])/y[i])^2
            count += 1
        end
    end
    return sqrt(ret/count)
end

function err(rgs::SupervisedModel, predictor, X, y, rows,
             parallel, verbosity, loss::Function=rms)
    return loss(view(y, rows), predict(rgs, predictor, X, rows, parallel, verbosity))
end

function err(rgs::SupervisedModel, predictor, X, y,
             parallel, verbosity, loss::Function=rms)
    return loss(y, predict(rgs, predictor, X, parallel, verbosity))
end

# function auc(truelabel::L) where L
#     _auc(y::AbstractVector{L}, yhat::AbstractVector{T}) where T<:Real = 
#         ROC.AUC(ROC.roc(yhat, y, truelabel))
#     return _auc
# end


# TODO: More classifier loss and metrics 


## MACHINES

abstract type Machine <: BaseType end


## MACHINES FOR TRANSFORMING

struct TransformerMachine{T<:Transformer} <: Machine
    transformer::T
    scheme

    function TransformerMachine{T}(transformer::T, scheme, ::Colon) where T <: Transformer
        return new{T}(transformer, scheme)
    end
end

function TransformerMachine(transformer::T, scheme, ::Colon) where T <: Transformer
    return TransformerMachine{T}(transformer, scheme, :)
end

# usual contructor:
function TransformerMachine(transformer::T, X;
                               parallel::Bool=true,
                               verbosity::Int=1, args...) where T <: Transformer
    scheme = fit(transformer, X, parallel, verbosity; args...)
    return TransformerMachine(transformer, scheme, :)
end

Machine(transformer::Transformer, X; args...) =
    TransformerMachine(transformer, X; args...)


function transform(mach::TransformerMachine, X)
    return transform(mach.transformer, mach.scheme, X)
end

function inverse_transform(mach::TransformerMachine, X; parallel=true, verbosity=1)
    return inverse_transform(mach.transformer, mach.scheme, X)
end


## COMMONLY USED TRANSFORMERS

# a null transformer, useful as default keyword argument
# (has no fit, transform or inverse_transform methods):
struct EmptyTransformer <: Transformer end

# no transformer is empty except the null transformer:
Base.isempty(transformer::Transformer) = false
Base.isempty(transformer::EmptyTransformer) = true

# For (a) remembering the features used in `fit` (calibration), in
# order presented, and selecting only those on tranforming new data
# frames; or (b) selecting only feature labels specified in the
# tranformer:
mutable struct FeatureSelector <: Transformer
    features::Vector{Symbol} # empty means do (a) above
end
FeatureSelector(;features=Symbol[]) = FeatureSelector(features)
function fit(transformer::FeatureSelector, X, parallel, verbosity)
    namesX = names(X)
    issubset(Set(transformer.features), Set(namesX)) ||
        @error "Attempting to select non-existent feature(s)."
    if isempty(transformer.features)
        return namesX
    else
        return transformer.features
    end
end
function transform(transformer::FeatureSelector, features, X)
    issubset(Set(features), Set(names(X))) ||
        @error "Supplied frame does not admit previously selected features."
    return X[features]
end 

# identity transformations:
struct IdentityTransformer <: Transformer end
fit(transformer::IdentityTransformer, y, parallel, verbosity) = nothing
transform(transformer::IdentityTransformer, scheme, y) = y
inverse_transform(transformer::IdentityTransformer, scheme, y) = y

"""
    struct RowsTransformer <: Transformer

A special transformer mapping `AbstractVector{Int}`s to
`AbstractVector{Int}`s, to deal with dropping rows from an input
`DataFrame`.

Suppose `df` is a `DataFrame` and `bad` is a vector of indices for
rows that have been dropped from `df` to form `df_safe`. Given a
vector of indices `rows`, I want `df[good_rows,:]` where `good_rows`
is `rows` with any elements of `bad` removed. However, I only have
access to `df_safe` (although I know `size(df, 1)`).  An appropriate
transformation of `rows` gives a vector of indices to use on
`df_safe`:

    mach = Machine(RowTransformer(size(df,1)), bad)
    df[good_rows,:] == df_safe[transform(mach, rows),:] # true

Analogous statements hold for array-like collections apart from
`DataFrames`.

"""
struct RowsTransformer <: Transformer
    original_num_rows::Int
end

function fit(transformer::RowsTransformer, missing_indices, parallel, verbosity)
    scheme = Array{Union{Int,Missing}}(undef, transformer.original_num_rows)
    counter = 1
    for i in 1:transformer.original_num_rows
        if i in missing_indices
            scheme[i] = missing
        else
            scheme[i] = counter
            counter += 1
        end
    end
    return scheme
end

transform(transformer::RowsTransformer, scheme, rows) =
    collect(skipmissing(view(scheme, rows)))


## MACHINES FOR SUPERVISED LEARNING

""" 
## `mutable struct SupervisedMachine`

### Keyword options

    drop_unseen=false

All calls to `fit!`, `err`, `cv` and `learning_curve` on the machine
will ignore rows with categorical features taking on values not seen
in training. Note: The `predict` method cannot be safely called on
data containing such rows.

    features = Symbol[]

Only specified features are used. Empty means *all* features are
used. The features used can also be controlled by `transformer_X`.

"""
mutable struct SupervisedMachine{P, M <: SupervisedModel{P}} <: Machine

    model::M
    transformer_X::Transformer
    transformer_y::Transformer
    scheme_X
    scheme_y
    rows_with_unseen::AbstractVector{Int}
    rows_transformer_machine::TransformerMachine{RowsTransformer}
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
        features=Symbol[],
        transformer_X=EmptyTransformer(),
        transformer_y=EmptyTransformer(),
        drop_unseen::Bool=false, 
        verbosity::Int=1) where {P, M <: SupervisedModel{P}}

        # check dimension match:
        size(X,1) == length(y) || throw(DimensionMismatch())

        # check valid `features`; if none provided, take to be all
        if isempty(features)
            features = names(X)
        end
        allunique(features) || @error "Duplicate features."
        issubset(Set(features), Set(names(X))) || @error "Invalid feature vector."

        # bind `X` to view of `X` containing only the features
        # specified, in the order specified (does not generate copy):
        X = X[features]        

        # check for missing data and report eltypes:
        if verbosity > 0
            kind_given_feature = Dict{Symbol,String}()
            @info "Element types of input features before transformation:"
        end
        for field in names(X)
            T = eltype(X[field])
            real_categorical = (T <: AbstractFloat ? "continuous" : "categorical")
            verbosity > 0 ? kind_given_feature[field] = real_categorical : nothing
            if ismissingtype(T) && verbosity > -1
                    @warn ":$field has a missing-element type. Treated as categorical."
            end
        end
        if verbosity > 0
            show(stdout, MIME("text/plain"), kind_given_feature)
            println()
        end

        # report size of data used for transformations
        percent_train = round(Int, 1000*length(train_rows)/length(y))/10
        verbosity < 0 || @info "$percent_train% of data used to compute "*
                              "transformation parameters."

        # assign transformers if not provided:
        if isempty(transformer_X)
            transformer_X = default_transformer_X(model)
        end
        if isempty(transformer_y)
            transformer_y = default_transformer_y(model)
        end            

        # if necessary, determine rows with categorical levels not
        # seen in X[train_rows, :] and build corresponding rows
        # transformer:
        if drop_unseen

            # get test_rows:
            test_rows = filter(eachindex(y)) do i
                !(i in train_rows)
            end

            seen, unseen = split_seen_unseen(X, train_rows, test_rows)
            if verbosity > -1
                if length(unseen) == 0 
                    @warn "All remaining rows of the inputs data have categorical "*
                         "features taking values not seen during data transformation."
                else
                    bad_percentage = round(Int, 1000*length(unseen)/length(test_rows))/10
                    @warn "$bad_percentage% of the remaining rows of input data "*
                         "(recorded in the attribute `rows_with_unseen`) "*
                         "contain "*
                         "patterns for which some categorical feature "*
                         "takes on values not seen during data "*
                         "transformation. These will be ignored in "*
                         "calls to err(), cv() and  learning_curve(). "*
                         "However, you will be unable to safely "*
                         "call predict() on such data."
                end
            end
        else
            unseen = Int[]
        end

        rows_transformer_machine = Machine(RowsTransformer(length(y)), unseen)
        
        mach = new{P, M}(model::M)
        mach.transformer_X = transformer_X
        mach.transformer_y = transformer_y
        mach.scheme_X = fit(mach.transformer_X, view(X, train_rows),
                            true, verbosity - 1)
        mach.scheme_y = fit(mach.transformer_y, y[train_rows], verbosity - 1, 1)
        mach.rows_with_unseen = unseen
        mach.rows_transformer_machine = rows_transformer_machine

        # transform the data, dumping any untransformable data if
        # drop_unseen = true, while throwing an exception otherwise:
        if isempty(unseen)
            try
                mach.Xt = transform(mach.transformer_X, mach.scheme_X, X)
                mach.yt = transform(mach.transformer_y, mach.scheme_y, y)
            catch exception
                if isa(exception, KeyError)
                    @error "KeyError: key $(exception.key) not found. Problably "*
                        "a categorical feature takes on values "*
                        "not encountered in the rows provided for "*
                        "computing data transformations. Try calling "*
                        "machine constructor with `drop_unseen=true`."
                else
                    throw(exception)
                end
            end
        else
            all_good_rows = filter(eachindex(y)) do i
                !(i in unseen)
            end
            mach.Xt = transform(mach.transformer_X, mach.scheme_X, view(X, all_good_rows))
            mach.yt = transform(mach.transformer_y, mach.scheme_y, view(y, all_good_rows))
        end
        
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

function fit!(mach::SupervisedMachine, rows;
              add=false, verbosity=1, parallel=true, args...)
    verbosity < 0 || softwarn(clean!(mach.model))

    # transform `rows` to account for rows that may have been dropped
    # during transformation:
    if !isempty(mach.rows_with_unseen)
        rows = transform(mach.rows_transformer_machine, rows)
    end
    
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
        @error "You must specify training rows in the first call to fit!\n"*
              "E.g., fit!(mach, train_rows)."
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
    mach.n_iter > 0 || @error string(mach, " has not been fitted.")
    Xt = transform(mach.transformer_X, mach.scheme_X, view(X, rows))
    yt = predict(mach.model, mach.predictor, Xt, parallel, verbosity)
    return inverse_transform(mach.transformer_y, mach.scheme_y, yt)
end

function predict(mach::SupervisedMachine, X; parallel=true, verbosity=1)
    mach.n_iter > 0 || @error string(mach, " has not been fitted.")
    Xt = transform(mach.transformer_X, mach.scheme_X, X)
    yt = predict(mach.model, mach.predictor, Xt, parallel, verbosity)
    return inverse_transform(mach.transformer_y, mach.scheme_y, yt)
end

function err(mach::SupervisedMachine, test_rows;
             loss=rms, parallel=false, verbosity=1, raw=false)

    mach.n_iter > 0 || @error "Attempting to predict using untrained machine."

    !raw || verbosity < 0 || @warn "Reporting errors for *transformed* target. "*
                                  "Use `raw=false` to report true errors."

    # transform `test_rows` to account for rows that may have been
    # dropped during transformation:
    if !isempty(mach.rows_with_unseen)
        verbosity < 1 ||
            @info "Ignoring rows with categorical data values "*
                 "not seen in transformation."
        test_rows = transform(mach.rows_transformer_machine, test_rows)
    end

    # transformed version of target predictions:
    raw_predictions = predict(mach.model, mach.predictor, mach.Xt, test_rows,
                            parallel, verbosity) 

    if raw # return error on *transformed* target, which is faster
        return loss(view(mach.yt, test_rows), raw_predictions)
    else  # return error for untransformed target
        return loss(inverse_transform(mach.transformer_y, mach.scheme_y,
                                      view(mach.yt, test_rows)),
                    inverse_transform(mach.transformer_y, mach.scheme_y,
                                      raw_predictions))
    end
end


## `SupervisedModel`  fall-back methods

# By default all supervised model types can be ensembled. 
supports_ensembling(model::SupervisedModel) = true

# default default_transformers:
default_transformer_X(model::SupervisedModel) = FeatureSelector()
default_transformer_y(model::SupervisedModel) = IdentityTransformer()

# for enforcing model parameter invariants:
clean!(model::SupervisedModel) = "" # no checks, emtpy message

# To allow for an extra `rows` argument in `setup` and `predict`:

# fallbacks:
setup(model::SupervisedModel, Xt, yt, rows, scheme_X, parallel, verbosity) =
    setup(model, Xt[rows,:], view(yt, rows), scheme_X, parallel, verbosity)
predict(model::SupervisedModel, predictor, Xt, rows, parallel, verbosity) =
    predict(model, predictor, Xt[rows,:], parallel, verbosity)


## CONSTANT-PREDICTING REGRESSOR (A `REGRESSOR` EXAMPLE)

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

    
## VALIDATION TOOLS

"""
## partition(rows::AbstractVector{Int}, fractions...)

Then splits the vector `rows` into a tuple of `Vector{Int}` objects
whose lengths are given by the corresponding `fractions` of
`length(rows)`. The last fraction is not provided, as it
is inferred from the preceding ones. So, for example,

    julia> partition(1:1000, 0.2, 0.7)
    (1:200, 201:900, 901:1000)

"""
function partition(rows::AbstractVector{Int}, fractions...)
    rows = collect(rows)
    rowss = []
    if sum(fractions) >= 1
        throw(DomainError)
    end
    n_patterns = length(rows)
    first = 1
    for p in fractions
        n = round(Int, p*n_patterns)
        n == 0 ? (@warn "A split has only one element"; n = 1) : nothing
        push!(rowss, rows[first:(first + n - 1)])
        first = first + n
    end
    if first > n_patterns
        @warn "Last vector in the split has only one element."
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

    isdefined(mach.model, :n) || @error "$(mach.model) does not support iteration."

    # save to be reset at end:
    old_n = mach.model.n
    
    !raw || verbosity < 0 ||
        @warn "Reporting errors for *transformed* target. Use `raw=false` "*
             " to report true errors." 

    # transform `train_rows` and `test_rows` to account for rows that
    # may have been dropped during transformation:
    if !isempty(mach.rows_with_unseen)
        verbosity < 1 ||
            @info "Ignoring rows with categorical data values "*
                 "not seen in transformation."
        train_rows = transform(mach.rows_transformer_machine, train_rows)
    end

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

    report = Dict{Symbol,Any}()
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
        push!(errors, err(mach, test_rows, raw=raw, loss=loss, verbosity=verbosity - 1))
        filter!(range) do x
            x > mach.n_iter
        end
    end

    verbosity < 1 || println("\nLearning curve added to machine report.")
    
    report[:learning_curve] = (n_iter_list, errors)
    merge!(mach.report, report)
    
    mach.model.n = old_n
    
    return n_iter_list, errors

end

""" 
    cv(mach::SupervisedMachine, rows; 
       n_folds=9, loss=rms, parallel=true, verbosity=1, raw=false, randomize=false)

Return a list of cross-validation losses for patterns with row indices
in `rows`, an iterator that is initially randomized when an optional
parameter `randomize` is set to `true`.

"""
function cv(mach::SupervisedMachine, rows; n_folds=9, loss=rms,
             parallel=true, verbosity=1, raw=false, randomize=false)

    !raw || verbosity < 0 ||
        @warn "Reporting errors for *transformed* target. Use `raw=false` "*
             " to report true errors."

    # transform `rows` to account for rows that may have been dropped
    # during transformation:
    if !isempty(mach.rows_with_unseen)
        verbosity < 1 ||
            @info "Ignoring rows with categorical data values "*
                 "not seen in transformation."
        rows = transform(mach.rows_transformer_machine, rows)
    end

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
        return err(mach, test_rows; loss=loss,
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
        return @distributed vcat for n in 1:n_folds
            Float64[get_error(firsts[n], seconds[n])]
        end
    else
        errors = Array{Float64}(undef, n_folds)
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

    julia> u1, u2, A = @curve x range(0, stop=1, length=100) α [1,2,3] x^α
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
            flush(stdout)
            # print(i,"\r"); flush(stdout) 
            push!(output, $(esc(code)))
        end
        collect($(esc(range))), [x for x in output]
    end
end

macro curve(var1, range1, var2, range2, code)
    quote
        output = Array{Float64}(undef, length($(esc(range1))), length($(esc(range2))))
        for i1 in eachindex($(esc(range1)))
            $(esc(var1)) = $(esc(range1))[i1]
            for i2 in eachindex($(esc(range2)))
                $(esc(var2)) = $(esc(range2))[i2]
                # @dbg $(esc(var1)) $(esc(var2))
                print((@colon $(esc(var1))), "=", $(esc(var1)), " ")
                print((@colon $(esc(var2))), "=", $(esc(var2)), "                    \r")
                flush(stdout)
                output[i1,i2] = $(esc(code))
            end
        end
        collect($(esc(range1))), collect($(esc(range2)))', output
    end
end

macro pcurve(var1, range, code)
    quote
        N = length($(esc(range)))
        pairs = @distributed vcat for i in eachindex($(esc(range)))
            $(esc(var1)) = $(esc(range))[i]
            print((@colon $(esc(var1))), "=", $(esc(var1)), "                    \r")
            flush(stdout)
            print(i,"\r"); flush(stdout) 
            [( $(esc(range))[i], $(esc(code)) )]
        end
        sort!(pairs, by=first)
        collect(map(first,pairs)), collect(map(last, pairs))
    end
end

"""
## `function compete(e0, e1; alpha=0.05)`

Given paired samples `e0` and `e1`, we test the
null-hypothesis that the underlying distributions have the same mean,
using the significance level `alpha`. Normality of the underlying
distributions is assumed and a two-sided t-test applied. 

### Return value

`(c, p)` where `p` is the p-value for the test and `c` is the competition outcome:

- '0' if the null is rejected and `e0` has the smaller mean (M1 "wins")
- '1' if the null is rejected and `e1` has the smaller mean (M0 "wins")
- 'D' if the null is accepted ("draw" because `p > alpha`)

"""
function compete(e0, e1; alpha=0.05)
    t = OneSampleTTest(e1-e0)
    p = pvalue(t)
    if p > alpha
        return 'D', p
    end
    if mean(e0) < mean(e1)
        return '0', p
    end
    return '1', p
end


""" 
## `function split_seen_unseen`

    split_seen_unseen(v::Vector, train_indices, test_indices)`

Detects on which indices in `test_indices` the `AbstractVector`, `v`,
takes on values not occuring in the `train_indices` part of the
vector. Returns a pair of integer vectors `(S, U)` where `S` are the
seen indices and `U` the unseen ones.

    split_seen_unseen(df::AbstractDataFrame, train_rows, test_rows)

Detects the rows in `test_rows` for which any of the categorical
features of `df` takes on a value not observed in
`train_rows`. Returns a pair of integer vectors `(S, U)` where `S` are
the unseen rows and `U` the unseen ones.

"""
function split_seen_unseen(v::AbstractVector, train_rows, test_rows)
    train_values = Set(v[train_rows])
    test_values  = Set(v[test_rows])
    unseen_values = setdiff(test_values, train_values)

    is_unseen = map(test_rows) do i
        v[i] in unseen_values
    end

    return test_rows[.!is_unseen], test_rows[is_unseen]

end 
    
function split_seen_unseen(df::AbstractDataFrame, train_rows, test_rows)

    # collect all rows with unseen categorical values together:
    unseen_rows = Int[]
    for j in 1:size(df, 2)
        if !(eltype(df[j]) <: AbstractFloat)
            seen, unseen = split_seen_unseen(df[j], train_rows, test_rows)
            append!(unseen_rows, unseen)
        end
    end
    unseen_rows = unique(unseen_rows)

    ordered_unseen = Int[]
    ordered_seen = Int[]

    for i in test_rows
        if i in unseen_rows
            push!(ordered_unseen, i)
        else
            push!(ordered_seen, i)
        end
    end

    return ordered_seen, ordered_unseen

end


## RECIPES FOR USE WITH Plots.jl

"""
    PlotableDict(d)

Returns a wrapped version of the dictionary `d` (assumed to have
`Real` values) to enable plotting with calls to `Plot.plot`, etc
(provided by a `Plots` recipe). In particular,

    plot(PlotableDict(d), ordered_by_keys=true)

plots the bars in the lexographic order of the keys. Otherwise, the
bars are ordered by the values corresponding to to each key.

"""
struct PlotableDict{KeyType,ValueType<:Real}
    dict::Dict{KeyType,ValueType}
end

@recipe function dummy(::Type{PlotableDict{KeyType,ValueType}},
                       pd::PlotableDict{KeyType,ValueType};
                       ordered_by_keys=false) where {KeyType,ValueType<:Real}

    seriestype := :bar

    dict_to_be_plotted = pd.dict
    x = String[]
    y = ValueType[]
    
    if ordered_by_keys
        kys = sort(collect(keys(dict_to_be_plotted)))
    else
        kys = keys_ordered_by_values(dict_to_be_plotted)
    end
    
    for k in kys
        push!(x, string(k))
        push!(y, dict_to_be_plotted[k])
    end
    offset = 0.05*maximum(abs.(y))
    xticks := Float64[]
    annotations := [(Float64(i), offset, x[i]) for i in eachindex(x)]
    y

end
            
mutable struct BootstrapHistogramOfMean 
        args
end 

"""
    bootstrap_histogram_of_mean(v; n=1e6, plotting_kws...)

Create a bootstrap sample of size `n` from `v` and plot the
corresponding histogram.

"""
bootstrap_histogram_of_mean(args...; kw...) = begin  
            RecipesBase.plot(BootstrapHistogramOfMean(args); kw...)
end

bootstrap_histogram_of_mean!(args...; kw...) = begin  
    RecipesBase.plot!(BootstrapHistogramOfMean(args); kw...)
end

@recipe function dummy(h::BootstrapHistogramOfMean; n=10^6)
    length(h.args) == 1 || typeof(h.args) <: AbstractVector ||
        @error "A BootstrapHistogramOfMean should be given one vector. "*
    "Got: $(typeof(h.args))"
    v = h.args[1]
    bootstrap = bootstrap_resample_of_mean(v; n=n)
    @series begin
        seriestype := :histogram
        alpha --> 0.5
        normalized := true
        bins --> 50
        bootstrap
    end
    if isdefined(:StatPlots) 
        @series begin
            seriestype := :density
            label := ""
            linewidth --> 2
            color --> :black
            bootstrap
        end
    else
        @info "For denisty approximation in a bootstrap_histogram_of_mean, "*
        "import StatPlots."
    end
end

mutable struct BootstrapHistogramOfMedian 
        args
end 

"""
    bootstrap_histogram_of_median(v; n=1e6, plotting_kws...)

Create a bootstrap sample of size `n` from `v` and plot the
corresponding histogram.

"""
bootstrap_histogram_of_median(args...; kw...) = begin  
            RecipesBase.plot(BootstrapHistogramOfMean(args); kw...)
end

bootstrap_histogram_of_median!(args...; kw...) = begin  
    RecipesBase.plot!(BootstrapHistogramOfMean(args); kw...)
end

@recipe function dummy(h::BootstrapHistogramOfMedian; n=10^6)
    length(h.args) == 1 || typeof(h.args) <: AbstractVector ||
        @error "A BootstrapHistogramOfMedian should be given one vector. "*
    "Got: $(typeof(h.args))"
    v = h.args[1]
    bootstrap = bootstrap_resample_of_median(v; n=n)
    @series begin
        seriestype := :histogram
        alpha --> 0.5
        normalized := true
        bins --> 50
        bootstrap
    end
    if isdefined(:StatPlots) 
        @series begin
            seriestype := :density
            label := ""
            linewidth --> 2
            color --> :black
            bootstrap
        end
    else
        @info "For denisty approximation in a bootstrap_histogram_of_median, "*
        "import StatPlots."
    end
end



end # module
