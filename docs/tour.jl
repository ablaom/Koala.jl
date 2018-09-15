## DEMONSTRATION OF BASIC USAGE

# Load the basic Koala toolset:
using Koala
using Statistics

# Load a built-in supervised learning data set:
X, y = load_ames();

# Note that `X`, which stores the input patterns, is a `DataFrame`
# object; the high-level Koala interface expects all inputs to be
# `DataFrame` objects. Columns of element type `T <: AbstractFloat`
# are treated as numerical features; all others are treated as
# categorical.

# In particular, note that a column `col` of element type
# `Union{Missings, T}` - will be treated as categorical. In that case
# replace any missing values in `col` and substitue `purify(col)` for
# `col` to get the eltype `T` you probably wanted. We can check for
# missing-type columns as follows:
[ismissingtype(eltype(X[j])) for j in 1:size(X, 2)]

# Or get a summary of eltype information:
describe(X)

# In Koala, normal practice is to store training and test data in a
# single `DataFrame`, `X`, padding unknown instances of the target
# `y`, and to construct integer vectors pointing to the rows of the
# training, validation and test parts of the data:
train, valid, test = partition(eachindex(y), 0.7, 0.1); # that's 70%, 10%, 20%.

# Choose a model for learning:
using KoalaTrees
tree = TreeRegressor()

# Here `tree` is just a `mutable struct` storing the parameters
# specifying how we should train a single tree regressor:

# We now construct a "machine" from the model, ie, wrap the model in
# transformed versions of the data (plus other stuff):
treeM = Machine(tree, X, y, train);

# If you are curious you can access the transformed data through the
# fields `Xt` and `yt`. By default, `TreeRegressor` target variables
# are not not actually transformed:
treeM.yt == y

# However, the `TreeRegressor` training algorithm expects inputs in the
# form of a custom type called a `DataTableau`:
treeM.Xt

# In such an object, all categorical values are relaballed as integers
# and converted to Floating type:
treeM.Xt.raw[1:6,:]

# Note that `Xt` and `yt represents *all* the transformed data. The
# `train` argument of `Machine` merely specifies which part of the
# data should be used to calibrate the transformations which, to
# safeguard against data leakage, should not include validation or
# test data. If we want to change the way we would like data to be
# transformed, then we must build a new machine:
ty = treeM.transformer_y
ty.standardize = true;
treeM = Machine(tree, X, y, train, transformer_y=ty, verbosity=2);
y[1:6]
treeM.yt[1:6,:]

# Now yt[train] will have zero mean and unit variance, but notice
# that the full `yt` vector has slightly different statistics:
mean(treeM.yt), std(treeM.yt)

# Now we train the machine, specifying the training rows:
fit!(treeM, train)

# And report the error on the validation rows:
err(treeM, valid)

# Note that target predictions are automatically *inverse transformed*
# and compared with the untransformed validation data to report the
# error we actually want.

# What can inspect `Koala` objects up to any specified depth:
show(treeM, 3)

# Statistics gathered during training of any machine are stored in the
# `report` field:
keys(treeM.report) |> collect
treeM.report[:feature_importance_plot]

# If we want to retrain on the *same* data, with a
# modified model parameter, then their is no need to respecify the
# rows:
tree.regularization = 0.5
fit!(treeM);
err(treeM, valid)

# To retrain using *different* data:
fit!(treeM, vcat(train, valid));
err(treeM, test)

# Let's tune the regularization parameter. We will use Koala's
# `@curve` macro:
r_vals, errs = @curve r range(0, stop=0.99, length=101) begin
    tree.regularization = r
    fit!(treeM, train)
    err(treeM, valid)
end;

# Plotting the results:
using UnicodePlots
lineplot(r_vals, errs)

# The optimal regularization is given by
r = r_vals[argmin(errs)]

# Here's how to obtain cross-validation errors (parallelized by default):
fulltrain = vcat(train, valid);
cv(treeM, fulltrain)

# If computation is slow, it occassionally help to report errors for
# the *transformed* target instead:
cv(treeM, fulltrain, raw=true, n_folds=6)

# We'll now fine-tune using cross-validation:
r_vals, errs = @curve r range(0.8, stop=0.99, length=51) begin
    tree.regularization = r
    mean(cv(treeM, fulltrain, verbosity=0))
end;

# And report final estimate for the error:
tree.regularization = r_vals[argmin(errs)]
errs = cv(treeM, fulltrain, n_folds=12)
println("95% confidence interval for error = ", mean(errs), " ± ", 2*std(errs))

# And the test error is:
err(treeM, test)


## ENSEMBLE MODELS

# Build an ensemble of `RegressorTree` models, :
using KoalaEnsembles
forest = EnsembleRegressor(atom=tree)

# If we tweak `tree` parameters, these are reflected in the forest:
tree.regularization = 0
tree.max_features = 4
show(forest, 2)


# Let's wrap the model in appropriately transformed data:
forestM = Machine(forest, X, y, train)

# A model with `n` as a parameter indicates it is iterative in some
# sense. Here `forest.n` is the number of models in the
# ensemble. Koala's built in function `learning_curve` can be used to
# determine the size of `n`. 
n_vals, errs = learning_curve(forestM, train, valid, 20:10:400)
lineplot(n_vals, errs)

# So the following is reasonable:
forest.n = 250

# Commonly in ensemble methods, predictions are the means of the
# predictions of each regressor in the ensemble. Here predictions are
# *weighted* sums and the weights are optimized to minimize the RMS
# training error. Since this sometimes leads to "under-regularized"
# models the training error is further penalized with a term measuring
# the deviation of the weights from uniformity. Set the parameter
# `forest.weight_regularization=1` (the default and maximum permitted
# value) and weights are completely uniform. Set
# `forest.weight_regularization=0` and the training error penalty is
# dropped altogether.

fit!(forestM, train)
forestM.report[:normalized_weights] 

# To refit the weights with a new regularization penalty, but without
# changing the ensemble itself, use ``fit_weights``:

forest.weight_regularization = 0.5
fit_weights!(forestM)
forestM.report[:normalized_weights] 

# Tuning the parameter `forest.weight_regularization` may be done
# with the help of the `weight_regularization_curve` function:
reg_vals, errs = weight_regularization_curve(forestM, test; range = linspace(0,1,51));
lineplot(reg_vals, errs)

# In this case maximum regulariation is indicated:
reg_vals[indmin(errs)]










