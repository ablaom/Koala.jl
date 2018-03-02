## Demonstration of basic usage

# Load the basic Koala toolset:
using Koala

# Load a built-in supervised learning data set:
X, y = load_ames();

# Note that `X`, which stores the input patterns, is a `DataFrame`
# object; the high-level Koala interface expects all inputs to be
# `DataFrame` objects. Columns of type `T <: Real` are treated as
# numerical features; all others are treated as categorical.

# In Koala, normal practice is to store training and test data in a
# single `DataFrame`, `X`, padding unknown instances of the target
# `y`, and to construct integer vectors pointing to the rows of the
# training, validation and test parts of the data:
train, valid, test = split(eachindex(y), 0.7, 0.1); # that's 70%, 10%, 20%.

# Choose a model for learning:
using KoalaTrees
tree = TreeRegressor()

# Here `tree` is just a `mutable struct` storing the parameters
# specifying how we should train a Light Gradient Boosting Machine:
showall(tree)

# Construct a "machine" from the model, which wraps the model in
# transformed versions of the data:
treeM = Machine(tree, X, y, train)
treeM.Xt[1:6,:]
treeM.yt[1:6]

# In this case the categorical features have been one-hot encoded; the
# target eltype has been converted to `Float64` but is otherwise
# untouched.

# Note that `Xt` and `yt represents *all* the transformed data. The
# `train` argument of `Machine` merely specifies which part of the
# data should be used to calibrate the transformations which, to
# safeguard against data leakage, should not include validation or
# test data. If we want to change the way we would like data to be
# transformed, then we must build a new machine:
ty = treeM.transformer_y
showall(ty)
ty.standardize = true
treeM = Machine(tree, X, y, train, transformer_y=ty, verbosity=2)
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

# If we call `showall` on our machine, we see a summary of its
# internals and also a ranking of feature importance, according to the
# training we just performed.
showall(treeM)

# If we want to retrain on the *same* data, with a modified model
# parameter, then their is no need to respecify the rows:
tree.regularization = 0.5
fit!(treeM)
err(treeM, valid)

# To retrain using *different* data:
fit!(treeM, vcat(train, valid))
err(treeM, test)

# Let's tune the regularization parameter. We will use Koala's
# `@curve` macro:
r_vals, errs = @curve r linspace(0,0.99,101) begin
    tree.regularization = r
    fit!(treeM, train)
    err(treeM, valid)
end;

# Here `linspace(0,0.99,101)` specifies the range of values for the
# variable `r`. Plotting the results:
using UnicodePlots
lineplot(r_vals, errs)

# The optimal regularization is given by
r = r_vals[indmin(errs)]

# Here's how to obtain cross-validation errors (parallelized by default):
fulltrain = vcat(train, valid);
cv(treeM, fulltrain)

# If computation is slow, it occassionally help to report errors for
# the *transformed* target instead:
cv(treeM, fulltrain, raw=true, n_folds=6)

# We'll now fine-tune using cross-validation:
r_vals, errs = @curve r linspace(0.8,0.99,51) begin
    tree.regularization = r
    mean(cv(treeM, fulltrain, verbosity=0))
end;

# And report final estimate for the error:
tree.regularization = r_vals[indmin(errs)]
errs = cv(treeM, fulltrain, n_folds=12)
println("95% confidence interval for error = ", mean(errs), " ± ", 2*std(errs))

# And the test error is:
err(treeM, test)


## Ensemble models

# Build an ensemble of `RegressorTree` models, :
using KoalaEnsembles
forest = EnsembleRegressor(atom=tree)
showall(forest)

# If we tweak `tree` parameters, these are reflected in the forest:
tree.regularization = 0
tree.max_features = 4
showall(forest.atom)

# Let's wrap the model in appropriately transformed data:
forestM = Machine(forest, X, y, train)

# A model with `n` as a parameter indicates it is iterative in some
# sense. Here `forest.n` is the number of models in the
# ensemble. Koala's built in function `learning_curve` can be used to
# determine the size of `n`. 
n_vals, errs = learning_curve(forestM, train, valid, 20:10:400)
lineplot(n_vals, errs)

# So the following is reasonable:
forest.n = 300

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
forestM.report[:normalized_weights] |> showall

# To refit the weights with a new regularization penalty, but without
# changing the ensemble itself, use ``fit_weights``:

forest.weight_regularization = 0.5
fit_weights!(forestM)
forestM.report[:normalized_weights] |> showall

# Tuning the parameter `forest.weight_regularization` may be done
# with the help of the `weight_regularization_curve` function:

reg_vals, errs = weight_regularization_curve(forestM, test; range = linspace(0,1,51));
lineplot(reg_vals, errs)

# In this case maximum regulariation is indicated:
reg_vals[indmin(errs)]










