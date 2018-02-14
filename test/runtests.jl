using Koala
using Base.Test

# helpers:
dict = Dict{String,Int}()
dict["c"] = 3
dict["b"] = 2
dict["a"] = 1
@test keys_ordered_by_values(dict) == ["a", "b", "c"]
bootstrap_resample_of_mean(randn(1000))

X, y = load_boston();
nrows = length(y)
train, test = splitrows(1:nrows, 0.8); # 80:20 split
rgs = ConstantRegressor()
features=[]
mach = SupervisedMachine(rgs, X, y, train, features=features)
showall(mach)
fit!(mach, train)
score = err(mach, test)
println("score = $score")
@test 9.5 < score && 9.6 > score

# next two tests must work because transform for ConstantRegressor()
# is the identity.
@test score == err(mach, test, raw=true)
@test predict(mach, X, test) ==
    predict(mach.model, mach.predictor, mach.Xt[test,:], false, false)

learning_curve(mach, test, [2, 4, 8, 1000])



