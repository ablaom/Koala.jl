using Koala
using Base.Test

# helpers:
dict = Dict{String,Int}()
dict["c"] = 3
dict["b"] = 2
dict["a"] = 1
@test keys_ordered_by_values(dict) == ["a", "b", "c"]
bootstrap_resample_of_mean(randn(100))

X, y = load_boston();
train, test = split(eachindex(y), 0.8); # 80:20 split

transformer = Koala.FeatureSelector(features=[:Indus, :Chas])
transformerM = Machine(transformer, X)
@test transform(transformerM, X) == X[[:Indus, :Chas]]

rgs = ConstantRegressor()
mach = Machine(rgs, X, y, train, transformer_X=transformer)
showall(mach)
fit!(mach, train)
score = err(mach, test)
println("score = $score")
fit!(mach) # fit again without recomputing cache
@test 9.5 < score && 9.6 > score

@test score == err(mach, test, raw=true)
@test predict(mach, X, test) ==
    predict(mach.model, mach.predictor, mach.Xt[test,:], false, false)

learning_curve(mach, train, test, [2, 4, 8, 1000], raw=false)
learning_curve(mach, train, test, [2000, 3000], raw=false, restart=false)
cv(mach, vcat(test, train))

u,v = @curve r linspace(0,10,50) (r^2 + 1)
u,v = @pcurve r linspace(0,10,50) (r^2 + 1)
u,v,w =@curve r linspace(0,10,5) s linspace(0,5,4) r*s^2




