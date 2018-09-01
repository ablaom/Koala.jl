using Test
using Koala
using DataFrames

# loss functions:
y = [1, 2, 3, 4]
yhat = y .+ 1
@test isapprox(rms(y, yhat), 1.0)
@test isapprox(rmsl(y, yhat),
               sqrt((log(1/2)^2 + log(2/3)^2 + log(3/4)^2 + log(4/5)^2)/4))
@test isapprox(rmslp1(y, yhat),
               sqrt((log(2/3)^2 + log(3/4)^2 + log(4/5)^2 + log(5/6)^2)/4))
@test isapprox(rmsp(y, yhat), sqrt((1 + 1/4 + 1/9 + 1/16)/4))

# helpers:
dict = Dict{String,Int}()
dict["c"] = 3
dict["b"] = 2
dict["a"] = 1
@test keys_ordered_by_values(dict) == ["a", "b", "c"]
bootstrap_resample_of_mean(randn(100))
v = [1,2,5,4,missing,7,missing]
@test hasmissing(v) 
@test ismissingtype(eltype(v))
@test Koala.leadingtype(eltype(v)) == Int
@test countmissing(v) == 2

v[5] = 42; v[7] = 42
@test eltype(purify(v)) == Int

X, y = load_boston();
train, test = split(eachindex(y), 0.8); # 80:20 split
describe(X)
get_meta(X)

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
learning_curve(mach, train, test, [2000, 3000], restart=false)
cv(mach, vcat(test, train))

u,v = @curve r linspace(0,10,50) (r^2 + 1)
u,v = @pcurve r linspace(0,10,50) (r^2 + 1)
u,v,w =@curve r linspace(0,10,5) s linspace(0,5,4) r*s^2


# test split_seen_unseen
v = ['a', 'b', 'b', 'c',
     'a', 'd', 'a', 'b', 'e']
trainrows = 1:4
testrows = 5:9
@test split_seen_unseen(v, trainrows, testrows) == ([5, 7, 8], [6, 9])

w = ["log", "house", "house", "house",
     "brick", "house", "log", "log", "log"]
df = DataFrame(v=v, w=w)
@test split_seen_unseen(df, trainrows, testrows) == ([7, 8], [5, 6, 9])

# time tests:
v=rand(UInt32, 10^7)
trainrows, testrows = split(eachindex(v), 0.9)
@time map(length, split_seen_unseen(v, trainrows, testrows))

# test drop_unseen capability
X[4] = map(Char, X[4])
X[9] = map(Char, X[9])
X[10] = map(Char, X[10])
train = 1:length(y) - 6
test = length(y) - 5 : length(y)
model = ConstantRegressor()
mach = Machine(model, X, y, train, drop_unseen=true)
fit!(mach, train)
err(mach, test)
learning_curve(mach, train, test, [2, 4, 8, 1000])
learning_curve(mach, train, test, [2000, 3000], restart=false)


t = Koala.RowsTransformer(10)
tM  = Machine(t, [2, 4, 5, 9])
@test transform(tM, 3:7) == [2, 3, 4]
df = DataFrame(letters=['a','b','c','d','e','f','g','h','i','j'])
df_shrunk = df[[1, 3, 6, 7, 8], :]
@test df_shrunk[transform(tM, 3:7), :letters] == ['c', 'f', 'g']

rgs = ConstantRegressor()
train, test = split(eachindex(y), 0.8); # 80:20 split
mach = Machine(rgs, X, y, train, drop_unseen=true)
showall(mach)
fit!(mach, train)
score = err(mach, test)
unseen = mach.rows_with_unseen
seen = filter(test) do i
    !(i in unseen)
end
@test mean(y[train]) ≈ mach.predictor
yhat = mean(y[train])*ones(length(y))
@test rms(yhat, y, seen) ≈ score
println("score = $score")
fit!(mach) # fit again without recomputing cache
@test 9 < score && 10 > score
@test score == err(mach, test, raw=true)
learning_curve(mach, train, test, [2, 4, 8, 1000], raw=false)
learning_curve(mach, train, test, [2000, 3000], restart=false)
cv(mach, vcat(test, train))

# test compete function:
@test compete(randn(1000) + 5, randn(1000))[1] == '1'

load_iris()
load_ames()
