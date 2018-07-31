# Koala ![](logo.png) 

A Julia machine learning environment combining convenience and control,
through a combination of high and low-level interfaces. Currently in
development and experimental.

A common high-level interface is provided through the systematic
implementation of default pretransformations of training and testing
data. In particular, all learning algorithms receive learning data in
a common format (a `DataFrame` object) with all algorithm-specific
transformations occurring under the hood. This allows for the quick and
efficient comparison of several learning models. To mitigate against
data leakage, data transformations are "fit" on training data
only. However, there is a provision for dealing automatically in
testing or (cross) validation with classes of categorical variables
not seen during in the fit, a common annoyance.

For an introductory tour, clone the repositories into directories that your
Julia installation can find and run [docs/tour.jl](docs/tour.jl) in your
Julia REPL.

### Current Koala machine learning library

[KoalaTrees](https://github.com/ablaom/KoalaTrees.jl): Supervised learning with single regularized trees 

[KoalaEnsembles](https://github.com/ablaom/KoalaEnsembles.jl): Build weighted ensemble learners (e.g., random forests, extreme random forests)

[KoalaLightGBM](https://github.com/ablaom/KoalaLightGBM.jl): A Koala
wrap of Microsoft's gradient tree boosting
[algorithm](https://github.com/Microsoft/LightGBM)

[KoalaElasticNet](https://github.com/ablaom/KoalaElasticNet.jl): The elastic net and lasso linear predictors

[KoalaRidge](https://github.com/ablaom/KoalaRidge.jl): Ridge regression and classification

[KoalaKNN](https://github.com/ablaom/KoalaKNN.jl): K-nearest neighbor 

[KoalaTransforms](https://github.com/ablaom/KoalaTransforms.jl): A library of common data transformations (and dependency of several of the other libraries)

KoalaFlux (coming soon): A wrap of Mike Innes' beautiful Julia
implementation of neural networks, including a facility to learn
categorical feature embeddings.

At present the above implement supervised regression (todo:
classification). To learn how to wrap your favorite machine learning
code for use in Koala, refer to:

[KoalaLow](https://github.com/ablaom/KoalaLow.jl): To expose Koala's low-level interface

