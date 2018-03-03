# Koala ![](logo.png) 

> A Julia machine learning environment combining convience and control,
through a combination of high and low-level interfaces. 

Currently in
development and experimental.

For an introductory tour, clone the repositories and run [this code](docs/tour.jl) in your Julia REPL.

## Current Koala machine learning library

[KoalaTrees](https://github.com/ablaom/KoalaTrees.jl): Supervised learning with single regularized trees 

[KoalaEnsembles](https://github.com/ablaom/KoalaEnsembles.jl): Build weighted ensemble learners (e.g., random forests, extreme random forests)

[KoalaLightGMB](https://github.com/ablaom/KoalaLightGBM.jl): A Koala wrap of Microsoft's tgradient tree boosting (algorithm)(https://github.com/Microsoft/LightGBM)]

[KoalaElasticNet](https://github.com/ablaom/KoalaElasticNet.jl): The elastic net and lasso linear predictors

[KoalaRidge](https://github.com/ablaom/KoalaRidge.jl): Ridge regression and classification

At present the above implement supervised regression (todo:
classification). To learn how to wrap your favourite machine learning
code for use in Koala, refer to:

[KoalaLow](https://github.com/ablaom/KoalaLow.jl): To expose Koala's low-level interface
