% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/models.R
\name{model_type}
\alias{model_type}
\alias{model_type.default}
\alias{model_type.lm}
\alias{model_type.glm}
\alias{model_type.ranger}
\alias{model_type.gam}
\alias{model_type.xgb.Booster}
\title{Define type of model}
\usage{
model_type(x)

\method{model_type}{default}(x)

\method{model_type}{lm}(x)

\method{model_type}{glm}(x)

\method{model_type}{ranger}(x)

\method{model_type}{gam}(x)

\method{model_type}{xgb.Booster}(x)
}
\arguments{
\item{x}{Model object for the model to be explained.}
}
\value{
Either \code{"classification"} or \code{"regression"}.
}
\description{
The function checks whether the model given by \code{x} is
supported, and if it is a regression- or a classification model. If \code{x} is
not a supported model the function will return an error message, otherwise it will
return either \code{"regression"} or \code{"classification"}.
}
\details{
See \code{\link{predict_model}} for more information about
what type of models \code{shapr} currently support.
}
