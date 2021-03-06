% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/clustering.R
\name{correlation_rectangles}
\alias{correlation_rectangles}
\title{Draw rectangles on the correlation matrix graph}
\usage{
correlation_rectangles(corr, cluster, k = 2, col = "yellow", lwd = 6)
}
\arguments{
\item{corr}{Numeric matrix. Needs to satisfy the conditions for a correlation matrix.}

\item{cluster}{An object of class \code{hclust}. See \code{\link[stats]{hclust}}}

\item{k}{Integer. Number of clusters}

\item{col}{Character. Represents the box color.}

\item{lwd}{Integer. Represents box line width.}
}
\description{
Draw rectangles on the correlation matrix graph
}
\author{
Anders Løland
}
\keyword{internal}
