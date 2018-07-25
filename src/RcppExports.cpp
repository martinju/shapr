// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// distance_cpp
arma::Cube<double> distance_cpp(NumericMatrix Xtrain, NumericMatrix Xtest);
RcppExport SEXP _shapr_distance_cpp(SEXP XtrainSEXP, SEXP XtestSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type Xtrain(XtrainSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Xtest(XtestSEXP);
    rcpp_result_gen = Rcpp::wrap(distance_cpp(Xtrain, Xtest));
    return rcpp_result_gen;
END_RCPP
}
// impute_cpp
NumericMatrix impute_cpp(IntegerVector ID, IntegerVector Comb, NumericMatrix Xtrain, NumericMatrix Xtest, IntegerMatrix S);
RcppExport SEXP _shapr_impute_cpp(SEXP IDSEXP, SEXP CombSEXP, SEXP XtrainSEXP, SEXP XtestSEXP, SEXP SSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type ID(IDSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type Comb(CombSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Xtrain(XtrainSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Xtest(XtestSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type S(SSEXP);
    rcpp_result_gen = Rcpp::wrap(impute_cpp(ID, Comb, Xtrain, Xtest, S));
    return rcpp_result_gen;
END_RCPP
}
// weighted_matrix
arma::mat weighted_matrix(List features, int m, int n);
RcppExport SEXP _shapr_weighted_matrix(SEXP featuresSEXP, SEXP mSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type features(featuresSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(weighted_matrix(features, m, n));
    return rcpp_result_gen;
END_RCPP
}
// feature_matrix_cpp
NumericMatrix feature_matrix_cpp(List features, int nfeatures);
RcppExport SEXP _shapr_feature_matrix_cpp(SEXP featuresSEXP, SEXP nfeaturesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type features(featuresSEXP);
    Rcpp::traits::input_parameter< int >::type nfeatures(nfeaturesSEXP);
    rcpp_result_gen = Rcpp::wrap(feature_matrix_cpp(features, nfeatures));
    return rcpp_result_gen;
END_RCPP
}
// weights_train_comb_cpp
arma::mat weights_train_comb_cpp(arma::mat D, arma::mat S, double sigma, std::string kernel_metric);
RcppExport SEXP _shapr_weights_train_comb_cpp(SEXP DSEXP, SEXP SSEXP, SEXP sigmaSEXP, SEXP kernel_metricSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type S(SSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< std::string >::type kernel_metric(kernel_metricSEXP);
    rcpp_result_gen = Rcpp::wrap(weights_train_comb_cpp(D, S, sigma, kernel_metric));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_shapr_distance_cpp", (DL_FUNC) &_shapr_distance_cpp, 2},
    {"_shapr_impute_cpp", (DL_FUNC) &_shapr_impute_cpp, 5},
    {"_shapr_weighted_matrix", (DL_FUNC) &_shapr_weighted_matrix, 3},
    {"_shapr_feature_matrix_cpp", (DL_FUNC) &_shapr_feature_matrix_cpp, 2},
    {"_shapr_weights_train_comb_cpp", (DL_FUNC) &_shapr_weights_train_comb_cpp, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_shapr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
