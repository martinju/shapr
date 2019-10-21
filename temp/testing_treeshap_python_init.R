
#### PAPER EXPERIMENT FRAMEWORK ####
#### Use the current setup for all experiements in the paper to ease reproducablity etc.

#rm(list = ls())

library(shapr)
library(data.table)
library(mvtnorm)
library(condMVNorm)
library(stringi)

library(xgboost)

####################### ONLY TOUCH THINGS IN THIS SECTION ################################
experiment = "B"
true_model <- "PiecewiseConstant"
fitted_model <- "XGBoost"
variables <- "Gaussian" # Gaussian, Gaussianmix, or GenHyp
notes <- "All var equal contribution"
X_dim <- 3
source.local <- ifelse(exists("source.local"),source.local,FALSE)


nTrain <- 2000
nTest <- 100
w_threshold = 1 # For a fairer comparison, all models use the same number of samples (n_threshold)
n_threshold = 10^3 # Number of samples used in the Monte Carlo integration

pi.G <- 1
sd_noise = 0.1
rho <- ifelse(exists("rho"),rho,0.5) # Do not edit
mu.list = list(rep(0,X_dim))
mat <- matrix(rho,ncol=X_dim,nrow=X_dim)
diag(mat) <- 1
Sigma.list <- list(mat)

#### Defining the true distribution of the variables and the model

samp_variables <- function(n,pi.G,mu.list,Sigma.list){

  X <- joint.samp.func(n = n,
                       pi.G,
                       mu.list,
                       Sigma.list)
  return(X)
}


samp_model <- function(n,X,sd_noise){
  y <- stepwiseConstant_fun1(X[,1]) + stepwiseConstant_fun2(X[,2])*1 + stepwiseConstant_fun3(X[,3])*1+ rnorm(n = n,mean=0,sd=sd_noise)
  #    y <-(X[,1]<0)*1 + 0.1*X[,2] + (X[,2]>-1)*1 - (X[,3]<1)*1 + (X[,3]<-1)*4 - (X[,3]>-1)*(X[,2]<-1)*1.5+ rnorm(n = n,mean=0,sd=sd_noise)
}

fit_model_func <- function(XYtrain){
  xgb.train <- xgb.DMatrix(data = as.matrix(XYtrain[,-"y"]),
                           label = XYtrain[,y])

  params <- list(eta =  0.3,
                 objective = "reg:linear",
                 eval_metric = "rmse",
                 tree_method="hist") # gpu_hist

  model <- xgb.train(data = xgb.train,
                     params = params,
                     nrounds = 50,
                     print_every_n = 10,
                     ntread = 3)
  return(model)
}



####################################################################################################

#### Autoset helping variables. DO NOT TOUCH ####

X_GenHyp <- (variables=="GenHyp")
(joint_csv_filename <- paste0("all_results_experiment_",experiment,"_dim_",X_dim,"_",true_model,"_",fitted_model,"_",variables,".csv")) # May hardcode this to NULL for not saving to joint in testing circumstances
(initial_current_csv_filename <- paste0("current_results_experiment_",experiment,"_dim_",X_dim,"_",true_model,"_",fitted_model,"_",variables))


source("temp/paper_helper_funcs.R",local = source.local) # Helper functions these experiments (mainly computing the true Shapley values)


source("temp/source_specifying_seed_and_filenames.R",local = source.local) # Setting random or fixed seed and filenames.

#### Sampling train and test data ---------
# Creating the XYtrain, XYtest, Xtrain and Xtest objects
source("temp/source_sampling_data.R",local = source.local)

#### Fitting the model ----------
#set.seed(123)

#model <- fit_model_func(XYtrain)
#model <- xgb.load("temp/xgb.mod.obj")
#xgb.save(model,"temp/xgb.mod.obj") # Need to wait a bit after saving and then loading this in python

library(reticulate)
#virtualenv_create("py3_6-virtualenv", python = "/usr/bin/python3.6") # Creating virtual environment with Python 3.6
use_virtualenv("py3_6-virtualenv")
#py_install("xgboost",envname = "py3_6-virtualenv")
#py_install("shap",envname = "py3_6-virtualenv")

Xtest_mat <- as.matrix(Xtest)

Xtrain_mat <- as.matrix(Xtrain)


reticulate::repl_python()
#### Python code ####
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import time

model = xgb.Booster()  # init model
model.load_model("temp/xgb.mod.obj")

## kernel shap sends data as numpy array which has no column names, so we fix it
def xgb_predict(data_asarray):
  data_asDmatrix =  xgb.DMatrix(data_asarray)
  return model.predict(data_asDmatrix)

py_pred_test = xgb_predict(r.Xtest_mat) # Test predictions in python

#sum((py_pred_test-r.pred_test)**2) # checking equality with r predictions

### Applying TreeSHAP ###

treeshap_dependent_explainer = shap.TreeExplainer(model = model,
                                                  feature_dependence = "tree_path_dependent")

treeshap_dependent_with_traindata_explainer = shap.TreeExplainer(model = model,
                                                                 data = r.Xtrain_mat,
                                                                 feature_dependence = "tree_path_dependent")

treeshap_independent_explainer = shap.TreeExplainer(model = model,
                                          data = r.Xtrain_mat,
                                          feature_dependence = "independent")

kernelshap_independent_explainer = shap.KernelExplainer(model = xgb_predict,
                                                        data = r.Xtrain_mat)

treeshap_dependent_shap = treeshap_dependent_explainer.shap_values(r.Xtest_mat)
#treeshap_dependent_with_traindata_shap = treeshap_dependent_with_traindata_explainer.shap_values(r.Xtest_mat)
treeshap_independent_shap = treeshap_independent_explainer.shap_values(r.Xtest_mat)
kernelshap_independent_explainer = kernelshap_independent_explainer.shap_values(r.Xtest_mat,nsamples = int(100000),l1_reg=0)

exit

#### Exit python code ####

python_output <- list(treeshap_dependent_shap = py$treeshap_dependent_shap,
                      treeshap_independent_shap = py$treeshap_independent_shap,
                      kernelshap_independent_explainer = py$kernelshap_independent_explainer,
                      py_pred_test = py$py_pred_test)

save(python_output,file = "temp/python_results.Rdata")


