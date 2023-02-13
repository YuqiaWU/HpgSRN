This Matlab package solves the L_q-regularized generalized linear model:

            min_{x\in\mathbb{R}^n} f(Ax) + \lambda \|x\|_q^q. 

This package includes two kinds of $f$:
1. f(y) = || y - b ||^2;   2. f(y) = \sum_{i=1}^m log(1+e^{-y_i}).

To use this package, you need first run startup.m to install the software.

Examples on how to use 'HpgSRN' can be found in [demo folder].