# cvxopt
Convex Optimization: Assignment and Project  
Runtian Zhai (ID: 1600012737)

## What is this?
This repository contains codes for assignments of the course
Convex Optimization by Professor Wen. 

## How to run the codes?
For a quick start, simply run Test_l1_regularized_problems.m in the
root folder and all methods will be automatically tested.
example_output.txt contains an example output of the program.  
A list of methods follows:

| Filename | Method |
| ------ | ------ |
| l1_pgd.m | Projected gradient method |
| l1_subgradient.m | Subgradient method |
| l1_smooth.m | Gradient method for smoothed primal |
| l1_smoothd.m | Smoothing with decreasing lambda |
| l1_smoothf.m | Smoothing with fast gradient method |
| l1_smooth(1,2).m | Other smoothers |
| l1_proximal.m | Proximal gradient method |
| l1_fistab.m | FISTA: Basic version |
| l1_fistad.m | FISTA: Descent version |
| l1_nes2.m | Nesterov's 2nd method |
| l1_cvx_mosek.m | Calling mosek from cvx |
| l1_mosek.m | Calling mosek directly |
| l1_cvx_gurobi.m | Calling gurobi from cvx |
| l1_gurobi.m | Calling gurobi directly |

## Acknowledgement
I would like to thank Professor Wen for this excellent course.
I would also like to thank TAs Jiang and Haoming, who spend their
time reading these assignments.