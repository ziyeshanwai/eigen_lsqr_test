#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

class LSQR
{
public:
	Eigen::VectorXd x;
	LSQR(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, const Eigen::VectorXd &x_ini, const double &gamma, const int max_ite);
	Eigen::VectorXd SolutionX();
};

