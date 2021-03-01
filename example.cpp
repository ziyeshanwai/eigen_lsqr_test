#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "LSQR.h"
#include "time.h"
#include "fstream"
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

int main_0()
{
    //Matrix size m x n
    int m = 2000*2;
    int n = 1000*2;
    double gamma = 1e-3;
    int max_ite = 300;
    clock_t  start, finish;
    // Determine A randomly
    std::cout<<"m: "<<m<<"\nn: "<<n<<std::endl;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    Eigen::SparseMatrix<double> Asp = A.sparseView();

    // Determine true minimizer x randomly
    Eigen::VectorXd x_t = Eigen::VectorXd::Random(n);
    Eigen::VectorXd error = Eigen::VectorXd::Random(n); // 需要赋值, 不然报错
    // Determine b (= A x_t)
    Eigen::VectorXd b = Asp * x_t;

    // Minimizer x
    Eigen::VectorXd x(n);
    x.setZero();

    //Run LSQR
    start = clock();
    std::cout<<"gamma: "<<gamma<<std::endl;
    std::cout<<"max_ite: "<<max_ite<<std::endl;
    std::cout<<"A rows is "<<Asp.rows()<<std::endl;
    std::cout<<"A cols is "<<Asp.cols()<<std::endl;
    std::cout<<"b rows is "<<b.rows()<<std::endl;
    std::cout<<"b cols is "<<b.cols()<<std::endl;
    LSQR lsqr = LSQR(Asp, b, x, gamma, max_ite);

    x = lsqr.SolutionX();
    finish = clock();
    std::cout <<"it takes " << (double)(finish-start)/CLOCKS_PER_SEC << "s to lsqr" << std::endl;

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Solution obtained by LSQR" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    for (int i = 0; i < x.rows(); i++){
        error(i) = x(i) - x_t(i);
    }

    std::cout<<"norm error is " << error.norm() <<std::endl;
    return 0;
}

int main()
{
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor>SMatrixXd;
    SMatrixXd A;
    SMatrixXd b;
    Eigen::MatrixXd bb;
//    Eigen::VectorXd bbb;
    double gamma = 1e-4;
    int max_ite = 3000;
    clock_t start, finish;

    Eigen::loadMarket(A, "\\\\192.168.20.63\\ai\\double_camera_data\\2020-08-21\\161240\\output_manal\\debug\\A.mtx");
    Eigen::loadMarket(b, "\\\\192.168.20.63\\ai\\double_camera_data\\2020-08-21\\161240\\output_manal\\debug\\b.mtx");
    Eigen::VectorXd x(A.cols());
//    A = A.transpose();
    std::cout<<"rows of A is "<< A.rows() << std::endl;
    std::cout<<"cols of A is "<< A.cols() << std::endl;
    bb = b.toDense();
    std::cout<<"size of B is "<<b.size()<<std::endl;
    Eigen::VectorXd bbb(Eigen::Map<Eigen::VectorXd>(bb.data(), bb.cols()*bb.rows()));
    std::cout<<"rows of bbb is "<<bbb.rows()<<std::endl;
    std::cout<<"cols of bbb is "<<bbb.cols()<<std::endl;
    std::cout<<"rows of x is "<<x.rows()<<std::endl;
    std::cout<<"cols of x is "<<x.cols()<<std::endl;
    std::cout<<"gamma: "<<gamma<<std::endl;
    std::cout<<"max_ite: "<<max_ite<<std::endl;
    start = clock();
    LSQR lsqr = LSQR(A, bbb, x, gamma, max_ite);
    x = lsqr.SolutionX();
    finish = clock();
    std::cout<<"it takes "<< (double)(finish-start)/CLOCKS_PER_SEC << "s to lsqr" << std::endl;
    return 0;
}
