
#ifndef GPOPTIMIZER
#define GPOPTIMIZER

#include <vector>

using namespace std;

namespace gplib {

    class OptAbstractFunction {
    public:
        virtual double operator()(vector<double> &params) = 0; //Evaluate the function to optimize
        virtual vector<double> derivative(vector<double> &params) = 0; //derivative with respect to params (Gradient)
    };

    class OptAbstractBounds {
    public:
        virtual bool operator()(vector<double> &params) = 0;
    };

    /**
    A implementation of bounds that by default does not apply any bound on the data. But you can use it to apply
    upper and lower bound on each variable independently.
    */
    class OptCubicBounds : OptAbstractBounds {
    public:
        void setBound(int id_param, pair<double> &bound, pair<bool> &apply);
        void unsetBound(int id_param);

        bool operator()(vector<double> &params) = 0;
    };

    /**
     A class to optimice a function with respect to a set of parameters, that allows to put bounds on
     the parameters.
     */
    class Optimizer {
    private:
        OptAbstractFunction *function_min; //Function to minimice
        OptAbstractBounds *params_bounds; //Bounds on the parameters
    public:
        Optimizer(OptAbstractFunction *function);
        Optimizer(OptAbstractFunction *function, OptAbstractBounds *bounds);
        ~Optimizer();
        void setFunction(OptAbstractFunction *new_function);
        OptAbstractFunction *getFunction();
        void setBounds(OptAbstractBounds *new_bounds);
        OptAbstractBounds *getBounds();

        /**
        Function that minimice the objective function with respect of the parameters with some gradient
        method. The parameters start in initParams and return the optimization params. This function only
        guaratee a local optima. Overwrite this to implement different optimization methods. 
        */
        virtual vector<double> minimize(vector<double> &initParams);
    };
}

#endif
