#include "Perceptron.h"
#include "Edge.h"

using namespace std;

namespace NeNet
{

Perceptron::Perceptron(Type type, u_int layer, u_int index) :
    _type(type),
    _layer(layer),
    _index(index),
    _delta(0),
    _output(0)
{
    /* Initializing activation functions
     *
     * Sigmoid activation functions are used for input and hidden neuron
     * and identity functions for output neurons
     */
    
    if (_type == INPUT || _type == HIDDEN || _type == OUTPUT)
    {
        _activationFun = [](double x) {
            return 1.0 / (1.0 + exp(-1.0 * x));
        };
        
        auto actFun = _activationFun;
        _activationFunDer = [actFun](double x) {
            return actFun(x) * (1.0 - actFun(x));
        };
    }
    else
    {
        _activationFun = [](double x) {
            return x;
        };
        
        _activationFunDer = [](double x) {
            return 1.0;
        };
    }
    
    /* Initializing error functions 
     *
     * Square distance error function used
     */
    
    _errorFun = [](double networkOutput, double sampleOut) {
        return pow(networkOutput - sampleOut, 2);
    };
    
    _errorFunDer = [](double networkOutput, double sampleOut) {
        return 2 * (networkOutput - sampleOut);
    };
    
}

void Perceptron::processInputs()
{
    _weightedSum = 0;
    for (const auto &predecessor : _predecessors)
    {
        _weightedSum += predecessor.lock()->getWeightedValue();
    }
    
    _output = _activationFun(_weightedSum);
        
    for (const auto &successor : _successors)
    {
        successor.lock()->setValue(_output);
    }
}

void Perceptron::calculateDelta(double sampleOutput)
{
    if (_type == OUTPUT)
    {
        _delta = _errorFunDer(_output, sampleOutput) * _activationFunDer(_weightedSum);
        for (const auto edge : _predecessors)
        {
            edge.lock()->setError(edge.lock()->getValue() * _delta);
        }
    }
    else
    {
        _delta = 0;
        for (const auto edge : _successors)
        {
            _delta += edge.lock()->getSuccessorDelta() * edge.lock()->getWeight();
        }
        
        for (const auto edge : _predecessors)
        {
            edge.lock()->setError(edge.lock()->getValue() *
                                  _activationFunDer(_weightedSum) *
                                  _delta);
        }
    }
}

}