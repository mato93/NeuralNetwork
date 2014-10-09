#pragma  once

#include <iostream>
#include <vector>
#include <Math.h>

namespace NeNet
{

enum Type
{
    INPUT = 0,
    HIDDEN = 1,
    OUTPUT = 2
};

class Edge;

class Perceptron
{
private:
    
    const u_int _layer;
    const u_int _index;
    
    Type _type;

    std::vector<std::weak_ptr<Edge>> _predecessors;
    std::vector<std::weak_ptr<Edge>> _successors;
public:
    std::function<double(double)> _activationFun;
    std::function<double(double)> _activationFunDer; // derivative of activation function
    std::function<double(double, double)> _errorFun;
    std::function<double(double, double)> _errorFunDer;
    
    double _delta;
    double _weightedSum;
    double _output;
    
public:
    Perceptron(Type type, u_int layer, u_int index);
    
    /* GETTERS */
    double getDelta() { return _delta; }
    double getOutput() { return _output; }
    double getType() { return _type; }
    
    void addPredecessor(std::shared_ptr<Edge> predecessor) {
        _predecessors.push_back(predecessor);
    }
    
    void addSuccessor(std::shared_ptr<Edge> successor) {
        _successors.push_back(successor);
    }
    
    /**
     * Used in forward propagation. 
     * Method aggregates weighted outputs from all predecessors,
     * feeds them to the activation functions and places the output on the output edges.
     */
    void processInputs();
    
    /**
     * Used in backward propagation.
     * Method calculates the delta of the multilayer perceptron 
     * stores it in member variable.
     */
    void calculateDelta(double sampleOutput);
    
    
};
    
}
