#pragma  once
#include "Perceptron.h"

#include <iostream>

namespace NeNet
{

static int _counter = 0;

enum EdgeType
{
    FIXED_VALUE = 0,
    VARIABLE_VALUE = 1
};

class Edge
{
private:
    
    EdgeType _type;
    
    double _value; // output of previous perceptron
    double _weight;
    double _error; // this quantity is being minimalized
    
    std::weak_ptr<Perceptron> _from;
    std::weak_ptr<Perceptron> _to;
    
    int _ID; // for debugging purposes
    
public:
    
    Edge(std::shared_ptr<Perceptron> from,
         std::shared_ptr<Perceptron> to,
         EdgeType type)
        : _value(1),
          _weight(1),
          _error(std::numeric_limits<int>::max()),
         _from(from),
         _to(to),
        _type(type)
    {
        _ID = _counter++;
    }
    
    double getValue() { return _value; }
    void setValue(double value) { _value = value; }
    
    double getWeight() { return _weight; }
    void setWeight(double weight) { _weight = weight; }
    
    double getError() { return _error; }
    void setError(double error) { _error = error; }
    
    double getWeightedValue() { return _value * _weight; }
    
    double getSuccessorDelta() { return _to.lock()->getDelta(); }
};

}
