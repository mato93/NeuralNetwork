//
//  Simple implementation of standard multilayered neural network.
//
//  NB: gnuplot required for graphical output of trained 3D function
//
//  Created by Matej Hamas in September 2014.
//  Copyright (c) 2014 Matej Hamas. All rights reserved.
//  Licensed under BSD

#include "NeuralNetwork.h"

#include <random>
#include <fstream>
#include <sstream>
#include <unistd.h>

class Edge;

//#define VERBOSE

using namespace std;

namespace NeNet
{

NeuralNetwork::NeuralNetwork(const int numOfInputs,
                             const std::vector<int> numsOfPerceptrons) :
    _numOfInputs(numOfInputs),
    _numOfLayers((int)numsOfPerceptrons.size()),
    _numsOfPerceptrons(numsOfPerceptrons)

{
    /* Creating the network skelet */
    for (int i = 0; i < _numOfLayers; i++)
    {
        vector<shared_ptr<Perceptron>> temp;
        for (auto j = 0; j < _numsOfPerceptrons[i]; j++)
        {
            Type t;
            if (i == 0) {
                t = INPUT;
            } else if (i == _numOfLayers - 1) {
                t = OUTPUT;
            } else {
                t = HIDDEN;
            }
            
            temp.push_back(make_shared<Perceptron>(t, i, j));
        }
        _network.push_back(temp);
    }
        
    /* Initializing INPUT layer*/
    for (int i = -1; i < numOfInputs; i++)
    {
        for (int perceptron = 0; perceptron < _network[0].size(); perceptron++)
        {
            _edges.push_back(make_shared<Edge>(nullptr, _network[0][perceptron], (i == -1) ? FIXED_VALUE : VARIABLE_VALUE));
            _network[0][perceptron]->addPredecessor(_edges.back());
        }
    }
    
    /* Initializing HIDDEN layers */
    for (int layer = 0; layer < _numOfLayers - 1; layer++)
    {
        const int originLayerSize = (int)_network[layer].size();
        const int destinationLayerSize = (int)_network[layer + 1].size();
        for (int i = -1; i < originLayerSize; i++)
        {
            for (int j = 0; j < destinationLayerSize; j++)
            {
                if (i == -1) {
                    _edges.push_back(make_shared<Edge>(nullptr, _network[layer + 1][j], FIXED_VALUE));
                } else {
                    _edges.push_back(make_shared<Edge>(_network[layer][i], _network[layer + 1][j], VARIABLE_VALUE));
                }
                
                
                if (i >= 0)
                {
                    _network[layer][i]->addSuccessor(_edges.back());
                }
                _network[layer + 1][j]->addPredecessor(_edges.back());
            }
        }
    }
    
    /* Initializing OUTPUT layer */
    for (int i = 0; i < _network[_numOfLayers-1].size(); i++)
    {
        _outputEdges.push_back(make_shared<Edge>(_network[_numOfLayers - 1][i], nullptr, VARIABLE_VALUE));
        _network[_numOfLayers - 1][i]->addSuccessor(_outputEdges.back());
    }
}

double NeuralNetwork::forwardPropagateWithError(const vector<double>& input, double output)
{
    forwardPropagate(input);
    return pow(useForSingleOutput(input) - output, 2);
}
    
void NeuralNetwork::forwardPropagate(const vector<double>& input)
{
    /* Place input values on input edges */
    for (int i = 0; i < _numOfInputs; i++)
    {
        for (int j = 0; j < _network[0].size() ; j++)
        {
            _edges[(i + 1) * _network[0].size() + j]->setValue(input[i]);
        }
    }
    
    /* Propagate through network */
    for (const auto &layer : _network)
    {
        for (const auto &perceptron : layer)
        {
            perceptron->processInputs();
        }
    }
}

void NeuralNetwork::backwardPropagate(const double sampleOutput)
{
    for (int i = _numOfLayers - 1; i >= 0; i--)
    {
        for (auto perceptron : _network[i])
        {
            perceptron->calculateDelta(sampleOutput);
        }
    }
    
}

    
void NeuralNetwork::train(const vector<pair<vector<double>, double>>& patterns,
                          const int numOfEpochs,
                          const double lowerBound,
                          const double upperBound,
                          double stepSize,
                          const bool decreaseLearningRate,
                          const double minStepSize)
{
    const double stepSizeDecrease = (stepSize - minStepSize) / numOfEpochs;
    
    /* Initializing random weights to edges */
    srand((unsigned int)time(nullptr));
    for (auto edge : _edges)
    {
        double random = ((double)rand() / RAND_MAX) * (upperBound - lowerBound) + lowerBound;
        edge->setWeight(random);
    }
    
#ifdef VERBOSE
    cout << "-----INITIAL WEIGHTS-----\n";
    for(const auto& edge : _edges)
    {
        cout << "w: " << edge->getWeight() << " v: " << edge->getValue() << " e: " << edge->getError() << endl;
    }
    cout << endl;
#endif
    
    /* Running sequential training on the neural network */
    int index = 0;
    for(int i = 0; i < numOfEpochs; i++) {
        double error = 0;
        for (const auto& pattern : patterns)
        {
            error += forwardPropagateWithError(pattern.first, pattern.second);
            backwardPropagate(pattern.second);
            
            for (auto edge : _edges)
            {
                edge->setWeight(edge->getWeight() - edge->getError() * stepSize);
            }
            
#ifdef VERBOSE
            cout << "----------PATTERN " << index << "----------\n";
            
            cout << "-----EDGES INFORMATION-----\n";
            for (const auto& edge : _edges)
            {
                cout << "w: " << edge->getWeight() << " v: " << edge->getValue() << " e: " << edge->getError() << endl;
            }
            
            cout << "--PERCEPTRONS INFORMATION--\n";
            for (int i = 0; i < _numOfLayers; i++)
            {
                for (int j = 0; j < _numsOfPerceptrons[i]; j++)
                {
                    cout << "P[" << i << ", " << j << "]: ";
                    auto p = _network[i][j];
                    cout << "o: " << p->getOutput() << " d: " << p->getDelta() << endl;
                }
            }
            cout << "---------------------------\n";
            cout << endl;
#endif
            index++;
        }
        cout << i << ": " << error / patterns.size() << endl;
        if (decreaseLearningRate)
        {
            stepSize -= stepSizeDecrease;
        }
        
    }
}

vector<double> NeuralNetwork::use(const vector<double>& input)
{
    forwardPropagate(input);
    
    vector<double> output;
    for(auto outputPerceptron : _network[_numOfLayers - 1])
    {
        output.push_back(outputPerceptron->getOutput());
    }
    
    return output;
}

function<double(double, double)> NeuralNetwork::get3DFunction()
{
    return [this](double x, double y) -> double {
        const vector<double> input = {x, y};
        // TODO change back
//        _network[1][0]->_activationFun = [](double x) {return x; };
//        _network[1][0]->_activationFunDer = [](double x) { return 1; };
        return useForSingleOutput(input);
    };
}

void NeuralNetwork::createDataFile3D(const double minX, const double maxX, const double numOfXPoints,
                                     const double minY, const double maxY, const double numOfYPoints,
                                     const std::string &filePath)
{
    const function<double(double, double)> trainedFunction = get3DFunction();
    
    const double stepX = (maxX - minX) / (numOfXPoints - 1);
    const double stepY = (maxY - minY) / (numOfYPoints - 1);
    
    ofstream dataFile;
    dataFile.open(filePath);
    dataFile << "#####################################################\n";
    dataFile << "# Data file for 3D function trained by neural network\n";
    dataFile << "#x : [" << minX << ", " << maxX << "], " << numOfXPoints << " points\n";
    dataFile << "#y : [" << minY << ", " << maxY << "], " << numOfYPoints << " points\n";
    dataFile << "#####################################################\n";
    
    for (double x = minX; x <= maxX; x += stepX)
    {
        for (double y = minY; y <= maxY; y += stepY)
        {
            dataFile << x << " " << y << " " << trainedFunction(x, y) << endl;
        }
    }
    
    dataFile.close();
}
    
void NeuralNetwork::plot3DWithGnuplot(const double minX, const double maxX, const double numOfXPoints,
                                      const double minY, const double maxY, const double numOfYPoints,
                                      const std::string& filePath,
                                      const std::string& outputPNGPath)
{
    FILE* gnuplot = popen("/usr/local/bin/gnuplot --persist","w");
    
    if(!gnuplot) {
        return;
    }
    stringstream ss;
    
    ss << "set xrange [" << minX << ":" << maxX << "]\n";
    ss << "set yrange [" << minY << ":" << maxY << "]\n";
    ss << "set dgrid3d " << numOfXPoints << "," << numOfYPoints << endl;
    ss << "set hidden3d\n";
    ss << "splot" << "\"" << filePath << "\" with lines notitle\n";

    fprintf(gnuplot, ss.str().c_str());

    stringstream ssPNG;
    ssPNG << "set terminal png\n";
    ssPNG << "set output " << "\"" << outputPNGPath << "\"\n";
    fprintf(gnuplot, ssPNG.str().c_str());
    
    fprintf(gnuplot, ss.str().c_str());

    pclose(gnuplot);
}

}

