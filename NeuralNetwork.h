#pragma  once

#include "Perceptron.h"
#include "Edge.h"

#include <iostream>
#include <vector>

namespace NeNet
{
    
class NeuralNetwork
{
private:
    int _numOfInputs; // number of inputs (i.e. num of dimensions)
    int _numOfLayers; // including input or output layer
    std::vector<int> _numsOfPerceptrons; // number of perceptrons in each layer
    std::vector<std::vector<std::shared_ptr<Perceptron>>> _network;
    std::vector<std::shared_ptr<Edge>> _edges;
    std::vector<std::shared_ptr<Edge>> _outputEdges;
    
    /**
     * Triggers forward propagation in network with given input
     */
    void forwardPropagate(const std::vector<double>& input);
    
    double forwardPropagateWithError(const std::vector<double>& input, double output);
    
    /**
     * Triggers backward propagation in network for given sample (pattern) output
     */
    void backwardPropagate(const double sampleOutput);
    
public:
    
    NeuralNetwork(const int numOfInputs,
                  const std::vector<int> numsOfPerceptrons);
    
    std::vector<std::shared_ptr<Edge>> getEdges() { return _edges; }
    
    /**
     * Triggers training of the network given vector of training patterns.
     */
    void train(const std::vector<std::pair<std::vector<double>, double>>& patterns,
               const int numOfEpochs,
               const double lowerBound,
               const double upperBound,
               double stepSize,
               const bool decreaseLearningRate = false,
               const double minStepSize = 0.01);
    
    /**
     * Use the network for producing output.
     * Is equivalent to forward propagation step.
     */
    std::vector<double> use(const std::vector<double>& input);
    
    /**
     * Return output as double if only 1 output is expected 
     * (instead of vector of doubles).
     */
    double useForSingleOutput(const std::vector<double>& input) {
        return use(input)[0];
    };
    
    /**
     * Returns 3D function trained by network.
     * Usable only in case 3D function is being trained.
     */
    std::function<double(double, double)> get3DFunction();
    
    /**
     * Creates data file consisting of triples X Y Z,
     * where Z = trained_function (X, Y).
     */
    void createDataFile3D(const double minX, const double maxX, const double numOfXPoints,
                          const double minY, const double maxY, const double numOfYPoints,
                          const std::string& filePath);
    
    /**
     * Given dataFile, plots the trained 3D function using gnuplot.
     */
    void plot3DWithGnuplot(const double minX, const double maxX, const double numOfXPoints,
                           const double minY, const double maxY, const double numOfYPoints,
                           const std::string& filePath,
                           const std::string& outputPNGPath);
    
    /**
     * Aggregates creating data file and plotting the 3D graph using gnuplot
     */
    void show3DFunction(const double minX, const double maxX, const double numOfXPoints,
                        const double minY, const double maxY, const double numOfYPoints,
                        const std::string& filePath,
                        const std::string& outputPNGPath)
    {
        createDataFile3D(minX, maxX, numOfXPoints,
                         minY, maxY, numOfYPoints,
                         filePath);
        
        plot3DWithGnuplot(minX, maxX, numOfXPoints,
                          minY, maxY, numOfYPoints,
                          filePath, outputPNGPath);
    }
};

}