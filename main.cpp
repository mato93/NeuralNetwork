//
//  Simple implementation of standard multilayered neural network.
//
//  NB: gnuplot required for graphical output of trained 3D function
//
//  Created by Matej Hamas in September 2014.
//  Copyright (c) 2014 Matej Hamas. All rights reserved.
//  Licensed under BSD

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "Document.h"
#include "NeuralNetwork.h"
#include "3DConsoleGrapher.h"

using namespace std;
using namespace NeNet;

static int zeros = 0;
static int ones = 0;


int main(int argc, const char *argv[])
{
    
    const int numOfInputs = 2;
    const vector<int> numsOfPerceptrons = { 7, 1};
    
    NeuralNetwork network(numOfInputs, numsOfPerceptrons);
    
    /* Generating random patterns */
    
    vector<pair<vector<double>, double>> patterns;
    const int numOfTrainingPatterns = 2000;
    const int numOfEpochs = 1000;
    const double lowerBound = 0;
    const double upperBound = 1;
    const double trainingRate = 0.1;
    
    /* Plane x + y */
//    function<double(double, double)> fun = [midVal](double x, double y) -> double {
//        if (x + y < midVal)
//        {
//            zeros++;
//            return 0;
//        } else
//        {
//            ones++;
//            return 1;
//        }
//    };
    
    /* Left half low*/
//    function<double(double, double)> fun = [lowerBound, upperBound, midVal](double x, double y) -> double {
//        if (x < lowerBound + (upperBound - lowerBound) / 2)
//        {
//            zeros++;
//            return 0;
//        }
//        else
//        {
//            ones++;
//            return 1;
//        }
//    };
    
    /* Top right corner low */
//    function<double(double, double)> fun = [lowerBound, upperBound](double x, double y) -> double {
//        const double mid = lowerBound + (upperBound - lowerBound) / 2;
//        if (mid <= x && mid <= y)
//        {
//            zeros++;
//            return 0;
//        }
//        else
//        {
//            ones++;
//            return 1;
//        }
//    };
    
    /* Bottom right and top left corners low */
//    function<double(double, double)> fun = [lowerBound, upperBound](double x, double y) -> double {
//        const double mid = lowerBound + (upperBound - lowerBound) / 2;
//        if ((mid <= x && y <= mid) || (x <= mid && mid <= y))
//        {
//            zeros++;
//            return 0;
//        }
//        else
//        {
//            ones++;
//            return 1;
//        }
//    };
    
    /* High in the center */
//    function<double(double, double)> fun = [lowerBound, upperBound](double x, double y) -> double {
//        const double mid = lowerBound + (upperBound - lowerBound) / 2;
//        const double eps = mid / 2;
//        if (mid - eps <= x && x <= mid + eps && mid - eps <= y && y <= mid + eps)
//        {
//            zeros++;
//            return 1;
//        }
//        else
//        {
//            ones++;
//            return 0;
//        }
//    };
    
    /* Test of function in 6D space*/
//    function<double(double, double, double, double, double)> fun = [](double a, double b, double c, double d, double e) -> double {
//        if (a + b + c + d + e < 2.5)
//        {
//            zeros++;
//            return 0;
//        }
//        else
//        {
//            ones++;
//            return 1;
//        }
//    };
    
    /* Torus */
    function<double(double, double)> fun = [lowerBound, upperBound](double x, double y) -> double {
        const double mid = lowerBound + (upperBound - lowerBound) / 2;
        double distance = sqrt(pow(x - mid, 2) + pow(y - mid, 2));
        //double randError = ((double)rand() / RAND_MAX) / 2;
        if (/*(0.1 < distance && distance < 0.2) || (0.2 < distance && distance < 0.4)*/ distance <= 0.3)
        {
            zeros++;
            return 1;
            //return randError;
        }
        else
        {
            ones++;
            return 0;
            //return 1 - randError;
        }
    };
    
    srand((u_int)time(nullptr));
    for (int i=0; i < numOfTrainingPatterns; i++)
    {
        double input1 = ((double)rand() / RAND_MAX) * (upperBound - lowerBound) + lowerBound;
        double input2 = ((double)rand() / RAND_MAX) * (upperBound - lowerBound) + lowerBound;

        auto output = fun(input1, input2);
        vector<double> v = {input1, input2};
        
        patterns.push_back(make_pair(v, output));
    }
    
    network.train(patterns, numOfEpochs, 0, 1, trainingRate);
    
    cout << "Training: " << numOfTrainingPatterns << endl;
    cout << "Zeros: " << zeros << endl;
    cout << "Ones : " << ones << endl;
    cout << "\n-----FINAL WEIGHTS-----\n";
    int temp = 0;
    for(auto edge : network.getEdges())
    {
        cout << temp << ": " << edge->getWeight() << endl;
        temp++;
    }
    cout << endl;
    
    network.show3DFunction(0,1,30,0,1,30,"/Users/Matej/Documents/NeNetworkOutput.dat", "/Users/Matej/Documents/NeNetworkOutput.png");
    
    /*
        ConsoleGrapher3D grapher3D(network.get3DFunction(), 0, 1, 0, 1, -0.1, 1.1);
        grapher3D.plot();
    */
    
    return 0;
}
