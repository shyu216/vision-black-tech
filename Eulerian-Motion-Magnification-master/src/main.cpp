#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <map>

#include "eulerian_motion_mag.h"

int main(int argc, char **argv)
{
    std::string input_filename;
    std::string output_filename;
    int input_width = 0;
    int input_height = 0;
    int output_width = 0;
    int output_height = 0;
    double alpha = 20;
    double lambda_c = 16;
    double cutoff_freq_low = 0.05;
    double cutoff_freq_high = 0.4;
    double chrom_attenuation = 0.1;
    double exaggeration_factor = 2.0;
    double delta = 0;
    double lambda = 0;
    int levels = 5;

    // print something to the console
    std::cout << "Hello, World!" << std::endl;

    if (argc <= 1)
    {
        // std::cerr << "Error: Input param filename must be specified!" << std::endl;
        // return 1;

        // Set default input param

        // input_filename		= data/baby.mp4
        // output_filename		= data/baby_mag.avi

        // input_width			= 640
        // input_height		= 480
        // output_width		= 960
        // output_height		= 544

        // alpha				= 10
        // lambda_c			= 16
        // cutoff_freq_low		= 0.01
        // cutoff_freq_high	= 1.0
        // chrom_attenuation	= 0.1
        // exaggeration_factor	= 2.0
        // delta				= 0
        // lambda				= 0
        // levels				= 6

        input_filename = "C:/Users/LMAPA/Documents/GitHub/vision-black-tech/Eulerian-Motion-Magnification/data/baby.mp4";
        output_filename = "C:/Users/LMAPA/Documents/GitHub/vision-black-tech/Eulerian-Motion-Magnification/data/baby_mag.avi";
        input_width = 640;
        input_height = 480;
        output_width = 960;
        output_height = 544;
        alpha = 10;
        lambda_c = 16;
        cutoff_freq_low = 0.01;
        cutoff_freq_high = 1.0;
        chrom_attenuation = 0.1;
        exaggeration_factor = 2.0;
        delta = 0;
        lambda = 0;
        levels = 6;
    }
    else
    {

        // Read input param file
        std::ifstream file(argv[1]);
        if (!file.is_open())
        {
            std::cerr << "Error: Unable to open param file: " << std::string(argv[1]) << std::endl;
            return 1;
        }

        // Parse param file for getting parameter values
        std::map<std::string, std::string> params;
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, '=') && std::getline(iss, value))
            {
                params[key] = value;
            }
        }

        if (params.find("input_filename") != params.end())
            input_filename = params["input_filename"];
        if (params.find("output_filename") != params.end())
            output_filename = params["output_filename"];
        if (params.find("input_width") != params.end())
            input_width = std::stoi(params["input_width"]);
        if (params.find("input_height") != params.end())
            input_height = std::stoi(params["input_height"]);
        if (params.find("output_width") != params.end())
            output_width = std::stoi(params["output_width"]);
        if (params.find("output_height") != params.end())
            output_height = std::stoi(params["output_height"]);
        if (params.find("alpha") != params.end())
            alpha = std::stod(params["alpha"]);
        if (params.find("lambda_c") != params.end())
            lambda_c = std::stod(params["lambda_c"]);
        if (params.find("cutoff_freq_low") != params.end())
            cutoff_freq_low = std::stod(params["cutoff_freq_low"]);
        if (params.find("cutoff_freq_high") != params.end())
            cutoff_freq_high = std::stod(params["cutoff_freq_high"]);
        if (params.find("chrom_attenuation") != params.end())
            chrom_attenuation = std::stod(params["chrom_attenuation"]);
        if (params.find("exaggeration_factor") != params.end())
            exaggeration_factor = std::stod(params["exaggeration_factor"]);
        if (params.find("delta") != params.end())
            delta = std::stod(params["delta"]);
        if (params.find("lambda") != params.end())
            lambda = std::stod(params["lambda"]);
        if (params.find("levels") != params.end())
            levels = std::stoi(params["levels"]);
    }
    
    // EulerianMotionMag
    EulerianMotionMag *motion_mag = new EulerianMotionMag();

    // Set params
    motion_mag->setInputFileName(input_filename);
    motion_mag->setOutputFileName(output_filename);
    motion_mag->setInputImgWidth(input_width);
    motion_mag->setInputImgHeight(input_height);
    motion_mag->setOutputImgWidth(output_width);
    motion_mag->setOutputImgHeight(output_height);
    motion_mag->setAlpha(alpha);
    motion_mag->setLambdaC(lambda_c);
    motion_mag->setCutoffFreqLow(cutoff_freq_low);
    motion_mag->setCutoffFreqHigh(cutoff_freq_high);
    motion_mag->setChromAttenuation(chrom_attenuation);
    motion_mag->setExaggerationFactor(exaggeration_factor);
    motion_mag->setDelta(delta);
    motion_mag->setLambda(lambda);
    motion_mag->setLapPyramidLevels(levels);

    // Init Motion Magnification object
    bool init_status = motion_mag->init();
    if (!init_status)
        return 1;

    // Run Motion Magnification
    motion_mag->run();

    // Exit
    delete motion_mag;
    return 0;
}