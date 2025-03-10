#include <QCoreApplication>
#include <QFileDialog>
#include <QSettings>
#include <QString>
#include <QDir>
#include <iostream>
#include "eulerian_motion_mag.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    // Default parameters
    QString input_filename;
    QString output_filename;
    int input_width = 640;
    int input_height = 480;
    int output_width = 960;
    int output_height = 544;
    double alpha = 10;
    double lambda_c = 16;
    double cutoff_freq_low = 0.01;
    double cutoff_freq_high = 1.0;
    double chrom_attenuation = 0.1;
    double exaggeration_factor = 2.0;
    double delta = 0;
    double lambda = 0;
    int levels = 6;

    // Let user select a video file
    input_filename = QFileDialog::getOpenFileName(nullptr, "Select Video File", QDir::homePath(), "Video Files (*.mp4 *.avi)");
    if (input_filename.isEmpty()) {
        std::cerr << "No file selected." << std::endl;
        return 1;
    }

    // Generate output filename in the same folder with parameters in the name
    QFileInfo fileInfo(input_filename);
    QString baseName = fileInfo.completeBaseName();
    QString outputDir = fileInfo.absolutePath();
    output_filename = QString("%1/%2_alpha%3_lambdaC%4_cutoffLow%5_cutoffHigh%6_chromAtt%7_exagFactor%8_delta%9_lambda%10_levels%11.avi")
                          .arg(outputDir)
                          .arg(baseName)
                          .arg(alpha)
                          .arg(lambda_c)
                          .arg(cutoff_freq_low)
                          .arg(cutoff_freq_high)
                          .arg(chrom_attenuation)
                          .arg(exaggeration_factor)
                          .arg(delta)
                          .arg(lambda)
                          .arg(levels);

    // EulerianMotionMag
    EulerianMotionMag *motion_mag = new EulerianMotionMag();

    // Set params
    motion_mag->setInputFileName(input_filename.toStdString());
    motion_mag->setOutputFileName(output_filename.toStdString());
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