clear;

dataDir = './data';
resultsDir = 'ResultsSIGGRAPH2012';

mkdir(resultsDir);

inFile = fullfile(dataDir,'myface2.mp4');
fprintf('Processing %s\n', inFile);

% Motion
amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,20,80,0.5,10,30, 0);

% Color
amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,100,6,50/60,60/60,30, 1);

inFile = fullfile(dataDir,'b5.mp4');
fprintf('Processing %s\n', inFile);

amplify_spatial_lpyr_temporal_iir(inFile, resultsDir, 10, 16, 0.5, 0.05, 0.1);
