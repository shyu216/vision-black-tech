clear;

dataDir = './data';
resultsDir = 'ResultsSIGGRAPH2012';

mkdir(resultsDir);

inFile = fullfile(dataDir,'myface2.mp4');
fprintf('Processing %s\n', inFile);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,30,80,0.0001,1,30, 0);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,30,80,1,5,30, 0);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,30,80,5,10,30, 0);