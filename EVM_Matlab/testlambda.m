clear;

dataDir = './data';
resultsDir = 'ResultsSIGGRAPH2012';

mkdir(resultsDir);

inFile = fullfile(dataDir,'myface2.mp4');
fprintf('Processing %s\n', inFile);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,20,100,0.5,10,30, 0);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,20,200,0.5,10,30, 0);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,20,300,0.5,10,30, 0);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,20,60,0.5,10,30, 0);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,20,40,0.5,10,30, 0);