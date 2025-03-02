clear;

dataDir = './data';
resultsDir = 'ResultsSIGGRAPH2012';

mkdir(resultsDir);

inFile = fullfile(dataDir,'myface2.mp4');
fprintf('Processing %s\n', inFile);

% Motion
amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,50,80,0.5,10,30, 0.1);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,50,80,0.5,10,30, 0.2);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,50,80,0.5,10,30, 0.4);

amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,50,80,0.5,10,30, 0.8);
