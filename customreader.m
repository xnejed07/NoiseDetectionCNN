function  data = customreader( filename )
%===================================================================
% Intracerebral EEG artifact identification using convolutional neural networks
%===================================================================
%% Authors
%===================================================================
% Petr Nejedly*,1,2,3
% Jan Cimbalnik, 1
% Petr Klimes, 1, 2
% Filip Plesinger, 2
% Josef Halamek, 2,
% Vaclav Kremen, 3,6
% Benjamin Brinkmann, 3,6
% Martin Pail, 4
% Milan Brazdil, 4,5
% Gregory Worrell 3,6
% Pavel Jurak 2
%===================================================================
%% Affiliation
%===================================================================
% 1) International Clinical Research Center, St. Anne’s University Hospital, Brno, Czech Republic
% 2) Institute of Scientific Instruments, The Czech Academy of Sciences, Brno, Czech Republic
% 3) Mayo Systems Electrophysiology Laboratory, Department of Neurology, Mayo Clinic, Rochester, Minnesota, U.S.A.
% 4) Brno Epilepsy Center, Department of Neurology, St. Anne’s University Hospital and Medical Faculty of Masaryk University, Brno, Czech Republic
% 5) CEITEC – Central European Institute of Technology, Masaryk University, Brno, Czech Republic
% 6) Department of Physiology and Biomedical Engineering, Mayo Clinic, Rochester, Minnesota, U.S.A.
%===================================================================
%% This is data loader for CNN object
% Works only for data with sampling frequency 500Hz
% Evaluation of power envelopes in specified frequency range
% Normalization of each signal to z-score
% RAW signal is bandpass filtered at 900Hz
[b,a]=butter(3,900/2500,'low');
data=load(filename);
data=data.data;
data(2,:)=zscore(BpPowerEnvelope(data(1,:),20,100,5000));
data(3,:)=zscore(BpPowerEnvelope(data(1,:),80,250,5000));
data(4,:)=zscore(BpPowerEnvelope(data(1,:),200,600,5000));
data(5,:)=zscore(BpPowerEnvelope(data(1,:),500,900,5000));
data(1,:)=zscore(filtfilt(b,a,data(1,:)));
end

