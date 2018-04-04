function [ envelope ] = BpPowerEnvelope( x,fmin,fmax,fs )
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
%% Computes bandpass power envelope by Hilbert transform 
[b,a]=butter(3,[fmin/(fs/2) fmax/(fs/2)],'bandpass');
x=filtfilt(b,a,x);
envelope=abs(hilbert(x)).^2;



end

