function [power,cf_list,varargout] = filter_LFP(signal,freq_band,varargin)
% FUNCTION [POWER,FREQUENCY_LIST] = FILTER_SIGNAL(SIGNAL,FREQ_BAND,optional=normalize)
% FUNCTION [POWER,FREQUENCY_LIST] = FILTER_SIGNAL(SIGNAL,FREQ_BAND,optional=normalize,optional=sampling_rate)
% FUNCTION [POWER,FREQUENCIY_LIST,PHASE] = FILTER_SIGNAL(SIGNAL,FREQ_BAND,optional=normalize,optional=sampling_rate)
%
% Filters the vector SIGNAL using a chirplet transfrom between FREQ_BAND.
% FREQ_BAND is a two element vector with values [minFreq maxFreq]
%
% Returns a matrix POWER with the power at each frequency across time.
% It also returns a vector FREQUENCY_LIST that contains the center
% frequencies used in the analysis. 
%
% the first optional input argument specifies if the power is to be normialized
% by the average power over time the default is true.
%
% the second optional argument is the sampling rate of the signal, if none
% is specified then the sampling rate is assumed to be 1000Hz, if this
% assumption is violated the results will be incorrect.
%
% The second ussage also returns the matrix PHASE containing the
% instantaneus phase for each frequeny as a function of time.
%
% The fractional bandwith used for each center frequency is 20% of the center freqency.
% 
% Dependencies: get_signal_parameters, make_signal_structure, make_center_frequencies, make_chirplet, 
%               filter_with_chirplet

if ~isempty(varargin),
    normalize=varargin{1};
else 
    normalize=true;
end

if (~isempty(varargin) && length(varargin)==2)
    sampling_rate=varargin{2};
else 
    sampling_rate=1000;
end

numpoints=length(signal);
sp=get_signal_parameters('sampling_rate',sampling_rate,'number_points_time_domain',numpoints);


s=make_signal_structure(...
    'raw_signal',signal,...
    'output_type','analytic',...
    'signal_parameters',sp);


minimum_frequency=freq_band(1); % Hz
maximum_frequency=freq_band(2); % Hz
number_of_frequencies=50;
minimum_frequency_step_size=.75;
%%%elr:  equally spaces frequencies
%temp1=[minimum_frequency:maximum_frequency];
%l=length(temp1)/number_of_frequencies;
%center_frequencies=[minimum_frequency+1:l:maximum_frequency];
%%%

center_frequencies=...
    make_center_frequencies(...
    minimum_frequency,...
    maximum_frequency,...
    number_of_frequencies,...
    minimum_frequency_step_size);

tfmat=zeros(number_of_frequencies,sp.number_points_time_domain);
cf_list=center_frequencies;

for f=1:number_of_frequencies
    clear g
    g.center_frequency=cf_list(f); % Hz
    g.fractional_bandwidth=0.2;
    g.chirp_rate=0;
    %g.duration_parameter=0;
    g=make_chirplet(...
        'chirplet_structure',g,...
        'signal_parameters',sp);
  
    fs=filter_with_chirplet(...
        'signal_structure',s,...
        'signal_parameters',sp,...
        'chirplet',g);
    
    tfmat(f,:)=fs.time_domain;
    
    
end

power=abs(tfmat).^2;
if normalize,
    power=bsxfun(@rdivide,power,mean(power,2));
end  %divides each frequency by the mean for that frequency(?)


varargout{1}=angle(tfmat);


