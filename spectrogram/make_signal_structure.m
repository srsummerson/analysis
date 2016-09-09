function ...
    s=make_signal_structure(varargin)

% function ...
%     s=make_signal_structure(...
%     'raw_signal',raw_signal,...
%     'output_type','analytic',...
%     'signal_parameters',sp);
%
% inputs --
%   raw_signal: real, time-domain signal
%   output_type: either 'real' or 'analytic'
%   signal_parameters: structure given as output from
%   get_sp.m
%
% outputs --
%   s: structure with fields
%         s.time_domain: signal in time domain
%         s.frequency_domain: signal in frequency domain

% test to see if the cell varargin was passed directly from
% another function; if so, it needs to be 'unwrapped' one layer
if length(varargin)==1 % should have at least 2 elements
    varargin=varargin{1};
end

for n=1:2:length(varargin)-1
    switch lower(varargin{n})
        case 'raw_signal'
            raw_signal=varargin{n+1};
        case 'output_type'
            output_type=varargin{n+1};
        case 'signal_parameters'
            sp=varargin{n+1};
    end
end

s.frequency_domain=fft(raw_signal,...
    sp.number_points_frequency_domain);


if sum(abs(imag(raw_signal)))~=0 % signal already complex
    s.time_domain=raw_signal; % do not change
elseif isequal(output_type,'real')
    s.time_domain=raw_signal;
elseif isequal(output_type,'analytic')
    s.time_domain=[];
    s.time_domain=ifft(s.frequency_domain,...
        sp.number_points_frequency_domain);
    s.time_domain=s.time_domain(...
        1:sp.number_points_time_domain);
    % this step scales analytic signal such that
    % real(analytic_signal)=raw_signal, but note that
    % analytic signal energy is double that of raw signal energy
    % sum(abs(raw_signal).^2)=0.5*sum(abs(s.time_domain).^2)
    s.frequency_domain(sp.frequency_support<0)=0;
    s.frequency_domain(sp.frequency_support>0)=...
        2*s.frequency_domain(sp.frequency_support>0);
end

