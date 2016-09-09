function g=complete_chirplet_parameters(g)

%function g=complete_chirplet_parameters(g)
%
% preferred usage:
% g.center_time=0;
% g.center_frequency=0;
% g.duration_parameter=0;
% g.chirp_rate=0;
% g=complete_chirplet_parameters(g);

% must have these:
if isfield(g,'center_time'); t0=g.center_time; else t0=0; end
if isfield(g,'center_frequency'); v0=g.center_frequency; end
if isfield(g,'chirp_rate'); c0=g.chirp_rate; end

% assign or compute duration parameter:
if isfield(g,'duration_parameter')
    s0=g.duration_parameter;
elseif isfield(g,'fractional_bandwidth')&(c0==0)
    fbw=g.fractional_bandwidth;
    s0=log((2*log(2))/(fbw^2*pi*v0^2));
elseif isfield(g,'frequency_domain_standard_deviation')&(c0==0)
    fstd=g.frequency_domain_standard_deviation;
    s0=-log(4*pi*fstd^2);
elseif isfield(g,'time_domain_standard_deviation')
    tstd=g.time_domain_standard_deviation;
    s0=log(4*pi*tstd^2);
end

% start with new version:
clear g
g.center_time=[];
g.center_frequency=[];
g.duration_parameter=[];
g.chirp_rate=[];
g.fractional_bandwidth=[];
g.time_domain_standard_deviation=[];
g.frequency_domain_standard_deviation=[];
g.time_frequency_covariance=[];
g.time_frequency_correlation_coefficient=[];
g.std_multiple_for_support=[];

% reenter given parameters:
g.center_time=t0;
g.center_frequency=v0;
g.duration_parameter=s0;
g.chirp_rate=c0;
% fixed parameter:
g.std_multiple_for_support=6;
% calculate other properties:
g.time_domain_standard_deviation=...
    sqrt(exp(s0)/(4*pi));
g.frequency_domain_standard_deviation=...
    sqrt((exp(-s0)+c0^2*exp(s0))/(4*pi));
g.time_frequency_covariance=(c0*exp(s0))/(4*pi);
g.time_frequency_correlation_coefficient=...
    g.time_frequency_covariance*(...
    g.time_domain_standard_deviation*...
    g.frequency_domain_standard_deviation)^-1;
g.fractional_bandwidth=2*sqrt(2*log(2))*...
    g.frequency_domain_standard_deviation/v0;% fwhm/center_frequency
