function [Power,cf_list] = make_spectrogram(lfp,Fs,fmax,trialave,makeplot)

%lfp is broadband signal in trials x time
%Fs is sampling frequency
%fmax = maximum frquency in spectrogram
%trialave is 0 or 1 (average over trials)
%makeplot is 0 or 1 to create a plot. If makeplot==1, it will be
%trial averaged.
%
%Has dependencies: filter_LFP
%
%2016 ELR

freq_band=zeros(2,length(lfp(1,:)));
freq_band(1,:)=1;
freq_band(2,:)=fmax;

powers=NaN(1,50,length(freq_band));
    for k=1:length(lfp(:,1))
        signal=lfp(k,:);
        [power,cf_list]=filter_LFP(signal,freq_band,1,Fs);
       powers(k,:,:)=power;
     end
     if trialave==1
     Power = squeeze(nanmean(powers,1));
     else
         Power = powers;
     end
     
     if makeplot==1
         figure
         if length(size(Power))>2
             Power = squeeze(nanmean(powers,1));
         end
             imagesc([],[],Power);
             set(gca,'YDir','Normal')
             set(gca,'YTick',[0:5:50]);
             set(gca,'YTickLabel',[0 cf_list(5:5:50)])
     end
end
