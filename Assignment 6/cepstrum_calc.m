function cepstrum = cepstrum_calc(audio)

audio_part = cat(1,audio,zeros(1024-length(audio),1));

spectrum_audio_part_2 = fftshift(fft(audio_part));

cepstrum = fftshift(fft(log10(abs(spectrum_audio_part_2).^2 + 1e-50)));

end