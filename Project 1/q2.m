folder ='1_1 1_2 3_1 3_2 4_1 4_2 6_1 6_2 7_1 7_2 8_1 8_2 9_1 9_2 13_1 15_1 15_2 16_1 16_2 18_1 18_2 19_1 19_2';
%folder = '8_1';
tshift=0.015;
list_folder = split(folder);

for k =1:length(list_folder) 
    
    disp("Speaker : "+string(list_folder(k)));
    
    mkdir("Cepstrum_15ms_10ms/"+list_folder(k));
    
    for j = 1:5
    
    [y,Fs] = audioread("Text Dependant SR/"+list_folder(k)+"/"+j+".wav");
    disp(Fs);
    t=1;
    file_write = fopen("Cepstrum_15ms_10ms/"+list_folder(k)+"/"+j+".csv","w");
    l=length(y);
    while(l>t+Fs*tshift)
    %for i = 1: floor(length(y)/(Fs*tshift))
        audio = y(t :t+Fs*tshift);
        t = t+ Fs*0.010;
        cepstrum = cepstrum_calc(audio);
        fprintf(file_write,'%d %d\n',real(cepstrum),imag(cepstrum));
    end
    
    fclose(file_write);
    end
    
end
