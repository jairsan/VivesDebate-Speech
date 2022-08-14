out_folder=audios_16khz_mono_trim/
mkdir -p $out_folder;
for i in {1..29};
do
    start=$(head -n 1 DATA/BIO_arg_timestamps/Debate$i.txt | cut -f 2 -d " ") 
    end=$(tail -n 1 DATA/BIO_arg_timestamps/Debate$i.txt | cut -f 3 -d " ") 
    start_clip=$(echo $start | awk '{print $0-0.1}')
    end_clip=$(echo $end | awk '{print $0+0.1}')
    sox audios_16khz_mono/Debate$i.wav $out_folder/Debate"$i"_"$start_clip"_"$end_clip".wav trim $start_clip =$end_clip
done
