
mkdir -p transcriptions_prepro/
mkdir -p adus_text_prepro/
#mkdir audios/

for i in {1..29};
do
    #python3 prepro_transcriptions.py DATA/Transcripció/Debate$i.txt > transcriptions_prepro/Debate$i.txt

    python3 prepro_transcriptions_csv.py VivesDebate_v3/Debate$i.csv > adus_text_prepro/Debate$i.txt
done

#./youtube-dl -o "audios/Debate%(autonumber)s.%(ext)s" --extract-audio --audio-format wav -a videos_links.lst
#./youtube-dl -o "audios/Debate%(autonumber)s.%(ext)s" --autonumber-start 14 --extract-audio --audio-format wav -a videos_links_14_upwards.lst
#./youtube-dl -o "audios/Debate%(autonumber)s.%(ext)s" --autonumber-start 21 --extract-audio --audio-format wav -a videos_links_21_upwards.lst
