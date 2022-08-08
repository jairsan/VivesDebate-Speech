
mkdir transcriptions_prepro/
#mkdir audios/

for i in {1..29};
do
    transc=$(cat DATA/Transcripció/Debate$i.txt |  tr "\\n" " ")
    echo "0 <END>" $transc > transcriptions_prepro/Debate$i.txt
done

#./youtube-dl -o "audios/Debate%(autonumber)s.%(ext)s" --extract-audio --audio-format wav -a videos_links.lst
#./youtube-dl -o "audios/Debate%(autonumber)s.%(ext)s" --autonumber-start 14 --extract-audio --audio-format wav -a videos_links_14_upwards.lst
#./youtube-dl -o "audios/Debate%(autonumber)s.%(ext)s" --autonumber-start 21 --extract-audio --audio-format wav -a videos_links_21_upwards.lst
