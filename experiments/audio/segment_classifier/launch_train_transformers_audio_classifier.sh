train_files="../../../data_preparation/DATA/BIO_arg_timestamps/Debate1.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate2.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate3.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate4.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate5.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate6.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate7.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate8.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate9.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate10.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate11.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate12.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate13.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate14.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate15.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate16.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate17.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate18.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate19.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate20.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate21.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate22.txt"
dev_files="../../../data_preparation/DATA/BIO_arg_timestamps/Debate24.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate25.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate26.txt"
WAV_FOLDER=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/audios_16khz_mono/


set -x
#for num_spans in 5 10;
for lr in 1e-5 5e-6;
do
    for num_spans in 5;
    do
      #With spans datasets
      spans_train=../SHAS-multi/spans/train.maxlen$num_spans/
      spans_dev=../SHAS-multi/spans/dev.maxlen$num_spans/

      output_dir=audio_classifier_"$num_spans"_lr$lr

      rm -r "$output_dir"_models "$output_dir"_extractor

      python3 ../../../src/train/train_transformers_classifier.py  --model_type audio --wav_folder $WAV_FOLDER --model_name facebook/wav2vec2-xls-r-300m \
       --train_files "$train_files" --eval_files "$dev_files" --output_dir_name $output_dir \
       --generate_train_datasets_from_spans_folder $spans_train \
       --generate_eval_datasets_from_spans_folder $spans_dev \
       --learning_rate $lr \
       --per_device_train_batch_size 14 \
       --gradient_accumulation_steps 20 \
       --per_device_eval_batch_size 14 \
       --num_train_epochs 24 \
       --lr_scheduler "cosine" \
       --warmup_ratio 0.15 \
       --fp16 true

     done
done
