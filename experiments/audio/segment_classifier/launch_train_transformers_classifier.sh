train_files="../../../data_preparation/DATA/BIO_arg_timestamps/Debate1.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate2.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate3.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate4.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate5.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate6.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate7.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate8.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate9.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate10.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate11.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate12.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate13.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate14.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate15.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate16.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate17.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate18.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate19.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate20.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate21.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate22.txt"
dev_files="../../../data_preparation/DATA/BIO_arg_timestamps/Debate24.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate25.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate26.txt"

for ratio in 0.10 0.50;
do

  output_dir=BERTa
  rm -r "$output_dir"_models "$output_dir"_tokenizer

  python3 ../../../src/train/train_transformers_classifier.py  --model_name PlanTL-GOB-ES/roberta-base-ca \
   --train_files "$train_files" --eval_files "$dev_files" --output_dir_name $output_dir \
   --learning_rate 5e-5 \
   --per_device_train_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --per_device_eval_batch_size 32 \
   --num_train_epochs 10 \
   --lr_scheduler "linear" \
   --warmup_ratio $ratio

done
