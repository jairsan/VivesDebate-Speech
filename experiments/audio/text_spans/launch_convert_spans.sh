
rm -r dev/ test/
mkdir -p dev/ test/
TIMESTAMPS_FOLDER=../../../data_preparation/DATA/BIO_arg_timestamps/

dev="raw/Debate24.raw raw/Debate25.raw raw/Debate26.raw"
test="raw/Debate27.raw raw/Debate28.raw raw/Debate29.raw"

python3 convert_spans_to_segmentation.py "$dev" $TIMESTAMPS_FOLDER dev/segmentation.yaml
python3 convert_spans_to_segmentation.py "$test" $TIMESTAMPS_FOLDER test/segmentation.yaml