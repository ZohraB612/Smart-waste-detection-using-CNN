import tensorflow_data_validation as tfdv
import apache_beam as beam
from tensorflow_metadata.proto.v0 import statistics_pb2

DATA_LOCATION = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\records\combined_records\validationhundred_per" \
                r".record "
OUTPUT_LOCATION = r"C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\validation"

with beam.Pipeline() as p:
    _ = (p
         # 1. Read out the examples from input files.
         | 'ReadData' >> beam.io.ReadFromTFRecord(file_pattern=DATA_LOCATION)
         # 2. Convert examples to Arrow RecordBatches, which represent example
         #    batches.
         | 'DecodeData' >> tf_example_decoder.DecodeTFExample()
         # 3. Invoke TFDV `GenerateStatistics` API to compute the data statistics.
         | 'GenerateStatistics' >> tfdv.GenerateStatistics()
         # 4. Materialize the generated data statistics.
         | 'WriteStatsOutput' >> WriteStatisticsToTFRecord(OUTPUT_LOCATION))
