import pandas as pd
import findspark
findspark.init()
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import sys
path_model = './trained_models/articles'
model = PipelineModel.load('./trained_models/articles')
# tfidf_model = ...... 
spark = SparkSession.builder.appName('myApi')\
                      .config('spark.executor.memory', '4gb')\
                      .config('spark.cores.max', '4').getOrCreate()
label_df = pd.read_csv('category_infor.csv')
print('done1')


def process(S):
    df_input = spark.createDataFrame([(1, S),],['id','all_token'])
    df_output = model.transform(df_input)
    label_id = df_output.select('prediction').rdd.flatMap(lambda x: x).collect()[0]
    label_id = int(label_id) + 100
    label_df_filter = label_df[label_df['id'] ==label_id]
    return label_id, label_df_filter['label_name'].values[0]


if __name__ == "__main__":
    S = sys.argv[1]
    x,y = process(S)
    print(y)