import pandas as pd
import pickle
# import findspark
# findspark.init()
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyvi import ViTokenizer
import sys
cv_model = pickle.load(open('./trained_models/tfidf_model.sav', 'rb'))
log_model = pickle.load(open('./trained_models/log_model.sav', 'rb'))
path_model = './trained_models/articles'
model = PipelineModel.load('./trained_models/articles')
# tfidf_model = ...... 
spark = SparkSession.builder.appName('myApi')\
                      .config('spark.executor.memory', '4gb')\
                      .config('spark.cores.max', '4').getOrCreate()
label_df = pd.read_csv('category_infor.csv')
print('done1')


def process(S):
    S = ViTokenizer.tokenize(S)
#     df_input = spark.createDataFrame([(1, S),],['id','all_token'])
#     df_output = model.transform(df_input)
#     label_id = df_output.select('prediction').rdd.flatMap(lambda x: x).collect()[0]
#     label_id = int(label_id) + 100
    #logsklearn
    x = cv_model.transform([S])
    y = log_model.predict(x)
    label_id = y[0]
    label_df_filter = label_df[label_df['id'] ==label_id]
    return label_id, label_df_filter['name'].values[0]


if __name__ == "__main__":
    S = sys.argv[1]
    x,y = process(S)
    print(y)