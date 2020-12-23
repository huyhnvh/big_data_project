import pandas as pd
import pickle
# import findspark
# findspark.init()
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession, Row
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
sc = spark.sparkContext
sqlContext = SQLContext(sc)
label_df = pd.read_csv('category_infor.csv')
print('done1')
l = []


def process(list_text, text_id):
    S = list_text
#     for i in list_text:
#         S.append(ViTokenizer.tokenize(i))
#     df_input = spark.createDataFrame([(1, S),],['id','all_token'])
#     df_output = model.transform(df_input)
#     label_id = df_output.select('prediction').rdd.flatMap(lambda x: x).collect()[0]
#     label_id = int(label_id) + 100
    #logsklearn
    x = cv_model.transform(S)
    y = log_model.predict(x)
#     label_ids = []
#     label_names = []
    for i in range(len(y)):
#         label_ids.append(i)
        label_df_filter = label_df[label_df['id'] == y[i]]
        label_name = label_df_filter['name'].values[0]
        l.append((text_id[i], y[i], label_name))
    rdd = sc.parallelize(l)
    people = rdd.map(lambda x: Row(id_new =x[0], cate_id = int(x[1]), cate_name = x[2]))
    df = sqlContext.createDataFrame(people)
    print(df.show())
#     df.write.csv("hdfs://namenode:9000/result/cate_table.csv")
    return len(l) 


if __name__ == "__main__":
    S = sys.argv[1]
    x,y = process(S)
    print(y)