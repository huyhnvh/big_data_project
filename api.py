# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import pandas as pd
import classification
import logging

LOGGER = logging.getLogger(__name__)
app = Flask(__name__)
 
@app.route('/TextClassification', methods=['POST'])
def process_classify():
    try:
        if "text" not in request.form:
            LOGGER.warn("Missing text field in form-data")
            return jsonify(dict(status=0, code = 400, msg = "input miss text field"))
        content = request.form["text"]
        
    except Exception as e:
        LOGGER.exception("Have error while get input classify text for news")
        return jsonify(dict(status=0, code = 400, msg = "format input error!"))
    try:    
        if content == '':
            rs = {
                "status": 1,
                "code": 200,
                "data": {
                    'text': '',
                    "cate_id": '150',
                    "cate_name": 'Text khác',
                    }
            }
            print(rs)
            return jsonify(rs)

        if content is not None:
            id_label, name_label = classification.process(content)
            print(str(id_label)+','+name_label)
            rs = {
                "status": 1,
                "code": 200,
                "data": {
                    'text': content,
                    "cate_id": str(id_label),
                    "cate_name": name_label,
                    }
            }
        else:
            rs = {
                "status": 1,
                "code": 200,
                "data": {
                    'text': '',
                    "cate_id": '',
                    "cate_name": '',
                    }
            }
        return jsonify(rs)
    except Exception as e:
        LOGGER.exception("Have error while classify text for news")
        return jsonify(dict(status=0, code = 500, msg = "error!"))

@app.route('/list_cate', methods=['GET'])
def process_list():
    df = pd.read_csv('category_infor.csv')
    id_labels = df['id'].values
    names = df['name'].values
    datas = []
    for i in range(len(id_labels)):
        datas.append({"cate_id": str(id_labels[i]),
                    "cate_name": names[i],
                    })
    rs = {"status": 1,
            "code": 200,
            "data": datas }
    return jsonify(rs)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port= '8000',debug=True)
