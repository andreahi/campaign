import os

from flask import Flask, request
from flask_restful import Resource, Api

from PredictionService import predict

os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)
api = Api(app)





class Employees(Resource):

    def get(self):
        birthdata = request.args["birthdata"]
        if request.args["history"]:
            history = request.args["history"].split(",")
            history = map(int, history)
        else:
            history = []

        if request.args["current"]:
            current = request.args["current"].split(",")
            current = map(int, current)
        else:
            current = []
        return predict(birthdata, history, current)


api.add_resource(Employees, '/product')  # Route_1

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=False)