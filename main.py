# libraries
from functions import *

import json
import os
from platform import python_version
from flask import Flask, jsonify, make_response
from flask_restx import Api, Resource, reqparse
import warnings
from flask import Response
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
plt.style.use('seaborn-white')


# ### API #####
app = Flask(__name__)
api = Api(app, version='1.11', title='Image Similarity API', description='Geti Solutions - created by Cristian Vergara',
          default="Available Methods", contact='cvergara@geti.cl')


#predict method
similarity_args = reqparse.RequestParser()
similarity_args.add_argument('url_img_cliente', type=str, help="URL de la imagen del Cliente", required=True)
similarity_args.add_argument('url_img_retail', type=str, help="URL de la imagen del Retail", required=True)
similarity_args.add_argument('crop', type=int, default=1, choices=(0, 1), help="Recortar producto:1 , NO Recortar:0",
                             required=True)
similarity_args.add_argument('model', type=str, default='VIT_Model', choices=('VIT_Model', 'ResNet50_v2'),
                             help="Modelo a utilizar para calcular score", required=True)
@api.route("/predict")
class SimilarityScore(Resource):
    @api.expect(similarity_args)
    @api.response(200, "OK")
    @api.response(400, "Bad Request")
    @api.response(404, "Not Found")
    @api.doc(description="Returns the Similarity Score based on the parameters")
    def get(self):
        try:
            args = similarity_args.parse_args()
            url_img_cliente = args.get('url_img_cliente')
            url_img_retail = args.get('url_img_retail')
            #check urls
            for url in [url_img_cliente, url_img_retail]:
                try:
                    check_url(url)
                except Exception as e:
                    response = make_response(jsonify({
                        "error": str(e),
                        "url with error": str(url),
                    }), 404)
                    return response

            crop = args.get('crop')
            model = args.get('model')

            if model == 'VIT_Model':
                score = similarity_score(url_img_cliente, url_img_retail, model_vit, crop=crop)
            if model == 'ResNet50_v2':
                score = similarity_score(url_img_cliente, url_img_retail, model_resnet50_v2_avg, crop=crop)

            response = make_response(jsonify({
                "similarity_score": round(float(score), 4),
                "model": str(model),
                "crop": int(crop),
                "url_cliente": str(url_img_cliente),
                "url_retail": str(url_img_retail),
            }), 200)
            return response

        except Exception as e:
            response = make_response(jsonify({
                "error": str(e)
            }), 400)
            return response


# plot method
plot_args = reqparse.RequestParser()
plot_args.add_argument('url_img_cliente', type=str, help="URL de la imagen del Cliente", required=True)
plot_args.add_argument('url_img_retail', type=str, help="URL de la imagen del Retail", required=True)
@api.route("/plot/threshold")
class PlotThreshold(Resource):
    @api.expect(plot_args)
    @api.response(200, "OK")
    @api.response(400, "Bad Request")
    @api.response(404, "Not Found")
    @api.doc(description="Plot the Original, Thresholded and Cropped Images for each image")
    def get(self):
        try:
            args = plot_args.parse_args()
            url_img_cliente = args.get('url_img_cliente')
            url_img_retail = args.get('url_img_retail')
            #check urls
            for index, url in enumerate([url_img_cliente, url_img_retail]):
                try:
                    check_url(url)
                except Exception as e:
                    response = make_response(jsonify({
                        "error": str(e),
                        "url with error": str(url),
                    }), 404)
                    return response

            fig = thresholding_display(url_img_cliente, url_img_retail)
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            return Response(output.getvalue(), mimetype='image/png')

        except Exception as e:

            response = make_response(jsonify({
                "error": str(e)
            }), 400)
            return response


if __name__ == "__main__":

    # HOST_NAME = 'localhost'
    # PORT = 5000
    app.run(debug=True)
