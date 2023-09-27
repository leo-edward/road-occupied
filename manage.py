import flask
from flask import request, jsonify, make_response
import numpy as np
import cv2
import base64
import io
import standalone
from PIL import Image

from __init__ import *

app = flask.Flask(__name__)
app.config["DEBUG"] = False

def init_errorhandler(app):
    """
        初始化错误处理
        success = jsonify({'success': True, 'code': 200, 'message': '请求成功'})
    """

    @app.errorhandler(400)
    def bad_request(error):
        return make_response(jsonify({'code': 400, 'message': '参数检验失败', 'data': ''}), 400)

    @app.errorhandler(404)
    def not_found(error):
        return make_response(jsonify({'code': 404, 'message': '请求资源不存在', 'data': ''}), 404)

    @app.errorhandler(405)
    def not_allowed(error):
        return make_response(jsonify({'code': 405, 'message': '错误请求方式', 'data': ''}), 405)

    @app.errorhandler(408)
    def not_allowed(error):
        return make_response(jsonify({'code': 408, 'message': '请求超时', 'data': ''}), 408)

    @app.errorhandler(500)
    def internal_server_error(error):
        return make_response(jsonify({'code': 500, 'message': '服务器错误', 'data': ''}), 500)


@app.route('/inference/road_occupied', methods=['POST'])
def inference():
    if request.method == 'POST':
        data = request.get_json()
        image = data['image']
        image = base64.b64decode(image)
        try:
            r_image, r_boxes, r_classes, r_scores = Model_inference.predict(image)
        except Exception as e:
            logger.error("Error: {}".format(e))
            return jsonify({
                'code': 500,
                'message': 'Error: {}'.format(e),
                'data': {}
                })
        r_image = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
        _, r_image = cv2.imencode('.jpg', r_image)
        r_image = base64.b64encode(r_image)
        r_image = r_image.decode('utf-8')
        # return jsonify({'image': r_image, 'boxes': r_boxes.tolist(), 'classes': r_classes.tolist(), 'scores': r_scores.tolist()})
        return jsonify({
            'code': 200,
            'message': 'image inference success',
            'data': {
                'image': r_image,
                'boxes': r_boxes.tolist(),
                'classes': r_classes.tolist(),
                'scores': r_scores.tolist()
                }
            })
    else:
        logger.error("Request method is not POST")
        return jsonify({
            'code': 400,
            'message': 'Request method is not POST',
            'data': {}
            })

if __name__ == "__main__":
    Model_inference = standalone.Predict()
    init_errorhandler(app)
    app.run(host='0.0.0.0', port=8000)