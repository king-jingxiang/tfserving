# Copyright 2019 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import requests
from tfserver import TFModel
import json

model_dir = os.path.join(os.path.dirname(__file__), "example_model/inception_v3")


def test_model():
    server = TFModel("inception_v3", model_dir, "input", "InceptionV3/Predictions/Reshape_1")
    server.load()

    input_tensor = np.random.normal(-1, 1, [299, 299, 3])

    request = {"instances": input_tensor.tolist()}
    response = server.predict(request)
    print(response["predictions"])


def rest_test_model():
    url = "http://127.0.0.1:8080/v1/models/mnist:predict"
    input_tensor = np.random.normal(-1, 1, [784])
    request = {"instances": input_tensor.tolist(), "signature_name": "predict_images"}
    response = requests.post(url, data=json.dumps(request))
    print(response.json())


if __name__ == '__main__':
    # test_model()
    # python __main__.py --model_dir=example_model/inception_v3 --model_name=inception_v3  --input_name=input --output_name=InceptionV3/Predictions/Reshape_1
    rest_test_model()
