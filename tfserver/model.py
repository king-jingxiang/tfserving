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

from typing import Dict

import kfserving
import numpy as np
import saved_model_metadata as tools
import tensorflow as tf

TENSORFLOW_MODEL_FILE = "saved_model.pb"


class TFModel(kfserving.KFModel):
    def __init__(self, name: str, saved_model_dir: str):
        super().__init__(name)
        self.name = name
        self.saved_model_dir = saved_model_dir
        self.metadata = tools.getMetadata(saved_model_dir)
        self.graph = None
        self.ready = False
        self._batch_size = 1

    # load saved_model file
    def load(self):
        self.graph = tf.Graph()
        with tf.Session(graph=self.graph) as sess:
            tf.saved_model.loader.load(sess, [self.metadata.tag_set], self.saved_model_dir)
            self.ready = True

    # predict with signature
    def predict(self, request: Dict) -> Dict:
        inputs = []
        try:
            #  转成tensorflow tensor
            inputs = np.array(request["instances"])
            input_shape = tuple(tools.flatten([(1), inputs.shape]))
            inputs = inputs.reshape(input_shape)
        except Exception as e:
            raise Exception("Failed to initialize Tensorflow Tensor from inputs: %s, %s" % (e, inputs))
        try:
            signature_name = request['signature_name']
            signature_info = self.metadata.signatute_info[signature_name]
            # TODO 暂时仅支持单个input/output layer
            input_operator = self.graph.get_tensor_by_name(signature_info.input_tensor[0].name)
            output_operator = self.graph.get_tensor_by_name(signature_info.output_tensor[0].name)


        except Exception as e:
            raise Exception("Failed to get signature %s" % e)
        try:
            with tf.Session(graph=self.graph) as sess:
                results = sess.run(output_operator, feed_dict={input_operator: inputs})
                return {"predictions": results.tolist()}
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
