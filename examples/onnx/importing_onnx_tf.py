import onnx
import warnings
from onnx_tf.backend import prepare

warnings.filterwarnings('ignore') # Ignore all the warning messages in this tutorial
model = onnx.load('model.onnx') # Load the ONNX file
tf_rep = prepare(model) # Import the ONNX model to Tensorflow

pass