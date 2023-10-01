# https://pytorch.org/docs/stable/onnx.html
import onnx

# Load the ONNX model
model = onnx.load("./best.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))


# https://github.com/gmalivenko/onnx-opcounter
from onnx_opcounter import calculate_params
import onnx

#model = onnx.load_model('./best.onnx')
params = calculate_params(model)

print('Number of params:', params)
