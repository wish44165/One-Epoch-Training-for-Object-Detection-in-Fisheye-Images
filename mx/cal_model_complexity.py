# https://github.com/gmalivenko/onnx-opcounter
from onnx_opcounter import calculate_params
import onnx

model = onnx.load_model('./best.onnx')
params = calculate_params(model)

print('Total number of params:', params, '\n')

# https://github.com/ThanatosShinji/onnx-tool
import onnx_tool
modelpath = './best.onnx'
onnx_tool.model_profile(modelpath) # pass file name
onnx_tool.model_profile(modelpath, savenode='best.txt') # save profile table to txt file
onnx_tool.model_profile(modelpath, savenode='best.csv') # save profile table to csv file