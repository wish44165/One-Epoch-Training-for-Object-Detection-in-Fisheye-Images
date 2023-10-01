import time
import cv2
import numpy as np
import onnxruntime


def inference(simulation_time):
    from memryx import NeuralCompiler
    from memryx import Simulator

########################################
    #model_pre = onnxruntime.InferenceSession("./model_0_yolov7-d6_pre.onnx", providers = ['CPUExecutionProvider'])
    model_mid = onnxruntime.InferenceSession("./model_0_best.onnx", providers = ['CPUExecutionProvider'])
    model_post = onnxruntime.InferenceSession("./model_0_best_post.onnx", providers = ['CPUExecutionProvider'])

    rand_input = np.random.rand(1, 3, 640, 640)
    
    ##################
    
    mid_out = model_mid.run([], {model_mid.get_inputs()[0].name : rand_input.astype(dtype='float32')})
    start = time.perf_counter()    
    #pre_out = model_pre.run([], {model_pre.get_inputs()[0].name : rand_input})   
    accl_start =  time.perf_counter()

    accl_end =  time.perf_counter()    
    post_out = model_post.run([], {model_post.get_inputs()[0].name : mid_out[0].astype(dtype='float32'), model_post.get_inputs()[1].name : mid_out[1].astype(dtype='float32'), model_post.get_inputs()[2].name : mid_out[2].astype(dtype='float32')})
    latency = ((time.perf_counter() - accl_end) + (accl_start - start))*1000  
      
    return (latency + simulation_time)


if __name__ == '__main__':
    simulation_time = 0
    latency_sum = 0
    iteration = 100

    for i in range(iteration):
        latency = inference(simulation_time)
        latency_sum = latency_sum + latency
        print(f"Iteration: {i},  Inference time: {(latency)} ms")

    print(f"total Iteration: {iteration},  Average Inference Time: {(latency_sum/iteration)} ms")

