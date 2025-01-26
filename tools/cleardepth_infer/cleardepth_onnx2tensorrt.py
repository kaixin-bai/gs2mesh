from pathlib import Path

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

from collections import OrderedDict
from typing import Dict, OrderedDict, List, Union
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def GiB(val):
    return val * 1 << 30


# This function builds an engine from a onnx model.
def build_engine(onnx_file_path, precision='fp16', dynamic_shapes=None):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""

    EXPLICIT_BATCH_FLAG = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH_FLAG)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER) # 使用 trt.OnnxParser 解析 ONNX 模型文件并构建 TensorRT 网络。它适合已经存在的 ONNX 文件，不需要手动添加 TensorRT 层

    # Parse model file
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loading ONNX file from path {onnx_file_path}...')
    with open(onnx_file_path, 'rb') as model:
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed parsing of ONNX file')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Input number: {network.num_inputs}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Output number: {network.num_outputs}')

    if dynamic_shapes is not None:
        # set optimization profile for dynamic shape
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input = network.get_input(i)
            min_shape = dynamic_shapes['min_shape']
            opt_shape = dynamic_shapes['opt_shape']
            max_shape = dynamic_shapes['max_shape'] #使用 dynamic_shapes 参数定义动态形状的 min_shape、opt_shape 和 max_shape，并使用 builder.create_optimization_profile() 添加优化配置。动态形状支持可以帮助优化内存使用，提升推理效率
            profile.set_shape(input.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # We set the builder batch size to be the same as the calibrator's, as we use the same batches
    # during inference. Note that this is not required in general, and inference batch size is
    # independent of calibration batch size.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(1))  # 1G

    if precision == 'fp32':
        pass
    elif precision == 'fp16':
        print('111111')
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
    else:
        raise ValueError('precision must be one of fp32, fp16, or int8')

    # Build engine.
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building an engine from file {onnx_file_path}; this may take a while...')
    serialized_engine = builder.build_serialized_network(network, config)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed creating Engine')
    return serialized_engine

def save_engine(engine, path):
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Saving engine to file {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(engine)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed saving engine')

def load_engine(path):
    print(f'Loading engine from file {path}')
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print('Completed loading engine')
    return engine




if __name__ == '__main__':
    ds = {
        "min_shape":(1, 3, 100, 200),
        "opt_shape":(1, 3, 720, 1280),
        "max_shape":(1, 3, 800, 1300),
    }
    engine = build_engine("/data/net/dl_data/ProjectDatasets_bkx/Stereo-related-ckpt-tensorboard/cleardepth_checkpoints/fulldataset_cleardepth_finetune_150000_onnx_tensorrt/clear_depth_full_pretrain.onnx", dynamic_shapes=ds)
    save_engine(engine, "clear_depth_full_pretrain.engine")
