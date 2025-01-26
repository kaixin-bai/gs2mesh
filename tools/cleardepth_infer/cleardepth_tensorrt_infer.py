import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

import pycuda.driver as cuda
import pycuda.autoinit
from typing import Dict, OrderedDict, List, Union
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
from PIL import Image
import time
from matplotlib import pyplot as plt
import argparse


def load_engine(path):
    print(f'Loading engine from file {path}')
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print('Completed loading engine')
    return engine


def get_input_tensor_names(engine: trt.ICudaEngine) -> List[str]:
    input_tensor_names = []
    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            input_tensor_names.append(binding)
    return input_tensor_names


def get_output_tensor_names(engine: trt.ICudaEngine) -> List[str]:
    output_tensor_names = []
    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
            output_tensor_names.append(binding)
    return output_tensor_names


def load_image_for_onnx(imfile):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.float32)
    img = np.ascontiguousarray(img)# 确保图像为 RGB 格式
    original_shape = img.shape[:2]
    img = img.transpose(2, 0, 1)  # 转换为 (C, H, W)
    img = np.expand_dims(img, axis=0)  # 增加 batch 维度 (1, C, H, W)
    return pad_image(img), original_shape  # 填充至 32 倍数


def pad_image(image, divis_by=32):
    _, _, h, w = image.shape
    pad_h = (divis_by - h % divis_by) % divis_by
    pad_w = (divis_by - w % divis_by) % divis_by
    padded_image = np.pad(image, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    return padded_image


def depad_image(padded_image, original_shape):
    h, w = original_shape
    # Remove padding
    depadded_image = padded_image[:h, :w]
    return depadded_image


class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        # print("[MyOutputAllocator::__init__]")
        super().__init__()
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name: str, memory: int, size: int, alignment: int) -> int:
        print("[MyOutputAllocator::reallocate_output] TensorName=%s, Memory=%s, Size=%d, Alignment=%d" % (tensor_name, memory, size, alignment))
        if tensor_name in self.buffers:
            del self.buffers[tensor_name]

        address = cuda.mem_alloc(size)
        self.buffers[tensor_name] = address
        return int(address)

    def notify_shape(self, tensor_name: str, shape: trt.Dims):
        print('asdsddf')
        print("[MyOutputAllocator::notify_shape] TensorName=%s, Shape=%s" % (tensor_name, shape))
        self.shapes[tensor_name] = tuple(shape)


class ProcessorV3:
    def __init__(self, engine: trt.ICudaEngine):
        # # 选择第一个可用的 GPU 设备
        # device = cuda.Device(0)
        # # 将所选设备设置为当前活动的上下文
        # self.cuda_context = device.make_context()
        # self.cuda_context.push()

        self.engine = engine
        self.output_allocator = OutputAllocator()
        # create execution context
        self.context = engine.create_execution_context()
        # get input and output tensor names
        self.input_tensor_names = get_input_tensor_names(engine)
        self.output_tensor_names = get_output_tensor_names(engine)
        # create stream
        self.stream = cuda.Stream()
        # Create a CUDA events
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()

    # def __del__(self):
    #     self.cuda_context.pop()

    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)

    def infer(self, inputs: Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray]) -> OrderedDict[str, np.ndarray]:
        """
        inference process:
        1. create execution context
        2. set input shapes
        3. allocate memory
        4. copy input data to device
        5. run inference on device
        6. copy output data to host and reshape
        """
        # set input shapes, the output shapes are inferred automatically

        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        if isinstance(inputs, dict):
            inputs = [inp if name in self.input_tensor_names else None for (name, inp) in inputs.items()]
        if isinstance(inputs, list):
            for name, arr in zip(self.input_tensor_names, inputs):
                self.context.set_input_shape(name, arr.shape)

        buffers_host = []
        buffers_device = []
        # copy input data to device
        for name, arr in zip(self.input_tensor_names, inputs):
            host = cuda.pagelocked_empty(arr.shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            device = cuda.mem_alloc(arr.nbytes)

            host[:] = arr
            cuda.memcpy_htod_async(device, host, self.stream)
            buffers_host.append(host)
            buffers_device.append(device)
        # set input tensor address
        for name, buffer in zip(self.input_tensor_names, buffers_device):
            self.context.set_tensor_address(name, int(buffer))

        # set output tensor allocator
        for name in self.output_tensor_names:
            # 设置输出张量地址为 nullptr，以启用自定义分配器
            self.context.set_tensor_address(name, 0)
            # 设置输出分配器
            self.context.set_output_allocator(name, self.output_allocator)
        # The do_inference function will return a list of outputs

        # Record the start event
        self.start_event.record(self.stream)
        # Run inference.
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # Record the end event
        self.end_event.record(self.stream)

        # self.memory.copy_to_host()

        output_buffers = OrderedDict()
        for name in self.output_tensor_names:
            print('!!!', self.output_allocator.shapes)
            arr = cuda.pagelocked_empty(self.output_allocator.shapes[name],
                                        dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            cuda.memcpy_dtoh_async(arr, self.output_allocator.buffers[name], stream=self.stream)
            output_buffers[name] = arr

        # Synchronize the stream
        self.stream.synchronize()

        return output_buffers


def main(args):
    engine = load_engine(args.tensorrt_path)

    for binding in engine:
        # name = engine.get_tensor_name(binding)
        dtype = engine.get_tensor_dtype(binding)
        print(f"Binding dtype {dtype}")

    processor = ProcessorV3(engine)

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

    total_time = 0
    count = 0

    for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        count += 1
        image1, original_shape = load_image_for_onnx(imfile1)

        print('qwe', image1.shape)
        start_time = time.time()
        image2, _ = load_image_for_onnx(imfile2)
        end_time = time.time()
        print(image1.dtype)
        total_time = end_time - start_time
        print(total_time)

        # 使用 ONNX 推理
        outputs = processor.infer([image1, image2])
        flow_up = outputs['disp']

        print(flow_up.dtype)

        # total_time += inference_time

        flow_up = flow_up.squeeze()  # 去掉 (1, 1, H, W) 中的冗余维度，得到 (H, W)

        if len(flow_up.shape) == 3 and flow_up.shape[0] == 1:
            flow_up = flow_up.squeeze(0)

        # 确保数据类型为 uint16，并应用适当的缩放
        flow_up_image = flow_up

        flow_up_image = depad_image(flow_up_image, original_shape)

        # 根据中间目录创建文件名
        # if args.middleburry:
        #     file_stem = imfile1.split('/')[-2][:-4]
        # else:
        file_stem = imfile1.split('/')[-1][:-4]

        # 使用 plt.imsave 保存图像，使用 jet 颜色映射
        output_path = output_directory / f"{file_stem}_output.png"
        plt.imsave(output_path, -flow_up_image, cmap='jet')

        # 如果需要同时保存为 numpy 格式
        if args.save_numpy:
            np.save(output_directory / f"{file_stem}.npy", flow_up_image)
    return total_time, count


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='/home/yiwen.liu/projects/RAFT-Stereo/depthanything_large/400000_raft-stereo.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/hdd1/kb/MyProjects/gs2mesh/tools/cleardepth_infer/images/left.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/hdd1/kb/MyProjects/gs2mesh/tools/cleardepth_infer/images/right.png")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/net/dl_data/ProjectDatasets_bkx/transparent_real_test/zed_store/left_*.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/net/dl_data/ProjectDatasets_bkx/transparent_real_test/zed_store/right_*.png")
    parser.add_argument('--output_directory', help="directory to save output",
                        default="/data/hdd1/kb/MyProjects/gs2mesh/tools/cleardepth_infer/output")
    parser.add_argument('--tensorrt_path', help="directory to save output",
                        default="/data/hdd1/kb/MyProjects/gs2mesh/tools/cleardepth_infer/clear_depth_full_pretrain.engine")

    args = parser.parse_args()

    total_time, count = main(args)

    end_time = time.time()
    average_time_per_image = total_time / count
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per image: {average_time_per_image:.2f} seconds")