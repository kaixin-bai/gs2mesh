import onnxruntime
import onnxruntime as ort
from pathlib import Path
import sys
sys.path.append('core')

import argparse
import glob
import numpy as np

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import time

"""
推理耗时太久，初步尝试解决，不行就用tensorrt版本的吧
尝试解决：https://blog.csdn.net/weixin_44212848/article/details/137044477
"""

DEVICE = 'cuda'
onnx_model_path = '/data/net/dl_data/ProjectDatasets_bkx/Stereo-related-ckpt-tensorboard/cleardepth_checkpoints/fulldataset_cleardepth_finetune_150000_onnx_tensorrt/clear_depth_full_pretrain.onnx'

def initialize():
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = True
    onnxruntime.set_default_logger_severity(1)
    ort_session = ort.InferenceSession(onnx_model_path, sess_options=session_options,
                                       providers=['CUDAExecutionProvider'])
    print(ort_session.get_providers())
    return ort_session

# 加载 ONNX 模型
ort_session = initialize()
print(ort_session.get_providers())


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

def load_image_for_onnx(imfile):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.float32)
    img = np.ascontiguousarray(img)# 确保图像为 RGB 格式
    original_shape = img.shape[:2]
    img = img.transpose(2, 0, 1)  # 转换为 (C, H, W)
    img = np.expand_dims(img, axis=0)  # 增加 batch 维度 (1, C, H, W)
    return pad_image(img), original_shape  # 填充至 32 倍数

def infer_onnx(ort_session, left_image, right_image):
    inputs = {
        ort_session.get_inputs()[0].name: left_image,
        ort_session.get_inputs()[1].name: right_image
    }

    start_time = time.time()
    print(start_time)
    outputs = ort_session.run(['disp'], inputs)
    end_time = time.time()
    inference_time = end_time - start_time
    return outputs[0], inference_time  # 返回输出和推理时间


def main(args):
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
        image2, _ = load_image_for_onnx(imfile2)
        print(image1.dtype)

        # 使用 ONNX 推理
        flow_up, inference_time = infer_onnx(ort_session, image1, image2)
        print("inference time: ", inference_time)
        print("infer finish")
        print(flow_up.dtype)

        total_time += inference_time

        flow_up = flow_up.squeeze()  # 去掉 (1, 1, H, W) 中的冗余维度，得到 (H, W)

        if len(flow_up.shape) == 3 and flow_up.shape[0] == 1:
            flow_up = flow_up.squeeze(0)

        # 确保数据类型为 uint16，并应用适当的缩放
        flow_up_image = (flow_up * 255).astype(np.uint16)

        flow_up_image = depad_image(flow_up_image, original_shape)

        # 根据中间目录创建文件名
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
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/hdd1/kb/MyProjects/gs2mesh/tools/cleardepth_infer/images/left.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/hdd1/kb/MyProjects/gs2mesh/tools/cleardepth_infer/images/right.png")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/net/dl_data/ProjectDatasets_bkx/transparent_real_test/zed_store/left_*.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/net/dl_data/ProjectDatasets_bkx/transparent_real_test/zed_store/right_*.png")
    parser.add_argument('--output_directory', help="directory to save output",
                        default="/data/hdd1/kb/MyProjects/gs2mesh/tools/cleardepth_infer/output")


    args = parser.parse_args()

    total_time, count = main(args)

    end_time = time.time()
    average_time_per_image = total_time / count
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per image: {average_time_per_image:.2f} seconds")