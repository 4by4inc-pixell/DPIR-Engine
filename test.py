"""
.
"""
import os
import unittest
from unittest import TestCase
import dpir_engine as E
import cv2
import numpy as np
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(__file__)
LOCAL_DIR = os.path.join(PROJECT_DIR, "local")
RESOURCE_DIR = os.path.join(PROJECT_DIR, "resource")
ONNX_PATH = os.path.join(LOCAL_DIR, "dpir_fp32.onnx")
DEVICE_ID = 0
INPUT_SAMPLE_IMAGE_PATH = os.path.join(RESOURCE_DIR, "lena_noise.png")
OUTPUT_SAMPLE_IMAGE_PATH = os.path.join(RESOURCE_DIR, "lena_original.png")


class EngineTests(TestCase):
    """
    EngineTests
    """

    def test_512_512_single_inference(self):
        """test_512_512_single_inference"""
        done = False
        input_test_size = (512, 512)
        output_test_size = input_test_size
        try:
            input_image = cv2.resize(
                cv2.imread(INPUT_SAMPLE_IMAGE_PATH, cv2.IMREAD_COLOR), input_test_size
            )
            output_image = cv2.resize(
                cv2.imread(OUTPUT_SAMPLE_IMAGE_PATH, cv2.IMREAD_COLOR), output_test_size
            )
            _h, _w, _c = input_image.shape
            engine = E.DPIREngine(
                onnx_path=ONNX_PATH,
                device_id=DEVICE_ID,
                input_height=input_test_size[1],
                input_width=input_test_size[0],
            )
            engine.set_input_data([input_image])
            engine.convert_data2input()
            engine.move_host2device()
            engine.inference()
            engine.move_device2host()
            engine.convert_output2data()
            result = engine.get_output_data()[0]
            mse = np.square(
                np.subtract(result.astype(np.float32), output_image.astype(np.float32))
            ).mean()
            cv2.imwrite(
                os.path.join(LOCAL_DIR, "test_512_512_single_inference_0.png"), result
            )
            print(f"mse : {mse}")
            done = True
        except Exception as _e:
            print(_e)
        self.assertEqual(done, True)

    def test_512_512_multi_inference(self):
        """test_512_512_multi_inference"""
        number_of_test = 10
        input_test_size = (512, 512)
        output_test_size = input_test_size
        input_image = cv2.resize(
            cv2.imread(INPUT_SAMPLE_IMAGE_PATH, cv2.IMREAD_COLOR), input_test_size
        )
        output_image = cv2.resize(
            cv2.imread(OUTPUT_SAMPLE_IMAGE_PATH, cv2.IMREAD_COLOR), output_test_size
        )
        _h, _w, _c = input_image.shape
        engine = E.DPIREngine(
            onnx_path=ONNX_PATH,
            device_id=DEVICE_ID,
            input_height=input_test_size[1],
            input_width=input_test_size[0],
        )
        done = False
        try:
            for index in tqdm(range(number_of_test)):
                engine.set_input_data([input_image])
                engine.convert_data2input()
                engine.move_host2device()
                engine.inference()
                engine.move_device2host()
                engine.convert_output2data()
                result = engine.get_output_data()[0]
                cv2.imwrite(
                    os.path.join(LOCAL_DIR, f"test_512_512_multi_inference{index}.png"),
                    result,
                )
            done = True
        except Exception as _e:
            print(_e)
        self.assertEqual(done, True)


if __name__ == "__main__":
    unittest.main()
