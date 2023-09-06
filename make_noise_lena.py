import cv2
import numpy as np


# def make_noise(std, gray):
#     height, width = gray.shape
#     img_noise = np.zeros((height, width), dtype=np.float)
#     for i in range(height):
#         for a in range(width):
#             noise = np.random.normal()  # 랜덤함수를 이용하여 노이즈 적용
#             set_noise = std * noise
#             img_noise[i][a] = gray[i][a] + set_noise
#     return img_noise


def main():
    power = 20
    image = cv2.imread("./resource/lena_original.png").astype(np.float32)
    noise_vector = np.random.standard_normal(image.shape).astype(np.float32) * power
    image = np.clip(image + noise_vector, 0, 255).astype(np.uint8)
    # for i in range(3):
    #     image[:, :, i] = make_noise(10, image[:, :, i])

    cv2.imwrite("./resource/lena_noise.png", image)


if __name__ == "__main__":
    main()
