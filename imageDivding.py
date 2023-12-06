# 16*16 으로 나누어 떨어지는 조각 (padding 사용)
# 사진을 0과 1의 폴더를 만들어서 순서대로

import cv2
import numpy as np
import os
import imageCombining

def saveImg(image, block_size, folder):
    height, width, _ = image.shape

    # 이미지 자르기/저장하기
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]

            block_filename = f"block_{y//block_size}_{x//block_size}.png"
            block_filepath = os.path.join(folder, block_filename)
            cv2.imwrite(block_filepath, block)

def padding(image, block_size):
    # 축소
    image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
    height, width = image.shape[:2]

    # pad 크기
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad,
                                      right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    print(f'pad 위아왼오 {top_pad, bottom_pad, left_pad, right_pad}, size {padded_image.shape}')
    return  padded_image

if __name__ == "__main__":
    img = cv2.imread('./images_charm/totoro.jpg')

    block_size = 30
    imgs_folder = "croppedImgs"
    os.makedirs(imgs_folder, exist_ok=True)

    padded_iamge = padding(img, block_size)

    saveImg(padded_iamge, block_size, imgs_folder)

    combined_image = imageCombining.combineImages(imgs_folder, 'combined')

    # 이미지 출력
    cv2.imshow('src', img)
    cv2.imshow('padded', padded_iamge)
    cv2.imshow('restored', combined_image)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
