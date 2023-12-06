# 16*16 으로 나누어 떨어지는 조각 합치고 패딩 제거
import cv2
import numpy as np
import os
import re

def stackBlocks(blocks, rows, cols, folder):
    combined_image = np.vstack([np.hstack(blocks[i*cols:(i+1)*cols]) for i in range(rows)])
    return combined_image

def combineImages(block_folder, output):
    block_lists = os.listdir(block_folder)
    sorted_block_lists = sorted(block_lists, key=lambda x: tuple(map(int, re.findall(r'\d+', x))))
    blocks = [cv2.imread(os.path.join(block_folder, filename)) for filename in sorted_block_lists]

    # rows, cols 결정
    numbers_list = [re.findall(r'\d+', file_name) for file_name in sorted_block_lists]
    rows = max(int(front) for front, _ in numbers_list) + 1
    cols = max(int(back) for _, back in numbers_list) + 1

    # image 저장
    combined_img = stackBlocks(blocks, rows, cols, output)
    cropped_img = cropPadding(combined_img)
    cv2.imwrite(os.path.join(output,'result.jpg'), combined_img)

    return cropped_img

def cropPadding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    coordinates = np.column_stack(np.where(gray>0))
    x,y,w,h = cv2.boundingRect(coordinates)

    cropped_image = image[x:x+w, y:y+h]

    return  cropped_image

if __name__ == "__main__":
    # img = cv2.imread('./images_charm/totoro.jpg')
    #
    # block_folder = './croppedImgs'
    # output = 'combined'
    # os.makedirs(output, exist_ok=True)
    # img = blockStack(block_folder, output)
    #
    # cv2.imshow('src', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    pass