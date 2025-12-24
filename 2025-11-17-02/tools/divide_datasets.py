import os
import random



dataset_name = "dataset"
file_path = f"{dataset_name}/images"
train_path = f"{dataset_name}/train/images"
test_path = f"{dataset_name}/test/images"


os.mkdir(f"{dataset_name}/train")
os.mkdir(f"{dataset_name}/train/images")
os.mkdir(f"{dataset_name}/train/labels")
os.mkdir(f"{dataset_name}/test")
os.mkdir(f"{dataset_name}/test/images")
os.mkdir(f"{dataset_name}/test/labels")

count = 0
file_list = os.listdir(file_path)
random.shuffle(file_list)
for file in file_list:
    file = f"'{file}'"
    if count < int(len(file_list) * 0.9):
        os.system(f'cp {os.path.join(file_path, file)} {os.path.join(train_path, file)}')
        os.system(f'cp {os.path.join(file_path.replace("images", "labels"), file)} {os.path.join(train_path.replace("images", "labels"), file)}')
    else:
        os.system(f'cp {os.path.join(file_path, file)} {os.path.join(test_path, file)}')
        os.system(f'cp {os.path.join(file_path.replace("images", "labels"), file)} {os.path.join(test_path.replace("images", "labels"), file)}')
    count += 1