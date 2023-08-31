import os
import shutil

def is_numeric_folder(name):
    return name.isdigit()

def delete_non_recognized_folders(folder_path):
    contents = os.listdir(folder_path)

    for item in contents:
        item_path = os.path.join(folder_path, item)
        
        if os.path.isdir(item_path):
            if item != "차량번호인식" and not is_numeric_folder(item):
                try:
                    delete_non_recognized_folders(item_path)
                    shutil.rmtree(item_path)  # 비어 있지 않은 디렉터리도 삭제
                    print(f"폴더 '{item}' 삭제됨")
                except Exception as e:
                    print(f"폴더 '{item}' 삭제 중 오류 발생: {e}")
            elif is_numeric_folder(item):
                delete_non_recognized_folders(item_path)

folder_path = "./snapshot"

delete_non_recognized_folders(folder_path)
