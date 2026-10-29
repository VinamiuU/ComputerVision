import os

def fix_slashes_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if "\\" in content: 
            content_fixed = content.replace("\\", "/")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_fixed)
            print(f"Đã sửa dấu gạch chéo trong tệp: {file_path} ✅")
        else:
            print(f"Không tìm thấy dấu '\' trong tệp: {file_path}. Không cần sửa.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp {file_path} để sửa. ❌")
    except Exception as e:
        print(f"LỖI khi xử lý tệp {file_path}: {e} ❌")


data_path = './SVRDD_TEST' 

test_list_file = os.path.join(data_path, "test.txt")    # Hoặc test_fixed.txt

print("--- Starting ... ---")
# if os.path.exists(train_list_file):
#     fix_slashes_in_file(train_list_file)
# if os.path.exists(val_list_file):
#     fix_slashes_in_file(val_list_file)
if os.path.exists(test_list_file):
    fix_slashes_in_file(test_list_file)
print("---DONE ---")


############### Create environment
# python -m venv CE340
# Set-ExecutionPolicy Unrestricted -Scope Process 
# .\ce340\Scripts\Activate.ps1
