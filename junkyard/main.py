import model2_esrgan_super_resolution as esrgan
import model3_easy_ocr as ocr
import model4_roboflow_license_number_extractor as robo

file_name = "02dfc345-68b9-418e-999a-46327e05e5e5.jpg"
before_path = f"./license_plate/{file_name}"
after_path = f"./super_resolution/{file_name}"

def main():    
    # Step 2: Plate Preprocessing
    print()
    esrgan.model_result(before_path, after_path)

    # Step 3: Number Result 2
    print()
    ocr.model_result(after_path)

    # Step 4: Number Result 3    
    print()    
    robo.model_result(after_path)
    print()

if __name__ == "__main__":
    main()