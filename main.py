import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tempfile

import cv2
import numpy as np
import streamlit as st
import supervision as sv
from PIL import Image
from ultralytics import YOLO

bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
bounding_box_annotator_plate = sv.BoundingBoxAnnotator(thickness=1)
annotator = sv.BoxAnnotator()

model_plate_ = YOLO("nhandienbanso.pt")
model_read_char = YOLO("nhandienchuso.pt")
label_annotator = sv.LabelAnnotator(text_scale=0.3)
char_annotator = sv.LabelAnnotator(text_scale=0.7)
# Thiết lập cho trang web
st.set_page_config(
    page_title="UPLOAD PICTURE TO TEST",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto",
)
# Đường link của ảnh trên mạng
background_image_url = "https://cdn.sforum.vn/sforum/wp-content/uploads/2023/07/hinh-nen-ai-57.jpg"

# Thiết lập CSS để thay đổi hình ảnh làm nền
background_style = f"""
    <style>
        .stApp {{
            background-image: url('{background_image_url}');
            background-size: cover;
        }}
    </style>
"""

# Sử dụng markdown để thêm CSS vào trang web
st.markdown(background_style, unsafe_allow_html=True)
# Sử dụng markdown để định dạng HTML và CSS
st.markdown(
    """
    <style>
        .custom-title {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='custom-title'>UPLOAD PICTURE TO TEST</h1>", unsafe_allow_html=True)
image_file = st.file_uploader("Upload a picture", type=["jpg", "png", "jpeg"])
image_placeholder = st.empty()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # File uploader
    if image_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(image_file.read())
        Ivehicle = cv2.imread(tfile.name)
        vehicle= model_plate_(Ivehicle,conf=0.6) # detect ra bản so xe
        image = Image.fromarray(Ivehicle)
        detections = sv.Detections.from_ultralytics(vehicle[0]) #ddojcc kết quả từ  model_plate_

        orig_area= detections.area # lay area của các bản số

        detections = detections[(detections.area > 1000)]  # nếu anh quá nhỏ sẽ bác bỏ

        if len(detections) < 1:
            if len(orig_area) > 0:
                st.markdown("<p style='color:white; font-size:50px;'>TOO SMALL</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:white; font-size:50px;'>NO DETECTION</p>", unsafe_allow_html=True)
            st.image(image)
        else:
            annotated_image = bounding_box_annotator.annotate(scene=Ivehicle, detections=detections) # ve bounding box len anh
            labels = [f" {model_read_char.model.names[class_id]}" for class_id in detections.class_id] # chu so

            box_list_plate = [] #

            crop_img_list_plate = []
            # lay bounding box cho tung bang so
            for box in detections.xyxy:
                box_list_plate.append(tuple(np.array(box, np.int64)))

            # dua vao bounding cat ra anh cac bang so
            for crop_region in box_list_plate:
                crop_img = image.crop(crop_region)
                crop_img = crop_img.convert("RGB")
                crop_img = np.array(crop_img)

                crop_img_list_plate.append(crop_img)

            for i, image_crop in enumerate(crop_img_list_plate):
                result_char = model_read_char(image_crop)
                char_detections = sv.Detections.from_ultralytics(result_char[0])

                if (len(char_detections)>=8):
                    st.image(result_char[0].plot(), width=600)
                    char_labels = [f"{model_read_char.model.names[class_id]}" for class_id in
                                   char_detections.class_id]

                    # get box coordinates in (left, top, right, bottom) format
                    boxes = result_char[0].boxes.xyxy
                    list__class = []
                    for class_name, box in zip(char_labels, boxes):
                        list__class.append({"class": class_name, "box": box})

                    sorted_list_class = sorted(list__class, key=lambda x: (x['box'][1]))

                    result_string = ""
                    current_row = []
                    rows = []
                    threshold_y = 0.5  # Điều chỉnh giá trị ngưỡng tùy theo yêu cầu

                    # Lặp qua từng bounding box
                    for box_info in sorted_list_class:
                        box = box_info['box']


                        # Kiểm tra xem hộp hiện tại có thuộc cùng một dòng không
                        if not current_row or abs((box[1]-current_row[-1][1])/(box[1]+current_row[-1][1])) < threshold_y :

                            current_row.append(box)

                        else:

                            rows.append(current_row)
                            current_row = [box]

                    # Thêm dòng cuối cùng vào danh sách các dòng
                    if current_row:
                        rows.append(current_row)

                    # In số lượng hộp trên từng dòng
                    row1 = sorted_list_class[:len(rows[0])]
                    row1 = sorted(row1, key=lambda x: (x['box'][0]))


                    for item in row1:
                        class_name = item['class']
                        result_string += class_name

                    row2 = sorted_list_class[len(rows[0]):]
                    row2 = sorted(row2, key=lambda x: (x['box'][0]))


                    for item in row2:
                        class_name = item['class']
                        result_string += class_name
                    labels[i] = result_string
                    print(f"Bien so xe {i + 1} la : {result_string}")
                else:
                    print("Bien so da bi che, hay lam ro bien so")
            a = char_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            st.image(a)