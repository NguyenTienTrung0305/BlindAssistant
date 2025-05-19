from torchvision import transforms
from util import *
from PIL import Image, ImageDraw, ImageFont
from model import *
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'Project/BlindAssistant/checkpoint_ssd300_v1.pth.tar'
checkpoint = torch.load(checkpoint,  map_location=torch.device('cpu'))
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)

model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def hex_to_rgb(hex_color):
    # Loại bỏ dấu '#' 
    hex_color = hex_color.lstrip('#')
    # Chuyển đổi chuỗi hex thành giá trị RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def detect_frame(frame, min_score, max_overlap, top_k, suppress=None):
    """_summary_

    Args:
        frame : numpy array
        original_image: PIL Image

    Returns:
        _type_: _description_
    """
    
    # chuyển đổi ảnh từ định dạng OpenCV (BGR) sang RGB, vì PIL yêu cầu ảnh ở dạng RGB
    # Image.fromarray() chuyển đổi ảnh từ numpy array sang PIL Image
    original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Transform
    # unsqueeze(0) => tạo batch = 1
    image = normalize(to_tensor(resize(original_image))).unsqueeze(0).to(device)
    
    # tập hợp các bounding box và scores được dự đoán từ features của image
    predicted_locs, predicted_scores = model(image)

    # kết hợp với nms để loại bỏ bớt các bounding box và scores
    det_boxes, det_labels, det_scores = model.detect_object(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Tọa độ
    # [x_min, y_min, x_max, y_max] => tỉ lệ % so với kích thước của ảnh
    # x_min = 0.1 => khoảng cách từ cạnh trái bounding box đến cạnh trái ảnh là 0.1*width của ảnh
    # [x_min, y_min, x_max, y_max] => [trái, trên, phải, dưới]
    det_boxes = det_boxes[0].to('cpu')
    
    # labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    
    if det_labels == ['background']:
        return frame

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    print(frame.shape[1] , " ", frame.shape[0])
    for i in range(det_boxes.size(0)):
        if suppress and det_labels[i] in suppress:
            continue

        box_location = det_boxes[i].tolist()
        label = det_labels[i]
        color = label_color_map[label]
        if isinstance(color, str):  # Nếu màu là chuỗi hex, chuyển đổi thành tuple RGB
            color = hex_to_rgb(color)
        color = (int(color[0]), int(color[1]), int(color[2]))  # Chuyển sang BGR

        # Vẽ khung và nhãn lên khung hình
        cv2.rectangle(frame, (int(box_location[0]), int(box_location[1])),
                      (int(box_location[2]), int(box_location[3])), color, 2)
        cv2.putText(frame, label.upper(), (int(box_location[0]), int(box_location[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return frame


# if __name__ == '__main__':
#     img_path = 'D:/Datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000009.jpg'
#     original_image = Image.open(img_path, mode='r')
#     original_image = original_image.convert('RGB')
#     detect(original_image, min_score=0.2, max_overlap=0.7, top_k=200).show()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    frame = detect_frame(frame, 0.15, 0.3, top_k=200)
    cv2.imshow('Real-time Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import torch
# from torchvision import transforms
# from util import *
# from PIL import Image, ImageDraw, ImageFont
# from model import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model checkpoint
# checkpoint = 'Project/BlindAssistant/checkpoint_ssd300.pth.tar'
# checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
# start_epoch = checkpoint['epoch'] + 1
# print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
# model = checkpoint['model']
# model = model.to(device)
# model.eval()

# # Transforms
# resize = transforms.Resize((300, 300))
# to_tensor = transforms.ToTensor()
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])


# def detect(original_image, min_score, max_overlap, top_k, suppress=None):
#     """
#     Detect objects in an image with a trained SSD300, and visualize the results.

#     :param original_image: image, a PIL Image
#     :param min_score: minimum threshold for a detected box to be considered a match for a certain class
#     :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
#     :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
#     :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
#     :return: annotated image, a PIL Image
#     """

#     # Transform
#     image = normalize(to_tensor(resize(original_image)))

#     # Move to default device
#     image = image.to(device)

#     # Forward prop.
#     predicted_locs, predicted_scores = model(image.unsqueeze(0))

#     # Detect objects in SSD output
#     det_boxes, det_labels, det_scores = model.detect_object(predicted_locs, predicted_scores, min_score=min_score,
#                                                              max_overlap=max_overlap, top_k=top_k)

#     # Move detections to the CPU
#     det_boxes = det_boxes[0].to('cpu')

#     # Transform to original image dimensions
#     original_dims = torch.FloatTensor(
#         [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
#     det_boxes = det_boxes * original_dims

#     # Decode class integer labels
#     det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

#     # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
#     if det_labels == ['background']:
#         # Just return original image
#         return original_image

#     # Annotate
#     annotated_image = original_image
#     draw = ImageDraw.Draw(annotated_image)
#     font = ImageFont.truetype("./calibril.ttf", 15)

#     # Suppress specific classes, if needed
#     for i in range(det_boxes.size(0)):
#         if suppress is not None:
#             if det_labels[i] in suppress:
#                 continue

#         # Boxes
#         box_location = det_boxes[i].tolist()
#         draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
#         draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
#             det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
#         # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
#         #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
#         # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
#         #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

#         # Text
#         text_bbox = font.getbbox(det_labels[i].upper())
#         print(det_labels[i])
#         text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
#         text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
#         textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
#                             box_location[1]]
#         draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
#         draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
#                   font=font)
#     del draw

#     return annotated_image


# if __name__ == '__main__':
#     img_path = 'D:/Datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
#     original_image = Image.open(img_path, mode='r')
#     original_image = original_image.convert('RGB')
#     detect(original_image, min_score=0.2, max_overlap=0.3, top_k=200).show()