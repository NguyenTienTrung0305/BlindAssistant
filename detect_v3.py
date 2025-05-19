# import cv2
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# model_path = 'Project/BlindAssistant/ssd_mobilenet1.tflite'

# # Load the labels into a list
# classes = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird",
#     "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
#     "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
#     "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
#     "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#     "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop",
#     "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
#     "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"
# ]


# # Define a list of colors for visualization
# COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)


# def preprocess_image(image, input_size):
#     """Preprocess the input image to feed to the TFLite model"""
#     img = tf.image.convert_image_dtype(image, tf.uint8)
#     resized_img = tf.image.resize(img, (300, 300))  # Resize to (320, 320) to match the model
#     return resized_img[tf.newaxis, :]


# def set_input_tensor(interpreter, image):
#     """Set the input tensor for the model."""
#     tensor_index = interpreter.get_input_details()[0]['index']
#     input_tensor = interpreter.tensor(tensor_index)()[0]
#     input_tensor[:, :] = image


# def get_output_tensor(interpreter, index):
#     """Get the output tensor from the model."""
#     output_details = interpreter.get_output_details()[index]
#     tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
#     return tensor


# def detect_objects(interpreter, frame, threshold):
#     """Detect objects in the frame using the TensorFlow Lite model"""
#     preprocessed_image = preprocess_image(frame, (300, 300))
#     set_input_tensor(interpreter, preprocessed_image)
#     interpreter.invoke()

#     boxes = get_output_tensor(interpreter, 0)
#     classes = get_output_tensor(interpreter, 1)
#     scores = get_output_tensor(interpreter, 2)
#     count = int(get_output_tensor(interpreter, 3))

#     results = []
#     for i in range(count):
#         if scores[i] >= threshold:
#             result = {
#                 'bounding_box': boxes[i],
#                 'class_id': classes[i],
#                 'score': scores[i]
#             }
#             results.append(result)
#     return results


# def run_real_time_detection(threshold=0.5):
#     """Run real-time object detection using webcam"""
#     cap = cv2.VideoCapture(0)

#     # Load the TFLite model
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Run object detection on the current frame
#         results = detect_objects(interpreter, frame, threshold=threshold)

#         # Draw the detection results on the frame
#         for obj in results:
#             ymin, xmin, ymax, xmax = obj['bounding_box']
#             xmin = int(xmin * frame.shape[1])
#             xmax = int(xmax * frame.shape[1])
#             ymin = int(ymin * frame.shape[0])
#             ymax = int(ymax * frame.shape[0])

#             class_id = int(obj['class_id'])
#             color = [int(c) for c in COLORS[class_id]]
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

#             y = ymin - 15 if ymin - 15 > 15 else ymin + 15
#             label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
#             cv2.putText(frame, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         # Display the frame with detection results
#         cv2.imshow("Object Detection", frame)

#         # Break the loop on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# # Run real-time object detection
# run_real_time_detection(threshold=0.6)



# import cv2
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# model_path = 'Project/BlindAssistant/ssd_mobilenet1.tflite'

# # Load the labels into a list
# classes = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird",
#     "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
#     "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
#     "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
#     "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#     "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop",
#     "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
#     "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"
# ]

# # Define a list of colors for visualization
# COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

# def preprocess_image(image, input_size):
#     """Preprocess the input image to feed to the TFLite model"""
#     img = tf.image.convert_image_dtype(image, tf.uint8)
#     resized_img = tf.image.resize(img, (300, 300))
#     return resized_img[tf.newaxis, :]

# def set_input_tensor(interpreter, image):
#     """Set the input tensor for the model."""
#     tensor_index = interpreter.get_input_details()[0]['index']
#     input_tensor = interpreter.tensor(tensor_index)()[0]
#     input_tensor[:, :] = image

# def get_output_tensor(interpreter, index):
#     """Get the output tensor from the model."""
#     output_details = interpreter.get_output_details()[index]
#     tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
#     return tensor

# def detect_objects(interpreter, frame, threshold):
#     """Detect objects in the frame using the TensorFlow Lite model"""
#     preprocessed_image = preprocess_image(frame, (300, 300))
#     set_input_tensor(interpreter, preprocessed_image)
#     interpreter.invoke()

#     boxes = get_output_tensor(interpreter, 0)
#     classes = get_output_tensor(interpreter, 1)
#     scores = get_output_tensor(interpreter, 2)
#     count = int(get_output_tensor(interpreter, 3))

#     results = []
#     for i in range(count):
#         if scores[i] >= threshold:
#             result = {
#                 'bounding_box': boxes[i],
#                 'class_id': classes[i],
#                 'score': scores[i]
#             }
#             results.append(result)
#     return results

# def detect_from_image(image_path, threshold=0.5):
#     """Run object detection on a single image"""
#     # Load the image
#     image = Image.open(image_path)
#     image_np = np.array(image)

#     # Load the TFLite model
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()

#     # Run object detection
#     results = detect_objects(interpreter, image_np, threshold=threshold)

#     # Draw the detection results on the image
#     for obj in results:
#         ymin, xmin, ymax, xmax = obj['bounding_box']
#         xmin = int(xmin * image_np.shape[1])
#         xmax = int(xmax * image_np.shape[1])
#         ymin = int(ymin * image_np.shape[0])
#         ymax = int(ymax * image_np.shape[0])
        
#         print(xmin , " ", ymin , " ", xmax , " ", ymax)
        

#         class_id = int(obj['class_id'])
#         color = [int(c) for c in COLORS[class_id]]
#         cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, 2)

#         y = ymin - 15 if ymin - 15 > 15 else ymin + 15
#         label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
        
#         print(label)
        
#         cv2.putText(image_np, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Display the image with detection results
#     cv2.imshow("Object Detection", image_np)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Run object detection on a specific image
# image_path = "D:/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/01.jpg" 
# # image_path = "D:/Datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000344.jpg" 
# detect_from_image(image_path, threshold=0.58)


# from transformers import pipeline
# import numpy as np
# from PIL import Image

# pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
# image = Image.open('D:/Datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000026.jpg')
# depth = pipe(image)["depth"]

# # print(depth)

# # depth_array = np.array(depth)

# # Kiểm tra kích thước và giá trị của mảng độ sâu
# # print("Shape of depth array:", depth_array.shape)
# # print("Depth values:\n", depth_array)

# import matplotlib.pyplot as plt

# plt.imshow(depth, cmap='jet', axis=1)
# plt.colorbar()
# plt.title("Depth Estimation")
# plt.show()

# # Chuyển đổi độ sâu thành một hình ảnh (chuẩn hóa độ sâu)
# # depth_image = (np.array(depth) * 255 / np.max(depth)).astype(np.uint8)
# # depth_pil_image = Image.fromarray(depth_image)

# # # Lưu hình ảnh độ sâu
# # depth_pil_image.save("D:/Datasets/depth_result.png")
# # print("Depth map saved at D:/Datasets/depth_result.png")