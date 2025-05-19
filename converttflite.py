import tensorflow as tf

# Chuyển đổi mô hình đã lưu sang TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(r"D:/Code/Python/Project/BlindAssistant/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model")

# Kích hoạt Flex Ops để hỗ trợ các phép toán TensorFlow không có sẵn trong TensorFlow Lite
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# Chuyển đổi mô hình
tflite_model = converter.convert()

# Lưu mô hình TFLite vào tệp
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model đã được chuyển đổi thành công!")
