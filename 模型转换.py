import tensorflow as tf

# 加载你现有的 Keras 模型
model = tf.keras.models.load_model('classifier/emotion_models/simple_CNN.530-0.65.hdf5')

# 创建 TFLite 转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 开启默认优化（包含权重剪枝和 INT8 量化），大幅减小体积并提升速度
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 执行转换并保存文件
tflite_quant_model = converter.convert()
with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)