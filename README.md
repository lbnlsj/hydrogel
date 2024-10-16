# 纤维复合水凝胶分析项目进度说明文档

## 项目概述

本项目旨在通过深度学习技术分析纤维复合水凝胶的结构和性能关系。我们的主要目标是建立一个能够在纤维凝胶结构图和位移/应变图之间进行双向预测的模型，同时也探索结构图与初始模量之间的关系。

## 当前进展

### 1. 数据加载和预处理

我们已经实现了从指定目录加载和预处理图像数据的功能。主要的数据加载函数是`load_and_preprocess_images`：

```python
def load_and_preprocess_images(folder_path):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('L')  # 转换为灰度图
            img = img.resize((256, 256))  # 调整大小
            img_array = np.array(img) / 255.0  # 归一化
            images.append(img_array)
            filenames.append(filename)
    return np.array(images), filenames
```

这个函数完成了以下几个关键步骤：
1. 读取指定文件夹中的所有PNG图像。
2. 将图像转换为灰度图，统一调整大小为256x256像素。
3. 对图像数据进行归一化处理（将像素值缩放到0-1范围）。
4. 返回处理后的图像数组和对应的文件名列表。

我们分别加载了输出图像和对应的输入图像：

```python
output_images, output_filenames = load_and_preprocess_images('sample/output')
print(f"Loaded {len(output_images)} images from output folder")

# 假设input文件夹中有对应的输入图像
input_folder = 'sample/input'
input_images = []
for filename in output_filenames:
    input_path = os.path.join(input_folder, filename)
    if os.path.exists(input_path):
        img = Image.open(input_path).convert('L')
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        input_images.append(img_array)
    else:
        print(f"Warning: No corresponding input image for {filename}")
```

这段代码确保了输入和输出图像是一一对应的，同时还添加了错误检查机制，以防某些输入图像缺失。

### 2. 模型架构

我们定义了一个基本的图像到图像转换模型，使用了卷积神经网络（CNN）的架构：

```python
def create_image_to_image_model():
    model = models.Sequential([
        layers.Input(shape=(256, 256, 1)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

这个模型使用了编码器-解码器结构，适合于图像到图像的转换任务：
- 编码器部分使用卷积和池化层逐步提取图像特征。
- 解码器部分使用上采样和卷积层重构输出图像。
- 最后一层使用sigmoid激活函数，确保输出像素值在0-1范围内。

### 3. 训练和评估

我们实现了一个通用的训练和评估函数：

```python
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=0.2, verbose=1)
    
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss}")
    
    return history, test_loss
```

这个函数执行以下步骤：
1. 训练模型，使用20%的训练数据作为验证集。
2. 在测试集上评估模型性能。
3. 返回训练历史和测试损失，以便进一步分析。

### 4. 主程序流程

在主程序中，我们实现了以下逻辑：

```python
if __name__ == "__main__":
    if len(input_images) == len(output_images):
        # 任务1：图像到图像的转换
        X_train, X_test, y_train, y_test = train_test_split(input_images, output_images, test_size=0.2, random_state=42)
        model_img2img = create_image_to_image_model()
        history_img2img, loss_img2img = train_and_evaluate(model_img2img, X_train, y_train, X_test, y_test)
        
        print("Training and evaluation completed.")
    else:
        print("Error: Number of input images does not match number of output images.")
        print(f"Input images: {len(input_images)}, Output images: {len(output_images)}")
```

这段代码首先检查输入和输出图像的数量是否一致，然后执行以下步骤：
1. 将数据集分割为训练集和测试集。
2. 创建图像到图像转换模型。
3. 训练模型并评估其性能。

## 下一步计划

1. 实现图像到数值（初始模量）的预测模型。
2. 添加反向任务：从输出图像预测输入结构。
3. 实现更复杂的模型架构，如U-Net或ResNet变体，以提高预测精度。
4. 添加更多的评估指标，如PSNR（峰值信噪比）和SSIM（结构相似性）。
5. 实现可视化功能，以直观地比较预测结果和真实图像。
6. 优化数据加载流程，增加数据增强技术以提高模型泛化能力。
7. 实现模型保存和加载功能，方便后续使用和进一步实验。

通过这些步骤，我们将逐步完善模型，提高其在纤维复合水凝胶分析任务中的性能和实用性。