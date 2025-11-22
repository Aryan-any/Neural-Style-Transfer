# Fast Neural Style Transfer

## Introduction 
Neural Style Transfer (NST) is an advanced technique in artificial intelligence that
merges the content of one image with the artistic style of another using convolutional
neural networks. This report explores the implementation and evaluation of fast Neural Style Transfer using the TransformerNet architecture, highlighting its applications,
methodology, results, and potential improvements. Through detailed analysis, the
report demonstrates how NST can transform images, maintaining the structural elements of the content image while adopting the stylistic features of the reference image.
Key findings include high-quality image generation and flexibility in style application,
though challenges such as computational demands and handling complex styles remains.

To view the project report,[Click Here](https://drive.google.com/file/d/1B7xWG_KSRPRHddRoMIkjo1fzwJUXnZdK/view?usp=sharing) or demo video, [Click Here](https://drive.google.com/file/d/1HcGsahvrPKm6-p8mgTMTgp0A_JbSovBE/view?usp=sharing).

## Table of Contents :bar_chart:
- Requirements
- Datasets Used
- Data Preprocessing
- Models Architecture
- Loss Functions
- Hyperparameters
- Results
- How To Run
- Challenges Faced
- References

## Requirements :bell:
```
pillow
torch
torchvision
tqdm
collections
```
## Dataset Used :school_satchel:
I have used COCO 2017 dataset for training the TransformerNet. The COCO (Common
Objects in Context) dataset is a large-scale object detection, segmentation, and captioning
dataset widely used in the computer vision community. By selecting 40,000 test images from
the COCO 2017 dataset, we ensured a diverse and comprehensive training set that includes
a wide range of content types and visual scenarios. This diversity is crucial for enhancing
the robustness and generalization capabilities of the TransformerNet model.
8 Styles images were used to train TransformerNets. Each style need a saparate TransformetNet to train.


For COCO 2017 dataset [Click Here]([https://www.kaggle.com/datasets/soumikrakshit/lol-dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset))

## Data Preprocessing :scissors:
1. Data Preprocessing for COCO 2017 Images
   ```ptython
   def TrainImagesPath(path):
    images_path = []
    for file in os.listdir(path):
        if file == "test2017":
            file_path = os.path.join(path, file)
            for f in os.listdir(file_path):
                image_path = os.path.join(file_path, f)
                images_path.append(image_path)
                if len(images_path) == 40000:
                    break
    return images_path
   ```
   The code focuses on a specific folder named ”test2017” within the provided path. This
  suggests it’s only interested in a particular subset of images for training, possibly due
  to limitations or training efficiency.
    ```python
    train_transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop((256, 256)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.mul(255))
                                    ])
    ```
    By applying transforms.Resize and transforms.CenterCrop, it enforces a uniform size
    (256x256 pixels) on all images. This ensures the model receives data with consistent dimensions. The transforms.ToTensor function converts the loaded images (likely in PIL
    format) into tensors, which is a format commonly used by machine learning frameworks like PyTorch. A lambda function to multiply each pixel value by 255. This
    normalizes the pixel intensities to a range between 0 and 1, which can be beneficial for
    certain machine learning algorithms. This is a lambda function that takes a tensor x
    (representing an image) as input. The mul function (short for multiply) element-wise
    multiplies each value in the tensor by 255.
2. Data Preprocessing for Style Images:
   ```python
   def LoadStyleImage(path, batch_size, device):
    style = Image.open(path).convert('RGB')
    style = style.resize((256, 256))
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1)
    style = style.to(device) 
    return style
   ```
   The image is opened using Image.open and converted to RGB format with convert('RGB'), ensuring it has three color channels. It's resized to 256x256 pixels with style.resize((256, 256)) for model consistency.    The preprocessed style image is then replicated batchsize times using style.repeat(batchsize, 1, 1, 1), which is useful for batch processing. Finally, the tensor is transferred to the specified device (CPU or    GPU) with .to(device).
   ```python
   style_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.mul(255))
                            ])
   ```
   Converts the PIL image format into a PyTorch tensor suitable for computations. Normalizes the pixel values between 0 and 255.

## Models Architecture :triangular_ruler:

### 1. ConvLayer :
This class represents a convolutional layer, a fundamental building block in Convolutional
Neural Networks (CNNs). It performs the core operation of a CNN - extracting features
from the input data using learnable filters (kernels).
### 2. ResidualBlock :
This class defines a residual block, a commonly used architecture in deep residual networks.
It introduces a ”shortcut connection” that allows the gradient to flow directly through the
layers, addressing the vanishing gradient problem that can hinder training deep neural networks.

### 3. UpsampleConvLayer :
This class defines a convolutional layer with upsampling capabilities. It’s often used in decoder parts of networks where the goal is to increase the spatial resolution of the feature maps.

### 4. TransformerNet
It combines convolutional layers, residual blocks, and upsampling convolutions to achieve
the desired style transfer effect.

## Loss Functions :
1. Feature reconstruction loss encourages the model to capture similar features rather than exact pixel values. It uses a pre-trained network to extract key features like edges and textures from both the target and output images. The loss is calculated based on the difference between these features, ensuring the model captures the essence of the target image, even if pixel values differ.
   ![image](https://github.com/vaibhavprajapati-22/Fast-Neural-Style-Transfer/assets/148644657/263a46e9-6c3c-4cb4-af04-6de59963da95)

2. Style reconstruction loss analyzes feature co-occurrences using a pre-trained network, calculating the Gram matrices to capture stylistic elements like colors and textures. By comparing these matrices between the generated and reference images across multiple layers, it ensures the model captures stylistic fingerprints accurately, from small-scale brushstrokes in lower layers to larger-scale color harmonies in higher layers
   
   ![image](https://github.com/vaibhavprajapati-22/Fast-Neural-Style-Transfer/assets/148644657/4bde0d5b-3421-464e-a37e-b7b4ca3368c3)

## Hyperparameters :dart:
All the model were trained on GPU P100 that is available on kaggle.
- Batch size : 4
-  Content weight : 1e5
- Style weight : 1e10
- Learning rate : 0.001
-  epochs : 1
## Results :loudspeaker:
![image](https://github.com/vaibhavprajapati-22/Fast-Neural-Style-Transfer/assets/148644657/e968c21e-4ade-4bda-8368-589dbdfb96d5)

![image](https://github.com/vaibhavprajapati-22/Fast-Neural-Style-Transfer/assets/148644657/ef5a717f-45a9-4cc0-9aaf-f8b445a64225)

Demo video, [Click Here](https://drive.google.com/file/d/1HcGsahvrPKm6-p8mgTMTgp0A_JbSovBE/view?usp=sharing).

## How To Run :gun:
  1. Clone the repository:
    <pre>
    <code class="python">
    git clone https://github.com/Aryan-any/Neural-Style-Transfer
    </code>
    </pre>
  2. Install the required dependencies:
     <pre>
      <code class="python">
        cd Fast-Neural-Style-Transfer
        pip install -r requirements.txt
      </code>
     </pre>
  3. Run main script
     <pre>
      <code class="python">
        streamlit run main.py
      </code>
     </pre>

After this default browser will be opened with local host. You can now select any of the 8 styles and upload a content image to get it stylized.

You can also visit the Streamlit deployed website by [clicking Here](https://fast-neural-style-transfer-vaibho.streamlit.app/).
## Challenges Faced :mag_right:
  - Due to limited GPU memory, training was not possible if the size of style image is
more than 2MB so we should resize it to (256, 256)
  - Each style requires a separate training, so its a time consuming process to get weights
for 8 styles.
## References :paperclip:
  1. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
  2. [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
