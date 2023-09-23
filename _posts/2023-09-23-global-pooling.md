---
title: Global Pooling in Convolutional Neural Networks
author: hoanglinh
categories: [Fundamental Concepts]
tags: []
math: true
img_path: posts_media/2023-09-23-global-pooling/
---

Các phép toán pooling đã trở thành một phần quan trọng trong mạng convolution neural network (CNN). Trong khi các phương pháp như max pooling và average pooling thường được biết đến nhiều hơn, những phiên bản ít biết hơn là global max pooling và global average pooling cũng được sử dụng ở các lớp cuối trong mạng CNN. Trong bài viết này, chúng ta sẽ tìm hiểu về `global average pooling` và `global max pooling`.

## The Classical Convolutional Neural Network

Như chúng ta đã viết, phương pháp CNN thường được sử dụng trong các bài toán liên quan đến hình ảnh và computer vision vì khả năng phân tích và học được các spatial structure của input. Từ đó, nó có thể liên kết được các mỗi quan hệ giữa các pixel gần nhau và vị trí của các objects trong ảnh. Mặt khác, mạng multilayer perceptrons (MLP) có thế mạnh là học được mối quan hệ giữa feature vectors và targets. Tuy nhiên, MLP lại hạn chế vì nó không có khả năng học được spatial relationship như trong các bài toán liên quan đến xử lý ảnh.

Rất nhiều các mạng classical CNN thực tế là kết hợp giữa convnets và MLPs. Ví dụ như đối với kiến trúc của mạng NN nổi tiếng LeNet, các layer ở trước gồm một tập hợp các CNN layer đi kèm với pooling layer, tiếp theo đó là linear layers (dense) được gắn vào cuối của mạng. Việc kết hợp này hiển nhiên là rất tốt vì tận dụng được thế mạnh của hai phương pháp.

![alexnet](alexnet.png)_AlexNet Architecture. Source: [https://blog.devgenius.io/alexnet-the-net-that-surpassed-cnns-5d551ba1b901](https://blog.devgenius.io/alexnet-the-net-that-surpassed-cnns-5d551ba1b901)")_

Tuy nhiên, các linear layers có xu hướng bị overfitting vì số lượng parameter lớn làm cho model thực tế chỉ cố nhớ data chứ không thực sự học được các patterns. Dropout regularization cũng được áp dụng nhưng vấn đề vẫn còn đó. Bên cạnh đó, mặc dù các CNN layer học spatial structures nhưng cũng đóng góp phần nào vào quá trình overfitting.

## Modern Solutions to a Classical Problem

Để ngăn chặn vấn đề overfitting trong mạng convnets, bước tiếp theo sau dropout regularization một cách hợp lý là hoàn toàn loại bỏ các lớp tuyến tính (linear layers). Nếu loại bỏ các lớp tuyến tính, chúng ta cần tìm một cách nào đó để giảm kích thước của các feature maps sau CNN và tạo ra một vector có kích thước bằng với số lớp cần phân loại, và đó là lúc global pooling come into play.

Hãy xem xét một bài toán phân loại 3 classes, trong đó lớp convnets sẽ giúp giảm kích thước của các feature maps cho đến khi chúng chỉ còn 3. Tuy nhiên, global pooling sẽ giúp tạo ra một vector có 3 phần tử, và biểu diễn này có thể được sử dụng bởi hàm mất mát để tính toán đạo hàm.

Điều này có nghĩa rằng thay vì sử dụng các linear layers để chuyển từ feature maps sang biểu diễn vector, ta sử dụng global pooling để thực hiện việc này. Global pooling có thể là global average pooling hoặc global max pooling, tùy thuộc vào cách bạn muốn tổng hợp thông tin từ các feature maps. Sau đó, vector này có thể được sử dụng để tính toán độ lớn của gradient và cập nhật trọng số của mô hình trong quá trình huấn luyện.

Giả sử vẫn với bài toán phân loại bên trên, output của ta gồm 3 classes và sau nhiều layer CNN, ta còn lại 3 feature maps of size . Note that, ta có thể dùng  convolution layer với kernel size bằng 3 để có được số lượng feature maps sau đó bằng với số lượng classes cần phân loại. Có 2 cách để tạo ra được 3 element vector là tính toán giá trị average hoặc max, hay cũng chính là global average / max pooling

![global-pooling](global-pooling.png)_Global pooling. Source: [https://medium.com/analytics-vidhya/the-world-through-the-eyes-of-cnn-5a52c034dbeb](https://medium.com/analytics-vidhya/the-world-through-the-eyes-of-cnn-5a52c034dbeb)")_

## Recommended resources for further learning

Here are some recommended books for delving into deep learning from scratch:
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://amzn.to/3YZeOAk) by Aurélien Géron
- [Deep Learning from Scratch: Building with Python from First Principles](https://amzn.to/40gyjFQ) by Seth Weidman
- [Data Science from Scratch: First Principles with Python](https://amzn.to/40ep3T7) by Joel Grus, a research engineer at the Allen Institute for Artificial Intelligence

## References

1. <https://blog.paperspace.com/global-pooling-in-convolutional-neural-networks/>