---
title: Principle Component Analysis (PCA) trong lĩnh vực sản xuất thông minh (Smart manufacturing) - Phần 1
author: hoanglinh
categories: [Fundamental Concepts]
tags: [feature extraction]
math: true
img_path: posts_media/2023-09-15-pca-and-applications/
---

Trong thời đại số hóa hiện nay, chúng ta thường gặp phải các bộ dữ liệu có nhiều features (high-dimensional data) trong nhiều lĩnh vực, và sản xuất không phải là ngoại lệ. Dữ liệu có thể bao gồm hàng trăm hoặc hàng nghìn biến số, và việc làm việc với chúng thực sự là một thách thức và cần có những phương pháp phù hợp để phân tích, tìm insights và thực hiện các phân tích sau đó. Trong bài viết này, chúng ta sẽ tìm hiểu về Nguyên Tắc Phân Tích Thành Phần (Principle Component Analysis) và cách nó có thể được áp dụng để giải quyết những hạn chế của dữ liệu có chiều cao trong lĩnh vực sản xuất.

Phân tích thành phần chính (PCA) là một kỹ thuật giảm chiều dữ liệu. PCA làm điều này bằng cách biến đổi dữ liệu sang một tập hợp các chiều mới, trong đó các chiều mới này được gọi là thành phần chính. Các thành phần chính được sắp xếp theo thứ tự giảm dần theo sự giải thích của chúng đối với biến thiên trong dữ liệu.

## Giới thiệu chung về feature selection / extraction

**Feature selection** là phương pháp chọn `subset` trong tất cả `features`

-   Dùng trong trường hợp các features không phải số (string), cần extract meaningful features hoặc muốn giữ giá trị đúng cho các features (keep measurements intact)

-   Cần có phương pháp search (grid search, meta-heuristic algorithms, etc.) → có thể tốn nhiều thời gian

**Feature extraction** thay vì chọn thì sẽ build một hàm linear combination tất cả features lại thành một số lượng features mới → **minimize the loss of information**

-   Rewrite as a set of weights that make up the transformation step; $y=Wx$, where $W$ are the weights, $x$ are the input features, and $y$ is the final transformed feature space.
-   Nhược điểm là khó giải thích sau khi combine lại mặc dù có thể cho kết quả tốt hơn → khó thực hiện các phân tích tiếp theo (ví dụ gene was partially involved together with other genes)

![pca-concept](pca-concept.webp){:width="600"}_Schematic overview of the Feature Extraction procedure that linearly transforms the input data in the form $y=Wx$ [1]_

## Cách hoạt động của PCA

Xét tập không gian gồm $k$ features, ta cần biểu diễn $j$ thành phần chính sao cho $j<k$. Tuy nhiên, để đơn giản, ta sẽ lấy ví dụ với $j=2$ trước.

1.  **Ta tính giá trị trung bình** của tất cả data theo từng features → shift data về gốc tọa độ 0 bằng cách trừ tất cả giá trị data cho mean value. Chú ý rằng biến đổi này không làm thay đổi khoảng cách giữa các data mà chỉ là shift nó về gốc tọa độ.

    ![pca-step1](step-1.png){:width="600"}_Tính giá trị trung bình và shift về gốc tọa độ_

2.  **Tìm đường fit cho tất cả data point** và đi qua gốc tọa độ sao cho khoảng cách giữa tất cả data đến đường thẳng đó là nhỏ nhất (có cách giải thích khác trong video [2])  bằng cách quay đường thẳng và thử cho đến khi tìm được tổng giá trị nhỏ nhất khoảng cách giữa tất cả data và đường fit. 

    Tuy nhiên, thường dễ dàng hơn nếu ta tính **maximize the distances from the projected data points to the origin (sum of squared distances (SS))** (cách tính chi tiết dựa theo định lý Pytago được trình bày rất dễ hiểu trong video [2]). Cuối cùng, ta tìm được đường thẳng với giá trị variance (sum of squared distances) tối đa.

    ![pca-step2](step-2.png){:width="600"}_Finding the best fit. Start with a random line (top) and rotate until it fits the data best by minimizing the distances from the data points to the line (bottom)_

3.  **Computing the Principal Components and the loadings**:

    Sau khi xác định được đường “best-fitted line”, hay còn gọi là 1st principle components PC1. Tiếp theo ta cần tính toán slope (góc) của PC1 để biết mức độ contribute của mỗi feature cho PC1.

    -   Trong ví dụ này, ta có thể để ý rằng, data có xu hướng phụ thuộc (spread out) vào feature 1 nhiều hơn feature 2. Giá trị feature 1 càng xa gốc tọa độ, giá trị của data cũng dãn theo. Ta có thể tính được góc (slope) của đường fit từ bước 2, nhận thấy mỗi 2 units feature 1 kéo theo giảm 1 unit với feature 2. Từ đó, ta có thể tính được eigenvector của PC1 là 2.33, tính toán theo hình dưới.

    -   Tuy nhiên, ta cần standardlize các giá trị lại theo unit vector, chỉ đơn giản là chia $a, b, c$ cho $2.23$. Như vậy, range của vector đều nằm trong khoảng $-1, 1$. Ví dụ, trong hình giá trị của $b=0.89$ nghĩa là feature 1 contribute rất nhiều cho PC1, giá trị càng lớn thì mức độ contribute càng nhiều.

        ![pca-contribute](pca-contribute.png){:width="600"}_Computing PC1 and PC2 and determining the loadings_

    -   Bước tiếp theo là xác định PC2, là đường thẳng cũng đi qua gốc tọa độ và vuông góc với PC1. Đối với các data có nhiều feature hơn, ta chỉ cần tìm đường PC1 và sau đó là các đường vuông góc với nó. *New latent variables, aka the PCs, are a linear combination of the initial features. The proportion of each feature that is used in the PC is named the coefficient.*

    ![pca-2com](2-pca.png){:width="600"}_Computing PC1 and PC2 and determining the loadings._

> **Standardization**: Thực tế là, trước khi thực hiện PCA, ta cần phải standardlize tất cả data. Lý do mình để phần này ở cuối cùng vì khi bạn đọc đã hiểu được concept của PCA, ta có thể hình dung ra các yếu tố có thể ảnh hưởng đến độ chính xác kết quả PCA. Rõ ràng như vậy, trong quá trình tìm đường fiting, nếu data không cùng range (diện tích ($m^2$) và giá nhà (tỷ)) hoặc data có outlier, đường fit sẽ bị ảnh hưởng rất lớn. Ta có thể thực hiện standardlize dễ dàng với Scikit-learn với hàm `StandardScaler()`.
{: .prompt-info}

## Loadings - Tính toán mức độ contribute của mỗi component

Ta có thể nhận ra rằng các components khó có thể giải thích và không có ý nghĩa thực sự vì chúng được xây dựng dựa trên sự kết hợp tuyến tính của các feature ban đầu. Nhưng chúng ta có thể phân tích các trọng số mô tả sự quan trọng của các features đối với component đó. Các trọng số bằng với các hệ số của các feature và cung cấp thông tin về feature nào đóng góp lớn nhất cho các principle component.

-   Các trọng số (loadings) có giá trị từ $-1$ đến $1$.
-   Giá trị tuyệt đối cao (gần 1 hoặc -1) cho ta biết rằng feature đó ảnh hưởng mạnh và giá trị gần 0 cho biết feature ảnh hưởng yếu đến component.
-   Dấu của một trọng số (+ hoặc -) cho biết liệu một biến và một thành phần chính có tương quan dương hay âm.

Chúng ta đã tính được các thành phần chính (PCs) và bây giờ ta xoay (transformation) toàn bộ bộ dữ liệu sao cho trục $x$ là hướng có phương sai lớn nhất (largest variance). Lưu ý rằng bước biến đổi này sẽ làm mất giá trị của các feature gốc. Mỗi PC sẽ chứa một tỉ lệ của tổng phương sai (total variance) và từ đó có thể biết được mức độ quan trọng của mỗi component, PC nào có tỉ lệ lớn nhất nghĩa là quan trọng nhất.

![pca-transformation](transformation.png){:width="600"}_Transformation of the entire dataset and determining computing the explained variance_

>   Từ đây, PCA cho phép ta reduce dimension của data mà giữ lại được tối đa thông tin. Tuy nhiên, cần chú ý là các PC này khó giải thích và không có ý nghĩa thực tế vì nó chỉ là combination của các features.
{: .prompt-info}

![pca-output](pca-output.png){:width="600"}_Ví dụ 87% of the information (variances) contained in the data are retained by the first five principal components. Source [4]_

## Conclusion and Future works

Trên đây là phần giải thích sơ bộ về thuật toán PCA. Hy vọng giúp bạn đọc có cái nhìn chung về phương pháp cũng như hiểu về bản chất làm thế nào và lựa chọn được phương pháp phù hợp cho từng bài toán. Trong bài tiếp theo, chúng ta sẽ cùng code PCA from scratch và với thư viện sẵn có cũng như ứng dụng thực tế trong nghiên cứu về thực tế.

## Recommended resources for further learning

Here are some recommended books for delving into deep learning from scratch:
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://amzn.to/3YZeOAk) by Aurélien Géron
- [Deep Learning from Scratch: Building with Python from First Principles](https://amzn.to/40gyjFQ) by Seth Weidman
- [Data Science from Scratch: First Principles with Python](https://amzn.to/40ep3T7) by Joel Grus, a research engineer at the Allen Institute for Artificial Intelligence

## References

1. <https://towardsdatascience.com/what-are-pca-loadings-and-biplots-9a7897f2e559>
1. Một bài giải thích khác cũng rất dễ hiểu: <https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d>
1. Giải thích concept rất dễ hiểu: <https://www.youtube.com/watch?v=FgakZw6K1QQ>
1. <http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/112-pca-principal-component-analysis-essentials/>