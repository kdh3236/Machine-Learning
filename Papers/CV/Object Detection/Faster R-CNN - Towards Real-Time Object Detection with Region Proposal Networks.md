# Faster R-CNN - Towards Real-Time Object Detection with Region Proposal Networks.md (2016)

저자: Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun

링크: https://arxiv.org/pdf/1506.01497

___

기존 Region Proposal Algorithm이 Location을 가정하기 때문에 실행 시간에 있어서 Bottleneck이 된다.

Region Proposal을 제외한 실행 시간은 굉장히 빨라졌지만, 여전히 **Region Proposal에 사용되는 시간은 여전히 오래 걸린다.**

이를 해결하기 위해, **Region Proposal Network (RPN)을** 도입한다.

- Object Detection에서 End-to-end pipeline이 가능하도록 한다.

- Object bound와 Object score를 예측하도록 한다.

하나의 Feature Map 위에 여러 크기의 Anchor Box를 이용한다.

- 이를 통해 FPN 없이도 여러 크기의 객체를 커버할 수 있다.
  
## Implementation

Faster R-CNN은 크게 두 부분으로 나뉜다.

- Region Proposal을 위한 Deep fully convolutional network
- Fast R-CNN에서 사용된 Detector (Second module)

먼저, Region Proposal Networks부터 살펴보자.

- Convolutional Network를 통과한 Feature map 위에 **N x N spatial window를 slide한다.**
- Box regression과 Object 여부를 예측하기 위한 2가지의 FC Layer로 전달된다.
- Spatial window에 해당되는 각 영역에는 $k$개의 Anchor box가 존재하며, 각 **Anchor box에 대해 Box regression과 Positive/Negative를** 예측한다.
- Training을 위해서 각 Anchor box가 Object인지 아닌지 (Positive / Negative) 두 가지 중 하나로 표시되도록 했으며, **Positive**는 Ground truth box와의 IoU가 임계값 이상, **Negative**는 IoU가 임계값 이하가 되도록 하였다.
- Multi-task loss를 사용하였다.

이후, NMS를 적용하여 Region을 생성하였다.

생성된 Region 위 RoI Pooling을 진행하고, Conv+FC Layer를 통과시켜 Class score와 Box Regression을 얻는다.

- Training 시, 마찬가지로 Multi-task loss를 사용한다.

## Detail

Box regression에서는 Box Offset의 타켓으로 아래의 값을 사용한다.

- 중심 좌표 (x, y) + 폭/높이 (w, h)

- anchor = $(x_a, y_a, w_a, h_a)$, gt = $(x, y, w, h)$

- $t_x = (x-x_a) / w_a$, $t_y = (y-y_a) / h_a$, $t_w = log(w / w_a)$, $t_h = log(h / h_a)$

Training 시, 주어진 Box prediction을 이용하여 $t_x, t_y, t_w, t_h$를 예측하여 Target과의 Loss를 계산한다.

Training 시, 위 네 개의 값을 예측하고자 한다.
