# **FCOS: Fully Convolutional One-Stage Object Detection (2019)**

저자: Zhi Tian, Chunhua Shen, Hao Chen, Tong He

링크: https://arxiv.org/pdf/1904.01355

____

`FCOS`: **Anchor-free box**,**proposal-free**한 **Per-pixel Single-stage detector**

- 기존 모델들은 `Anchor box`로 인한 **Hyperparameter의 증가, 계산량의 증가, Anchor box로 인한 Background class가 너무 많아지는** 등의 문제를 가짐.

기존 모델들과 다르게 `Centerness`를 도입함

- Target의 중심으로부터 먼 위치에 있는 `Box`의 정확도가 낮아 이를 보완하기 위해 도입.

## Implementation

**Multi-level prediction**

- Pretrained CNN의 Stride가 크게 되면, Location이 없는 위치를 Recall하는 것이 불가능해진다.

  - Stride가 커지면 크기가 작은 객체를 탐지하는 것이 굉장히 어려워진다.  
  
- 이를 해결하기 위해, Pretrained CNN의 각 Layer에서의 Feature space에 대해 Prediction 수행하고, 나중에 하나로 합치는 방식을 채택하였다.

- 여러 크기의 객체를 탐지하기에 유리하고, 기존 Anchor-based model에서도 사용하던 방식이다.

**GT of boxes**: $B_i$ = $x_0^{(i)}, y_0^{(i)}, x_1^{(i)}, y_1^{(i)}, c^{(i)}$

- left-top and right-bottom corners

- $c=0$: Background

**locations**: $xc, yc$

- 원본 이미지에서의 Sample처럼 사용
- Feature space의 각 Pixel에 대응되는 Receptive field의 중심점
- 하나의 location이 두 개 이상의 Box에 매칭되면 `Ambiguous sample`이라고 부른다.
- Location을 기준으로 Prediction이 진행된다.

**deltas**: $l, t, r, b$

- $l = xc - x_0$, $t = yc - y_0$, $r = x_1 - xc$, $b = y_1 - yc$
- 위의 수식 전부 해당 Level의 Stride로 나누어야 한다.
- 모델은 위 방법대로 구한 deltas를 예측해야 한다.

**loss**

- Class는 `focal loss` 이용
- Box regression은 `IOU loss` 이용
- Centerness는 `BCE` 이용

**centerness**

- (min(l, r)*min(t, b)) / (max(l, r)*max(t,b) 의 제곱근으로 구한다.

**Score**

- Class probability과 Centerness를 통해 Geometric mean을 이용하여 구한다.
- Centerness: 박스 중심에서 멀리 떨어진 위치의 예측을 자동으로 깎아 점수를 낮춘다.

## Inference

모델이 예측한 `Deltas`를 이용하여 Box를 예측한다.

- x0' = xc - ls, y0' = yc - ts, x1' = r*s + xc, y1' = bs + yc

각 Level에서의 에측을 합친 후에 **NMS**를 진행한다.

- `Training` 때에는 모든 Box를 이용해야 하기 때문에 사용하지 않고, `Inference` 때 정확도를 높이기 위해 사용한다.
