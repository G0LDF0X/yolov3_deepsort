# 과제 개요
## 과제 선정 배경 및 필요성

과제 선정 배경 및 필요성 최근 객체 추적 분야에서 딥러닝의 활용도가 증가하고 있다. 물체의 다중 객체 추적(MOT), 혹은 다중 대상 추적(MTT)은 여전히 발전 가능성이 높은 분야며, 컴퓨터 비전 분야에서도 많은 활용도를 예측하고 있다. 따라서 이러한 객체 추적의 알고리즘 중 하나인 DEEPSORT 알고리즘이 보다 높은 성능을 낼 수 있는 방법에 대한 과제를 진행하고자 한다.

## 과제 주요 내용
객체 추적은 객체의 classification보다 더 높은 난이도를 보이는데, 물체가 가려질 경우 기존의 추적하던 ID와 동일한 물체인지 판별하는 과정의 난이도가 높기 때문이다. 또한 대부분의 객체 추적은 Real-Time을 요구하기 때문에 탐지기의 성능 또한 중요하다. 본 연구에서는 YOLO-v3 모델을 사용하여 MOT를 진행하고, 이 과정에서 추적 정확도를 향상시킬 수 있는 방법 및 알고리즘을 연구한다.

## 최종 결과물의 목표
기존의 모델과 개선된 DEEPSORT 알고리즘을 적용한 모델 각각에 MOT metric을 적용하여, 기존의 모델보다 개선된 MOTA, MOTA값을 얻도록 한다. 구체적으로는 동일한 영상을 각각의 모델로 객체 추적을 진행한 뒤, MOT metric을 통해 ground truth와 비교하여 각각의 MOTA와 MOTP값을 계산한다.

# 과제 수행 방법
먼저 기존 DEEP SORT 알고리즘의 구조에 대해 조사한 뒤, MOT Benchmark에서 제공하는 영상을 이용해 DEEPSORT 알고리즘이 제대로 물체 추적을 진행하고 있는지 체크한다. 이후 MOT metric을 통해 groundtruth와 해당 영상을 비교하고, 코드의 정상 작동을 확인하면 DEEPSORT 알고리즘의 Code Review를 진행한다. 코드의 분석이 끝나면 타 코드와 비교해보거나 혹은 관련된 논문 탐색, 코드의 심층 분석 등을 통해 어떤 방식으로 DEEPSORT를 개선시킬 것인지 결정한 뒤 기존의 모델과 비교를 진행한다.

# 진행 내용
## 과제진행 내용
현재 일정표에 있는 DEEP SORT 코드 탐색과 Matching Cascade 스터디, 기존의 모델에 MOT metric 적용 및 확인, DEEP SORT 코드 세부 리뷰 항목이 끝난 상태다. 현재는 코드의 어떤 부분을 수정하여 개선시킬 것인지에 대한 과제를 진행하고 있다.

## 진행내용의 주요특징 및 설명
먼저 GitHub에 접속하여 YOLO-v3를 이용한 DEEP SORT 알고리즘을 찾아보았다. 그중에는 Tensorflow를 사용하는 것도 있었고 keras를 이용하는 것, mxnet을 이용하거나 darknet을 이용하는 것도 있었다. 목적은 코드의 분석이므로 각각의 차이점을 확인하기 위해 모든 코드를 다운받은 뒤, Matching Cascade의 동작에 대해 확인했다. Matching Cascade는 DEEP SORT 알고리즘에서 가장 중요한 부분으로, 기존의 SORT 알고리즘과의 차이점을 두는 부분이기도 하다. 기존의 SORT 알고리즘은 Detection과 Track을 IOU Matching에 넣은 뒤, Unmatched Track은 삭제, Unmatched detection은 새 트랙으로, matched track은 칼만 필터로 업데이트를 진행해 트랙에 다시 넣었는데 DEEP SORT에서는 이 IOU Matching의 앞에 Matching Cascade 단계가 들어간다. 즉, detection과 track중에서도 confirmed된 track을 먼저 Matching Cascade에 넣은 뒤, 이 단계에서 matched 된 track은 칼만 필터로 업데이트하여 트랙에 다시 넣는다. 여기서 unmatched track, unmatched detection, unconfirmed track을 이용하여 다시 IOU matching을 진행하는데, Matched된 track은 Matching Cascade와 동일하게 칼만 필터로 업데이트하여 트랙에 넣고, Unmatched detection은 SORT에서의 IOU matching과 동일하게 새 트랙으로 넣고, Unmatched track을 바로 삭제했던 SORT와는 달리 Unmatched Track이 Confirmed 되었고, max_age보다 작은 경우에는 트랙에 넣는다. 그 외의 경우에는 트랙에서 삭제한다. 이후 GitHub에서 MOT를 측정할 수 있는 metric 코드를 발견하여 MOT Benchmark의 영상을 넣어 metric을 체크해보았으며, 모든 코드가 정상적으로 작동하는 것을 확인한 이후에야 DEEP SORT의 세부 Code Review를 진행했다. 가장 상단에서 작동하는 main 코드를 중심으로 각각의 코드가 어떤 파이썬 파일과 연결되어 있고 어떻게 동작하는지 확인했으며, 다른 코드와의 차이점 비교를 진행하며 개선점을 찾는 중이다.

# 향후계획
먼저 DEEP SORT의 구조 자체(detection.py, iou_matching.py, kalman_filter.py, linear_assignment.py, nn_matching.py, preprocessing.py, track.py, tracker.py)는 동일하기 때문에, 다른 방식의 DEEP SORT 코드 두 개를 비교 분석하여 어떤 차이점을 가지고 있는지 확인하고 이후 코드를 개선해본다. 개선한 코드로 DEEP SORT의 MOT를 진행한 뒤, metric으로 해당 모델을 평가한 뒤 개선되었는지 개선되지 않았는지의 여부를 판단한다. 개선되지 않았을 경우, 다른 방법으로 코드의 개선을 시도, 이후를 반복한다.
