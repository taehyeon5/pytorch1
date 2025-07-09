import torch

#1.x,y 입력
x = torch.tensor([1.0,2.0,3.0])
y_true = torch.tensor([2.0,4.0,6.0])

#2.w 초기화
w = torch.tensor([0.1],requires_grad=True)

#3.예측값 만들기 w * x
y_pred = w * x

#4.손실값 계산
loss = torch.mean((y_pred - y_true) ** 2)

#w,예측값,손실값 출력
print(f"현재 w: {w.item():.4f}")
print(f"예측값:{y_pred}")
print(f"손실값:{loss.item():.4f}")

#5.역전파
loss.backward()
print(f"손실함수를 w에 대해 미분한 기울기:{w.grad.item():.4f}")

#6.경사하강법 (w 업데이트 및 기울기 초기화)
learning_rate = 0.1
w_updated = w - learning_rate * w.grad

print(f"업데이트된 w:{w_updated.item():.4f}")