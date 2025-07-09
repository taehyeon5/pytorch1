import torch

# 1) 데이터 입력
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])  

# 2) 파라미터 초기화
w = torch.tensor([1.0], requires_grad=True)

# 3) 학습률 설정
learning_rate = 0.1

# 4) 반복문으로 학습 (epoch)
for epoch in range(20):
    # 1) 예측값
    y_pred = w * x
    # 2) 손실함수
    loss = torch.mean((y_pred - y_true) ** 2)
    # 3) 역전파로 기울기 계산
    loss.backward()
    # 4) 파라미터 업데이트 (경사하강법)
    with torch.no_grad():  
        w -= learning_rate * w.grad
    # 5) 기울기 초기화
    w.grad.zero_()
    # 6) 로그 출력
    print(f"Epoch {epoch+1:2d} | w: {w.item():.4f} | Loss: {loss.item():.4f}")