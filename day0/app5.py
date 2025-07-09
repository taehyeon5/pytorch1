import torch
import torch.nn as nn 

# 1) 데이터
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

# 2) 모델 클래스 정의
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.1]))
        self.b = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return self.w * x + self.b

# 3) 모델 인스턴스 생성
model = LinearRegressionModel()

# 4) 옵티마이저
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 5) 반복문 (Epoch)
for epoch in range(20):
    # 순전파 (예측)
    y_pred = model(x)
    # 손실함수 (MSE)
    loss = torch.mean((y_pred - y_true) ** 2)
    # 역전파
    loss.backward()
    # 파라미터 업데이트 & 기울기 초기화
    optimizer.step()
    optimizer.zero_grad()
    # 로그 출력
    print(f"epoch:{epoch+1:2d} | w:{model.w.item():.4f} | b:{model.b.item():.4f} | loss:{loss.item():.4f}")
