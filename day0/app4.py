import torch

#1.x,y값 넣기
x = torch.tensor([1.0,2.0,3.0])
y_true = torch.tensor([2.0,4.0,6.0])

#2.w값초기화
w = torch.tensor([0.1],requires_grad=True)
b = torch.tensor([0.0],requires_grad=True)

#3.옵티마이저 생성(lr,opt)
learning_rate = 0.1
optimizer = torch.optim.SGD([w,b],lr=learning_rate)

#4.반복
for epoch in range(10):
#4-1.예측값설정
    y_pred = w * x + b
#4-2.손실함수
    loss = torch.mean((y_pred - y_true) ** 2)
#4-3.역전파
    loss.backward()
#4-4.손실함수 및 기울기 초기화(옵티마이저)
    optimizer.step()
    optimizer.zero_grad()
#4-5.출력
    print(f"epoch:{epoch+1:2d} | w:{w.item():.4f} | b:{b.item():.4f} | loss:{loss.item():.4f}")