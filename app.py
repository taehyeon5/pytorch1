#day1(mini-batch && matplotlib)
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
#matplotlib 라이브러리 import하기
import matplotlib.pyplot as plt

x = torch.tensor([[1.0],[2.0],[3.0]])
y_true = torch.tensor([[2.0],[4.0],[6.0]])

tensordataset = TensorDataset(x,y_true)
dataloader = DataLoader(tensordataset,batch_size=2,shuffle=True)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([[0.1]]))
        self.b = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self,x):
        return x @ self.w + self.b    
    
model = LinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

#Epoch별 평균 loss 저장할 리스트
loss_list = []

for epoch in range(10):
#한 Epoch 동안의 누적 loss(2번의 loss의 합)
    epoch_loss = 0

    for batch_x,batch_y in dataloader:
        y_pred = model(batch_x)
        loss=torch.mean((y_pred - batch_y) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
#batch별 loss를 누적
        epoch_loss += loss.item()
#epoch별 loss 평균
    avg_loss = epoch_loss / len(dataloader)
#list에 추가
    loss_list.append(avg_loss)
#6-7.avg_loss 출력
    print(f"epoch:{epoch+1:2d} | w:{model.w.item():.4f} | b:{model.b.item():.4f} | loss:{avg_loss:.4f}")
    
#그래프그리기
plt.plot(range(1,len(loss_list)+1),loss_list) #X축 = Epoch 번호, Y축 = Loss 값들
plt.xlabel('Epoch') #x축 이름
plt.ylabel('Loss') #y축 이름
plt.title('Training Loss Curve') #제목
plt.grid() #격자표기
plt.show() #그래프 창 띄움
