import torch
import torch.nn as nn
import matplotlib.pyplot as plt

house_size = torch.randn(100 , 1) * 5 +10
house_prices = house_size * 2 + torch.randn (100 , 1) *5 + 100
class PricePrediction(nn.Module) :
  def __init__(self) :
    super().__init__()
    self.Linear = nn.Linear(1,1)
  def forward (self , x) :
    return self.Linear(x)


model = PricePrediction()
criterion = nn.MSELoss ()
optimizer = torch.optim.Adam(model.parameters() , lr = 0.01)

for epoch in range (100) :
  pred = model (house_size)
  loss = criterion(pred , house_prices)
  optimizer.zero_grad ()
  loss.backward ()
  optimizer.step ()

  if epoch % 20 == 0 :
    print (f'epoch : {epoch} loss : {loss.item()}')
