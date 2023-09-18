import torch

def evaluate(model, test_loader,train_loader):
 correct = 0
 total = 0
 with torch.no_grad():
   
   for data in train_loader:
     images, labels = data
     outputs = model(images)
     _, predicted = torch.max(outputs.data, 1)
     total += labels.size(0)
     correct += (predicted ==labels).sum().item()
   accuracy = 100 * correct / total
   print('Accuracy of the network on the 10000 train images: %d %%' % accuracy)

   correct = 0
   total = 0
   for data in test_loader:
     images, labels = data
     outputs = model(images)
     _, predicted = torch.max(outputs.data, 1)
     total += labels.size(0)
     correct += (predicted ==labels).sum().item()
 accuracy = 100 * correct / total
 print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))