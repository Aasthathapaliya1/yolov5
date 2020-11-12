import torch



def convert_model(model, save_path,input=torch.rand(size=(1, 3, 112, 112)).clone().detach().requires_grad_(True)):
    model = torch.jit.trace(model, input)
    torch.jit.save(model, save_path)
    print("saved suceesfully")

model_path="/home/acer/treeleaf/Aastha/vixx/resources/yolov5x.pt"
model = torch.jit.load(model_path)
save_path = '/home/acer/treeleaf/Aastha/yolov5/experiments/models/model.pt'
convert_model(model,save_path)
