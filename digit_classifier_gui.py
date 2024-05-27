import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)
    
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)
    
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

model = Network()




class DigitClassifierGUI(tk.Tk):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.canvas_size = 300
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        
        self.init_ui()

    def init_ui(self):
        self.canvas = self.create_canvas()
        self.classify_button = self.create_classify_button()
        self.clear_button = self.create_clear_button()
        self.prediction_text = tk.StringVar()
        self.prediction_label = tk.Label(self, textvariable=self.prediction_text)

      
        self.canvas.grid(row=0, column=0, pady=2, sticky='W')
        self.classify_button.grid(row=0, column=1, pady=2, padx=2)
        self.clear_button.grid(row=0, column=2, pady=2, padx=2)
        self.prediction_label.grid(row=1, column=0, pady=2, padx=2)

    def create_canvas(self):
        canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg='white', cursor='cross')
        canvas.bind("<Button-1>", self.save_last_point)
        canvas.bind("<B1-Motion>", self.draw_lines)
        return canvas

    def create_classify_button(self):
        return tk.Button(self, text="Classify", command=self.classify_handwriting)

    def create_clear_button(self):
        return tk.Button(self, text="Clear", command=self.clear_all)

    def clear_all(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def classify_handwriting(self):
        img = self.image.resize((28, 28))
        img = 255 - np.array(img)  
        img = img / 255.0 
        img = img.reshape(1,28,28,1)
        img = torch.from_numpy(img).float()
        with torch.no_grad():
            output = self.model(img)
        ps = torch.exp(output)
        probab = list(ps.numpy()[0])
        _, predicted = torch.max(output, 1)
        pred = predicted.item()
        confidence = probab[pred]
        self.prediction_text.set('Prediction: {}'.format(pred, confidence * 100))

    def save_last_point(self, event):
        self.lastx, self.lasty = event.x, event.y

    def draw_lines(self, event):
        self.canvas.create_line((self.lastx, self.lasty, event.x, event.y), fill='black', width=8, capstyle='round', smooth=True)
        self.draw.line((self.lastx, self.lasty, event.x, event.y), fill='black', width=8)
        self.lastx, self.lasty = event.x, event.y


model = Network()  
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()
app = DigitClassifierGUI(model)
app.mainloop()





# from flask import Flask, render_template, request, jsonify
# import base64
# from PIL import Image
# import io
# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F

# app = Flask(__name__)

# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64,10)
    
#     def forward(self,x):
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.log_softmax(self.fc3(x), dim=1)
#         return x

# model = Network()
# model.load_state_dict(torch.load("trained_model.pth"))
# model.eval()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/classify', methods=['POST'])
# def classify():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'})
    
#     img_file = request.files['image']
#     if img_file.filename == '':
#         return jsonify({'error': 'No image selected'})
    
#     img_bytes = img_file.read()
#     img = Image.open(io.BytesIO(img_bytes))
#     img = img.convert('L').resize((28, 28))  # Convert to grayscale and resize to 28x28
#     img_np = np.array(img) / 255.0  # Normalize pixel values
#     img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)  # Convert to tensor

#     with torch.no_grad():
#         output = model(img_tensor)
    
#     _, predicted = torch.max(output, 1)
#     prediction = predicted.item()
    
#     return jsonify({'prediction': prediction})

# if __name__ == '__main__':
#     app.run(debug=True)
