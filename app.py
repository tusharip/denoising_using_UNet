

import torch
from model import UNet
import numpy as np
import os
import flask
from flask import Flask,request,jsonify,url_for,render_template
import cv2
import base64

torch.hub.download_url_to_file('https://www.dropbox.com/s/fvwei51fgepl9zu/parameters.pt?dl=1', './weight.pt')
weights = './model.pt'

imgsz   = 512

device = torch.device('cpu')

# torch.hub.download_url_to_file('https://www.dropbox.com/s/a1puv47v6tmrk6j/weights.pt?dl=1', weights)

# Load model
model = UNet(1, 1)

# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('Total Parameters: ', total_params)
# print('Trainable Parameters: ', trainable_params)
model.load_state_dict(torch.load(weights))
params=sum([p.numel() for p in model.parameters() if p.requires_grad ])
print(params)


model.to(device).eval()
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = "./static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def predict(inp,path):
    with torch.no_grad():
        img1=inp
        img=torch.tensor([[img1]],dtype=torch.float32)
        print(img.shape)
        #img=img.permute((0,3,1,2))
        out=model(img)
        print("predicteddddddddddddddddddddddddddd")
        print(out.shape)

        img=out[0,0].cpu().detach().numpy()
        print(img.shape)

        cv2.imwrite(path,img)
        print("file updated")
    return




@app.route('/',methods=['GET','POST'])
def home():
    if flask.request.method =="GET":
        return render_template("index.html")
    else:
        f = request.files["image"]
        fmat =f.filename.split('.')[-1]
        path = f'./static/img.{fmat}'

        f.save(path)

        input_img=cv2.imread(path,0)
        input_img=cv2.resize(input_img,(512,512),interpolation=cv2.INTER_CUBIC)
        inp_path=os.path.join(app.config["UPLOAD_FOLDER"],"inp.png")
        cv2.imwrite(inp_path,input_img)

        pred_path=f'./static/pred.png'
        predict(input_img,pred_path)
        return render_template("upload.html",img1=inp_path,img2=pred_path)


if __name__ == "__main__":
    print("app started")
    app.run(debug=True)
    pass
