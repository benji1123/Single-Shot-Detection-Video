import torch                                   # to  perform rapid NN (gradient) alculation with a "Dynamic Computational Graph"
from torch.autograd import Variable            # to convert torch-tensor into torch-objects [Tensor, Gradient]
import cv2                                     # to draw rectangles on frame
from data import BaseTransform, VOC_CLASSES as labelmap         
      # BaseTransform to configure image to be NN input
      # To create a dictionary that maps our classes to #s
      
from ssd import build_ssd                      # to construct NN
import imageio                                 # to perform processing on video with detect()

from tkinter import Tk                         # file selection dialog-box func
from tkinter.filedialog import askopenfilename


# Input video-frame to NN for classification | return video-frame with OBJ highlighted.........
def detect(frame, net, transform):
    h,w = frame.shape[:2];                      # purpose?
    
    # NN processes frame (input) in specific format......
    frame_t = transform(frame)[0];                       # get frame in req. dimension/color
    x = torch.from_numpy(frame_t).permute(2,0,1);        # get frame in tensor-form () & grb-color notation
    x = Variable(x.unsqueeze(0));       
        # convert frame to NN's required (torch) "Variable" form 
        # add required "Batch" element
    
    # NN-Output is Coodinate of Detected OBJ in curr.frame......
    y = net(x);                         
    detections = y.data;                # = [Batch, No.Detected Class-Types, No.Class-Occurences, {score,x0,y0,x1,y1}] 
    scale = torch.Tensor([w,h, w,h]);   # detection-positons must be on 0-1 scale | this makes it happen in Ln 47

    # NN Output is Mined for Valid Detections ......
    for i in range(detections.size(1)) :                 
        j=0;    # occurence of class
        while detections[0,i,j,0] > 0.6 :                                # Extract detected-OBJ as TENSOR | if score > 0.6
            coord = (detections[0, i, j, 1:] * scale).numpy();           # Set validate OB to Numpy.array()    
            
            # NN-Detections are Visualized on Video-Frame ...... 
            cv2.rectangle(frame, (int(coord[0]),int(coord[1])), (int(coord[2]),int(coord[3])), (255,0,0), 2);     # draw RECT around NN-detection 
            #[medium, (coord1), (coord2), color, thickness]    
            cv2.putText(frame, labelmap[i-1], (0,int(coord[3])), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 5, cv2.LINE_AA);    # label NN-detection
            # [medium, text-content, text position (upper-left), font, text-size, color, text-thickness, text-style (continuous)]
            
            j += 1; # 'j' iterates the occurences of i'th detected class
    return frame;



# THE ARGUMENTS OF detect()---Net, Transform, Frame---ARE GENERATED ......
    

# The SSD Neural Network is Created in 2 Lines of Code
net = build_ssd('test');                                    # 'test' (as opposed to 'train') is the arg  | our SSD is "trained" (see below)
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).
 # connecting NN to pre-traind weights......
    # torch => loads Tensor of weights
    # load_state_dict => attributes weights to NN
    # everything else => formalities

transform = BaseTransform(net.size, (104/256.0,117/256.0,123/256.0));       # this OBJ holds: standard configuration for NN video-frame-input | size && color-scale


# Input-Video frames are Classifed one-by-one | write new Video with Targets Highlighted......

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

reader = imageio.get_reader(filename);
fps = reader.get_meta_data()['fps'];
writer = imageio.get_writer('ben.mp4', fps = fps);

for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform);         #targets are highlighted in each frame of 'reader'
    writer.append_data(frame);
    print(i);
print("detection complete | see the newly-created output-video in your working directory!");
writer.close();