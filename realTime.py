"""
@filename: realTime.py
@Author: Pravesh Bawangade
"""
from src.models import TransformerNet
from src.utils import *
import torch
from torch.autograd import Variable
import os
import cv2


def main(checkpoint_models):

    cap = cv2.VideoCapture("video/input.mp4")

    os.makedirs("images/outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)

    for checkpoint_model in checkpoint_models:
        transformer.load_state_dict(torch.load(checkpoint_model, map_location=torch.device('cpu')))
        transformer.eval()

        # For recording video
        # frame_width = int(760)
        # frame_height = int(240)
        # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame_width, frame_height))
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (320, 240))
            if ret:
                # Prepare input frame
                image_tensor = Variable(transform(frame)).to(device).unsqueeze(0)
                # Stylize image
                with torch.no_grad():
                    stylized_image = transformer(image_tensor)
                # Add to frames
                output_img = cv2.cvtColor(deprocess(stylized_image), cv2.COLOR_BGR2RGB)

                cv2.imshow("image", output_img)
                # out.write(output_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    checkpoint_model = ["checkpoints/mosaic_8000.pth", "checkpoints/shakuntala_4000.pth",
                        "checkpoints/starry_night_4000.pth", "checkpoints/hokage_4000.pth",
                        "checkpoints/ganesh_2000.pth"]
    main(checkpoint_model)
