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
from utils import overlay_emoji as oe


def main(checkpoint_models, images_list):

    cap = cv2.VideoCapture(0)

    os.makedirs("images/outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)

    # For recording video
    frame_width = int(760)
    frame_height = int(240)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 4, (frame_width, frame_height))

    for i in range(len(checkpoint_models)):
        transformer.load_state_dict(torch.load(checkpoint_models[i], map_location=torch.device('cpu')))
        transformer.eval()

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (380, 240))
            overlay_painting = cv2.imread(images_list[i])
            overlay_painting = cv2.resize(overlay_painting,(130,90))
            background = oe(frame.copy(), overlay_painting, 250, 150)
            if ret:
                # Prepare input frame
                image_tensor = Variable(transform(frame)).to(device).unsqueeze(0)
                # Stylize image
                with torch.no_grad():
                    stylized_image = transformer(image_tensor)
                # Add to frames
                output_img = cv2.cvtColor(deprocess(stylized_image), cv2.COLOR_BGR2RGB)

                background = (background - np.min(background)) / (
                        np.max(background) - np.min(background)
                )  # this set the range from 0 till 1
                background = (background * 255).astype(np.uint8)
                vis = np.concatenate((background, output_img), axis=1)

                frame_normed = np.array(vis, np.uint8)

                cv2.imshow("image", frame_normed)
                out.write(frame_normed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    checkpoint_model = ["checkpoints/mosaic_8000.pth", "checkpoints/shakuntala_4000.pth",
                        "checkpoints/starry_night_4000.pth", "checkpoints/hokage_4000.pth",
                        "checkpoints/ganesh_2000.pth"]
    images_list = ["src/images/styles/mosaic.jpg", "src/images/styles/shakuntala.jpg",
                   "src/images/styles/starry_night.jpg", "src/images/styles/hokage.jpg",
                   "src/images/styles/ganesh.jpg"]
    main(checkpoint_model, images_list)
