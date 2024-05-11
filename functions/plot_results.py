import os
import cv2
import matplotlib.pyplot as plt

def plot_results(image, lime_explanation, grad_cam_result, filename, predicted_class, app):
    # Construct the file path using Flask's static variable
    save_path = os.path.join(app.static_folder, 'plot_results.png')

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image[0])
    ax[0].set_title(f'Original Image')

    ax[1].imshow(lime_explanation)
    ax[1].set_title(f'LIME Explanation : {predicted_class}')

    ax[2].imshow(image[0])
    ax[2].imshow(cv2.resize(grad_cam_result, (image.shape[2], image.shape[1])), alpha=0.5, cmap='jet')
    ax[2].set_title(f'Grad-CAM : {predicted_class}')

    # Save the plot as an image using the constructed file path
    plt.savefig(save_path)

    # Close the plot to free up memory
    plt.close(fig)
