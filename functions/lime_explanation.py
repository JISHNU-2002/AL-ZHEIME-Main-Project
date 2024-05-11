from lime import lime_image
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries

def generate_lime_explanation(model, image):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    lime_explanation, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

    # Convert Lime explanation to RGB format
    lime_explanation_rgb = gray2rgb(lime_explanation)

    # Overlay Lime explanation on the original image using the mask
    marked_explanation = mark_boundaries(image[0], mask)

    return marked_explanation
