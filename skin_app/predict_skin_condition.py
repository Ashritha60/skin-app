import os
import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, StringVar, Toplevel, Frame, Canvas, PhotoImage
from tkinter.ttk import Style
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('densenet_model.keras')

categories = ['acne', 'dark spots', 'normal skin', 'puffy eyes', 'wrinkles']

# Global variables
img_path = None
preview_img = None

# Treatment suggestions
def suggest_treatment(condition):
    treatments = {
        'acne': [
            "Use a salicylic acid-based cleanser twice a day.",
            "Apply a benzoyl peroxide cream to affected areas.",
            "Moisturize with a non-comedogenic, oil-free lotion.",
            "Always wear sunscreen (SPF 30+) during the day."
        ],
        'dark spots': [
            "Use a gentle exfoliating cleanser with glycolic acid.",
            "Apply a Vitamin C serum daily to brighten skin tone.",
            "Protect your skin with a broad-spectrum sunscreen.",
            "Consider professional treatments like chemical peels."
        ],
        'normal skin': [
            "Cleanse with a gentle hydrating cleanser twice a day.",
            "Use a lightweight moisturizer with hyaluronic acid.",
            "Apply sunscreen with at least SPF 30 in the morning.",
            "Maintain a healthy diet and stay hydrated."
        ],
        'puffy eyes': [
            "Apply a cold compress for a few minutes in the morning.",
            "Use an under-eye cream with caffeine or retinol.",
            "Avoid excess salt and stay hydrated throughout the day.",
            "Sleep with your head slightly elevated."
        ],
        'wrinkles': [
            "Cleanse with a mild, hydrating cleanser.",
            "Apply a retinol-based serum at night.",
            "Use a peptide-rich moisturizer to boost skin elasticity.",
            "Always wear sunscreen to prevent further aging."
        ],
    }
    return treatments.get(condition, ["Consult a dermatologist for personalized advice."])

# Predict the skin condition
def predict_skin_condition(model, img_path, categories):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return categories[class_index]

# Introduction screen
def intro_screen():
    intro_frame = Frame(root, bg=palette["background"])
    intro_frame.pack(fill="both", expand=True)

    Label(
        intro_frame,
        text="✨ Welcome to the Skin Condition Analyzer ✨",
        font=("Arial", 30, "bold"),
        fg=palette["primary_text"],
        bg=palette["background"],
        pady=40,
        anchor="center"
    ).pack()

    Label(
        intro_frame,
        text=( 
            "Our Vision: To empower you with accurate insights about your skin health "
            "and provide personalized skincare recommendations for a radiant you.\n\n"
            "How to Use: Simply upload a picture of your face, and our AI will analyze your "
            "skin condition. You'll receive tailored treatment suggestions to enhance your skincare routine."
        ),
        font=("Arial", 18),
        fg=palette["secondary_text"],
        bg=palette["background"],
        wraplength=900,
        justify="center",
        pady=20,
        anchor="center"
    ).pack()

    Button(
        intro_frame,
        text="Let's Get Started!",
        font=("Arial", 16, "bold"),
        bg=palette["button_bg"],
        fg=palette["button_fg"],
        command=lambda: transition_to(image_upload_screen, intro_frame),
        width=25,
        pady=10
    ).pack(pady=40)

# Image upload and processing screen
def image_upload_screen():
    upload_frame = Frame(root, bg=palette["background"])
    upload_frame.pack(fill="both", expand=True)

    Label(
        upload_frame,
        text="Upload an Image for Analysis",
        font=("Arial", 24, "bold"),
        fg=palette["primary_text"],
        bg=palette["background"],
        pady=40,
        anchor="center"
    ).pack()

    # Image preview canvas
    canvas = Canvas(upload_frame, width=250, height=250, bg=palette["secondary_bg"])
    canvas.pack(pady=20)

    def upload_image():
        global img_path, preview_img
        img_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if img_path:
            img = Image.open(img_path)
            img.thumbnail((250, 250))
            preview_img = ImageTk.PhotoImage(img)
            canvas.create_image(125, 125, image=preview_img)

    Button(
        upload_frame,
        text="Upload Image",
        font=("Arial", 16, "bold"),
        bg=palette["button_bg"],
        fg=palette["button_fg"],
        command=upload_image,
        width=20,
        pady=10
    ).pack(pady=20)

    Button(
        upload_frame,
        text="Process Image",
        font=("Arial", 16, "bold"),
        bg=palette["button_bg"],
        fg=palette["button_fg"],
        command=lambda: process_image(upload_frame),
        width=20,
        pady=10
    ).pack(pady=20)

    Button(
        upload_frame,
        text="Go Back",
        font=("Arial", 16, "bold"),
        bg=palette["button_bg"],
        fg=palette["button_fg"],
        command=lambda: transition_to(intro_screen, upload_frame),
        width=20,
        pady=10
    ).pack(pady=20)

# Process and display results
def process_image(current_frame):
    if not img_path:
        return

    condition = predict_skin_condition(model, img_path, categories)
    treatment = suggest_treatment(condition)

    def display_results():
        results_frame = Frame(root, bg=palette["background"])
        results_frame.pack(fill="both", expand=True)

        Label(
            results_frame,
            text=f"Predicted Skin Condition: {condition}",
            font=("Arial", 24, "bold"),
            fg=palette["highlight_fg"],
            bg=palette["highlight_bg"],
            relief="raised",
            padx=20,
            pady=20,
            width=50,
            anchor="center"
        ).pack(pady=40)

        treatment_box = Frame(results_frame, bg=palette["secondary_bg"], padx=20, pady=20)
        treatment_box.pack(pady=20)

        Label(
            treatment_box,
            text="Treatment Suggestions:",
            font=("Arial", 18, "bold"),
            fg=palette["primary_text"],
            bg=palette["secondary_bg"],
            anchor="center"
        ).pack()

        for step in treatment:
            Label(
                treatment_box,
                text=f"• {step}",
                font=("Arial", 16),
                fg=palette["primary_text"],
                bg=palette["secondary_bg"],
                wraplength=700,
                justify="center",
                anchor="center"
            ).pack(pady=10)

        Button(
            results_frame,
            text="Try Another Image",
            font=("Arial", 16, "bold"),
            bg=palette["button_bg"],
            fg=palette["button_fg"],
            command=lambda: transition_to(image_upload_screen, results_frame),
            width=20,
            pady=10
        ).pack(pady=40)

        Button(
            results_frame,
            text="Go Back",
            font=("Arial", 16, "bold"),
            bg=palette["button_bg"],
            fg=palette["button_fg"],
            command=lambda: transition_to(intro_screen, results_frame),
            width=20,
            pady=10
        ).pack(pady=20)

    transition_to(display_results, current_frame)

# Transition between screens
def transition_to(new_screen, current_frame):
    current_frame.destroy()
    new_screen()

# Define a color palette with soft blues and complementary shades
palette = {
    "background": "#d9ebf2",  # Soft Light Blue
    "primary_text": "#2c3e50",  # Dark Blue
    "secondary_text": "#4f9ab3",  # Light Blue
    "button_bg": "#66c2ff",  # Sky Blue
    "button_fg": "#ffffff",  # White
    "secondary_bg": "#b3d9f7",  # Pale Blue
    "highlight_bg": "#a3c6ea",  # Soft Blue
    "highlight_fg": "#1f3a52"  # Deep Blue
}

# Initialize the main window
root = Tk()
root.title("Skin Condition Analyzer")
root.state("zoomed")

# Start with the intro screen
intro_screen()

# Run the main loop
root.mainloop()
