import gradio as gr
import cv2
import utils
import numpy as np

left_image = None
right_image = None
result_image = None

def read_left_img(image):
    global left_image
    left_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    

def read_right_img(image):
    global right_image
    right_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)


def display_image(choice):
    if choice == 'Grayscale':
        return result_image
    else:
        return cv2.applyColorMap(result_image, cv2.COLORMAP_JET)


def identity(resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, delate, erode, display_mode):
    p1 = p1 * block_size ** 2
    p2 = p2 * block_size ** 2
    global result_image
    result_image = utils.compute(left_image, right_image, resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, delate, erode)
    return display_image(display_mode)


with gr.Blocks() as demo:
    with gr.Accordion("Upload Image"):
        with gr.Row():
            left_image = gr.Image(label="left_image")
            right_image = gr.Image(label="right_image")
            left_image.upload(read_left_img, left_image)
            right_image.upload(read_right_img, right_image)
        with gr.Row():
            resize = gr.Slider(label="Resize", minimum=0.1, maximum=1, step=0.1)
            preprocesssing = gr.CheckboxGroup(["Histogram Equalization (CLAHE)", "Median Blur", "Edge Detection (Canny)"], label="Preprocceing")

    with gr.Row():
        with gr.Accordion("Hyper Paraneter"):
            with gr.Row():
                min_disparity = gr.Slider(label="min_disparity", minimum=0, maximum=64, step=1)
                num_disparities = gr.Slider(label="num_disparities", minimum=16, maximum=16*16, step=16)
            with gr.Row():
                block_size = gr.Slider(label="block_size", minimum=3, maximum=21, step=2)
                uniqueness_ratio = gr.Slider(label="uniqueness_ratio", minimum=1, maximum=15, step=1)
            with gr.Row():
                speckle_window_size = gr.Slider(label="speckle_window_size", minimum=0, maximum=200, step=1)
                speckle_range = gr.Slider(label="speckle_range", minimum=1, maximum=10, step=1)
            with gr.Row():
                p1 = gr.Slider(label="p1", minimum=8, maximum=64, step=8)
                p2 = gr.Slider(label="p2", minimum=8, maximum=64, step=8)
            with gr.Row():
                dilate = gr.Slider(label="dilate", minimum=1, maximum=7, step=2)
                erode = gr.Slider(label="erode", minimum=1, maximum=7, step=2)
            
        with gr.Accordion("Results"):
            display_mode = gr.Radio(["Grayscale", "Color"], label="Display Mode", value="Grayscale")
            results = gr.Image(show_download_button=True)
    resize.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    preprocesssing.select(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    min_disparity.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    num_disparities.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    block_size.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    uniqueness_ratio.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    speckle_window_size.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    speckle_range.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    p1.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    p2.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    dilate.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    erode.release(identity, inputs=[resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, dilate, erode, display_mode], outputs=[results])
    display_mode.select(display_image, inputs=[display_mode], outputs=[results])

demo.launch()