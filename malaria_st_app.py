
import numpy as np
import pandas as pd
import streamlit as st

from cellpose import models, io
from cellpose.utils import outlines_list, masks_to_outlines
import cv2
import tifffile
from PIL import Image
import time, os
#pdf export
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import base64

import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torchvision.transforms as transforms

# from streamlit group
from load_css import local_css
local_css("style.css")

def imread(image_up):
    # ext = os.path.splitext(image_up.name)[-1]
    # if ext== '.tif' or ext=='tiff':
    #     img = tifffile.imread(image_up)
    #     return img
    # else:
    img = plt.imread(image_up)
    return img

@st.cache(show_spinner=False)
def run_segmentation(model, image, diam, channels, flow_threshold, cellprob_threshold):
    masks, flows, styles, diams = model.eval(image, 
            # batch_size = 8,
            diameter = diam, # 100
            channels = channels,
            invert = True,
            # rescale = 0.5,
            net_avg=False,
            flow_threshold = flow_threshold, # 1
            cellprob_threshold = cellprob_threshold, # -4
                            )
    return masks, flows, styles, diams

#from cellpose
# @st.cache
def show_cell_outlines(img, maski, color_mask):

    outlines = masks_to_outlines(maski)
    
    # plot the WordCloud image     
    fig, ax = plt.subplots(figsize = (8, 8))                   
    outX, outY = np.nonzero(outlines)
    imgout= img.copy()
    h = color_mask.lstrip('#')
    hex2rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    imgout[outX, outY] = hex2rgb
    # imgout[outX, outY] = np.array([255,75,75])
    ax.imshow(imgout)
    #for o in outpix:
    #    ax.plot(o[:,0], o[:,1], color=[1,0,0], lw=1)
    ax.set_title('Predicted outlines')
    ax.axis('off')
    
    return fig

@st.cache(show_spinner=False)
def get_cell_outlines(masks):
    outlines_ls = outlines_list(masks)
    return outlines_ls

def get_crop_from_outline(cell, tmp_img):
    # get xy coords
    x = cell.flatten()[::2]
    y = cell.flatten()[1::2]
    # remove cells on the border
    if 0 in x or 0 in y or tmp_img.shape[0] - 1 in y or tmp_img.shape[1] - 1 in x:
        return 0
    # mask outline
    mask = np.zeros(tmp_img.shape, dtype=np.uint8)
    channel_count = tmp_img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    # fill contour
    cv2.fillConvexPoly(mask, cell, ignore_mask_color)
    masked_image = cv2.bitwise_and(tmp_img, mask)
    # crop the box around the cell
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = masked_image[topy:bottomy+1, topx:bottomx+1,:]
    return out

@st.cache
def transform_image(arr):
    my_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        # transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
#     image = Image.open(io.BytesIO(image_bytes))
    im = Image.fromarray(arr)
    return my_transforms(im).unsqueeze(0)

class_names = ["un", 'ring', 'troph', 'shiz']

def get_prediction(arr):
    tensor = transform_image(arr)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return class_names[y_hat]

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'


st.title('P. falciparum Malaria Detection and Classification')
st.text('Segmentataion -> Single cell ROI -> Classification')

page = st.sidebar.selectbox("Choose a stain", ('Giemsa', 'Stain type 2', 'Sample images'))

st.sidebar.title("About")
st.sidebar.info(" - Segmentation: [Cellpose] (https://github.com/MouseLand/cellpose)    \n \
- Classification of ROI: pretrained convnet + fine-tuning      \n \
- Trained on Giemsa stained P. _falsiparum_     \n \
Powered by PyTorch, [Streamlit] (https://docs.streamlit.io/en/stable/api.html) ")


ls_images = []
ls_df = []
file_up = None
figs = []

if page == 'Sample images':
    img_list = ["./images/T0D2_1.tif", "./images/T14D2_2.tif", "./images/T38D2_2.tif", "./images/T48D2_2.tif" ]
    img_captions = ["After 2 hours", "After 14 hours", "After 38 hours", "After 48 hours", "<choose>"]
    st.image(img_list, caption = img_captions[:4], width = int(698/2))
    selected_image = st.selectbox("Choose a sample image to analyze", img_captions, 4)
    if selected_image != "<choose>":
        selected_image = img_captions.index(selected_image)
        file_up = img_list[selected_image]
        # st.text(file_up)
        image = tifffile.imread(file_up)

else:
    file_up = st.file_uploader("Upload an image", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True)
    if len(file_up) > 0:
        for fobj in file_up:
            image = imread(fobj)
            ls_images.append(image)

if len(ls_images) > 0:
    # @st.cache
    # image = Image.open(file_up)
    

    fig, ax = plt.subplots(figsize = (8,8))
    ax.imshow(ls_images[0])
    ax.axis("off")
    ax.set_title('Selected image')
    st.pyplot(fig)

    st.subheader('Segmentation parameters')

    diameter = st.number_input('Diameter of the cells [pix]', 0, 500, 100, 10)
    # st.write('The current number is ', diameter)

    # flow_threshold = 1
    flow_threshold = st.slider('Flow threshold (increase -> more cells)', .0, 1.1, 1.0, 0.1)
    st.write("", flow_threshold)

    # cellprob_threshold = -4
    cellprob_threshold = st.slider('Cell probability threshold (decrease -> more cells)', -6, 6, -4, 1)
    st.write("", cellprob_threshold)

    color_mask = '#000000'
    # color_mask = st.color_picker('Pick a color for cell outlines', '#000000')
    # st.write('The current color is', color_mask)

    no_diam = st.checkbox("Auto diameter computation")
    if no_diam:
        diameter = None
        st.warning("Note: it takes more time to run with this option. After the first run you will see the computed diameter. \
        Unchek the checkbox and manually put the computed diamater to speed up the app for future runs.")
    if st.button('Analyze'):
        for fidx, image in enumerate(ls_images):
            st.markdown('### Processing')
            st.write("Image #", fidx +1 , file_up[fidx].name)
        # DEFINE CELLPOSE MODEL
        # model_type='cyto' or model_type='nuclei'
            with st.spinner("Running segmentation"):
                model = models.Cellpose(gpu=False, model_type ='cyto')
                # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
                channels = [[0,0]] #* len(files) # IF YOU HAVE GRAYSCALE

                since = time.time()
                # img = io.imread(filename)
                masks, flows, styles, diam = run_segmentation(model, image, diameter, channels, 
                                                    flow_threshold, cellprob_threshold)
                st.text('Initial cell count: {} '.format(masks.max()))
                diameter = diam
                time_elapsed = time.time() - since
                st.write('Time spent on segmentation {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
                if no_diam:
                    st.write('Computed diameter: {:.1f} pix'.format(diameter) )
            # if st.button('Show results'):
                # DISPLAY segmenttaion
                # fig = show_cell_outlines(image, masks, color_mask)
                # st.pyplot(fig)
            with st.spinner("Getting single cells"):
                outlines_ls = get_cell_outlines(masks)

            
            with st.spinner("Loading Model"):
                device = torch.device('cpu')
                # Load cnn model
                PATH = "model.pth"
                model = torch.load(PATH, map_location = device)
                model.eval()
            
            size_thres = diameter*0.8
            tmp_img = image.copy()
            d_results = {"un": [],
                        "ring": [],
                        "troph": [],
                        "shiz": []
                        }
            with st.spinner("Running inference..."):
            # st.text("Running inference ...")
                since = time.time()
                for idx, cell in enumerate(outlines_ls[:]):
                    # get crop
                    out = get_crop_from_outline(cell, tmp_img)
                    # remove cells at the edge
                    if not isinstance(out, int):
                        # size threshold
                        if out.shape[0] < size_thres or out.shape[1] < size_thres:
                            continue
                        # predict the stage of the cell p
                        stage = get_prediction(out)
                        d_results[stage].append(idx)
    
            time_elapsed = time.time() - since
            st.write('time spent on classification {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


            with st.spinner("Plotting results"):
                t = "<div> <span class='highlight yellow'> Ring </span> \
                        <span class='highlight magenta'> Troph </span>      \
                        <span class='highlight cyan'> Shiz </span>      </div>"
                st.markdown(t, unsafe_allow_html=True)

                colors_stage = { "un": '#000000', "ring": "#ffc20a", 
                    "troph": "#40b0a6", "shiz": "#d35fb7" }
                fig, ax = plt.subplots(figsize = (8,8))
                # yellow: ring; magenta: troph; cyan: shiz
                ax.imshow(image)

                for k in class_names:
                    for cell in d_results[k]:
                        coord = outlines_ls[cell]
                        ax.plot(coord[:,0], coord[:,1], c = colors_stage[k], lw=1)
                ax.set_title(file_up[fidx].name)
                ax.axis('off')
                st.pyplot(fig)
                figs.append(fig)

                out_stat = []
                for k in class_names:
                    out_stat.append(len(d_results[k]))
                df = pd.DataFrame({'image': file_up[fidx].name, "all_cells":np.sum(out_stat), 
                            "R":out_stat[1], "T":out_stat[2], "S":out_stat[3], 
                            "Parasitemia": 1 - out_stat[0]/np.sum(out_stat)}, index = [0])
                st.write(df)
                ls_df.append(df)
        # export_as_pdf = st.button("Export Report")

        # if export_as_pdf:
        pdf = FPDF()
        for idx, fig in enumerate(figs):
            pdf.add_page()
            pdf.set_font('Arial', '', 10)
            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name)
                    pdf.image(tmpfile.name, 5, 25, 200, 200)
            pdf.multi_cell(0, 7, txt = ls_df[idx].iloc[0].to_string(), border = 0, align = 'L')
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "mds-out")
        st.markdown(html, unsafe_allow_html=True)