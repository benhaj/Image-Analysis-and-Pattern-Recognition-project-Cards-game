#!/usr/bin/env python
# coding: utf-8

# imports
import skimage.io
import matplotlib.pyplot as plt
import os
import cv2 as cv
import cv2
import numpy as np

import matplotlib.image as mpimg
import pandas as pd
from torchvision.models import resnet18
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import xgboost
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import cross_val_score
from skimage.morphology import erosion, dilation, opening, closing
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_objects 
from skimage.morphology import (disk, square, diamond)
from tqdm import tqdm


## load game
def load(game ,path=False):
    num = list(range(1,14))
    names = [str(i)+'.jpg' for i in num]
    if path:
        ic = skimage.io.imread_collection([game+nm for nm in names])
    else:
        ic = skimage.io.imread_collection(["train_games/"+game+'/'+nm for nm in names])

    round_im = skimage.io.concatenate_images(ic)
    return round_im, names


## Segmentation

def plot_contours(img, contours):    
    fig, axes = plt.subplots(1, len(contours), figsize=(30,40))
    for idx in range(len(contours)):
        mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
        out = np.zeros_like(img) # Extract out the object and place into output image
        out[mask == 255] = img[mask == 255]

        # Now crop
        (y, x, z) = np.where(mask == 255)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        out = out[topy:bottomy+1, topx:bottomx+1, :]
        img_cropped = img[topy:bottomy+1, topx:bottomx+1, :]
        axes[idx].imshow(img_cropped)

def get_contours_box(contours):
    boxes = []
        
    for idx in range(len(contours)):
        c = contours[idx].reshape(-1, 2)
        # Now crop
        (x, y) = (c[:, 1], c[:, 0])
        (miny, minx) = (int(np.rint(np.min(y))), int(np.rint(np.min(x))))
        (maxy, maxx) = (int(np.rint(np.max(y))), int(np.rint(np.max(x))))
        boxes.append([miny, maxy, minx, maxx])
        
    return boxes

def filter_overlapping(contours, boxes):
    flags = [True] * len(contours)
    
    for i in range(len(contours)):
        
        if not flags[i]:
            continue
            
        curr_box = boxes[i]
        
        if ((curr_box[0] < 100) and (curr_box[2] < 500)) or ((curr_box[0] < 500) and (curr_box[2] > 4000)):
            flags[i] = False
            continue
        
        for j in range(i, len(contours)):
            
            new_box = boxes[j]
            
            enclosed = ((curr_box[0] < new_box[0]) &
                       (curr_box[1] > new_box[1]) &
                       (curr_box[2] < new_box[2]) &
                       (curr_box[3] > new_box[3]))
            
            if enclosed:
                flags[j] = False
                
    contours = [contours[i] for i in range(len(contours)) if flags[i]]
    boxes = [boxes[i] for i in range(len(contours)) if flags[i]]
    return contours, boxes

def plot_image(img, cmap='gray',):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap=cmap)
    plt.show()

def distance(p1,p2):
    result= ((((p2[0] - p1[0])**2) + ((p2[1]-p1[1])**2))**0.5)
    return result

def assign_cards_dealer(cards, dealer):
    '''
    returns :
        ordered cards [Player1_card, Player2_card, Player3_card, Player4_card]
            each PlayerX_card is a tuple with (((x_center,y_center),(width,height),angle_rot),card_contour)
        dealer_player = indx in [1,2,3,4] indicating the player
    '''
    card_centers = [(cv.minAreaRect(card),card) for card in cards]
    sorted_cards = sorted(card_centers)
    ordered = []
    if (sorted_cards[1][0][0][1]<sorted_cards[2][0][0][1]):
        ordered= [sorted_cards[2],sorted_cards[3],sorted_cards[1],sorted_cards[0]]
    else:
        ordered= [sorted_cards[1],sorted_cards[3],sorted_cards[2],sorted_cards[0]]

    dealer_rect = cv.minAreaRect(dealer)
    distances_to_dealer = [distance(dealer_rect[0],sorted_card[0][0]) for sorted_card in ordered]
    dealer_player = np.argmin(distances_to_dealer)+1
    return ordered,dealer_player, dealer_rect


def put_text(img,ordered,dealer_player,dealer_rect):
    # put dealer text (x_center - width/2 , y_center - height/2)
    dealer_text_org = (int(dealer_rect[0][0])-250 , int(dealer_rect[0][1]-dealer_rect[1][1]/2)-100)
    result = cv.putText(img.copy(), 'Dealer', dealer_text_org, cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), thickness=7)
    
    # put player 1 text
    player1_text_org = (int(ordered[0][0][0][0]-ordered[0][0][1][1]/2)-50,int(ordered[0][0][0][1]+ordered[0][0][1][0]/2)+200)
    result = cv.putText(result, 'Player 1', player1_text_org, cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), thickness=7)

    # put player 3 text
    player3_text_org = (int(ordered[2][0][0][0]-ordered[2][0][1][1]/2)-50,int(ordered[2][0][0][1]-ordered[2][0][1][0]/2)-50)
    result = cv.putText(result, 'Player 3', player3_text_org, cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), thickness=7)
    
    # put player 2 text
    player2_text_org = (int(ordered[1][0][0][0]-ordered[1][0][1][1]/2)-50,int(ordered[1][0][0][1]+ordered[1][0][1][0]/2)+100)
    result = cv.putText(result, 'Player 2', player2_text_org, cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), thickness=7)

    # put player 4 text
    player4_text_org = (int(ordered[3][0][0][0]-ordered[3][0][1][1]/2)+100,int(ordered[3][0][0][1]+ordered[3][0][1][0]/2)+200)
    result = cv.putText(result, 'Player 4', player4_text_org, cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), thickness=7)
    
    return result


def preprocess_image(img):
    # remove unwanted borders
    crop = img[150:-270,30:-30,:]
    # convert to graylevel
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # apply adaptive thresholding (since parts of same images doesn't have same luminosity/contrast )
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 5)
    # black background
    result = 255 - thresh
    return result , crop

def seg_pipe_final(img):
    drawn_contours = []
    kernel = np.ones((3,3),np.uint8)
    black_white_im, cropped_img = preprocess_image(img)
    opening = cv.morphologyEx(black_white_im, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(5,500))
    kernel3 = cv.getStructuringElement(cv.MORPH_RECT,(500,5))
    closing_edge_1 = cv.morphologyEx(closing[:, :5], cv.MORPH_CLOSE, kernel2)
    closing_edge_2 = cv.morphologyEx(closing[:, -5:], cv.MORPH_CLOSE, kernel2)
    closing_edge_3 = cv.morphologyEx(closing[:5,:], cv.MORPH_CLOSE, kernel3)
    closing_edge_4 = cv.morphologyEx(closing[-5:,:], cv.MORPH_CLOSE, kernel3)
    closing[:, :5] = closing_edge_1
    closing[:, -5:] = closing_edge_2
    closing[:5, :] = closing_edge_3
    closing[-5:, :] = closing_edge_4
    contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda contour: cv.contourArea(contour), reverse= True)
    boxes = get_contours_box(contours)
    contours, boxes = filter_overlapping(contours, boxes)
    contours = contours[:5]
    boxes = boxes[:5]
    
    def green_ratio(box):
        segment_cropped = cropped_img[box[2]:box[3]+1, box[0]:box[1]+1, :]
        r_med = np.median(segment_cropped[:, :, 0])
        g_med = np.median(segment_cropped[:, :, 1])
        b_med = np.median(segment_cropped[:, :, 2])
    
        return 2 * g_med / (r_med + b_med)

    medians = [green_ratio(box) for box in boxes]
    dealer_idx = np.argmax(medians)
    dealer_contour = contours.pop(dealer_idx)
    dealer_box = boxes.pop(dealer_idx)
    contours.append(dealer_contour)
    boxes.append(dealer_box)

    
    return cropped_img, contours, boxes
    
def plot_bounding_boxes(cropped_img,contours, boxes):
    im = cropped_img.copy()
    for c, b in zip(contours, boxes):
        x,y,w,h = b[0], b[2], b[1] - b[0], b[3] - b[2]
        cv.rectangle(im,(x,y),(x+w,y+h),(0,0,255),10)
        
    return im


def detect_red_or_black(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    mask1 = cv.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv.inRange(img_hsv, (175,50,20), (180,255,255))

    ## Merge the mask and crop the red regions
    mask_red = cv.bitwise_or(mask1, mask2)
    
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)    
    ## Gen lower mask (0-5) and upper mask (175-180) of BLACK
    mask_black = cv.inRange(img_hsv, (0, 0, 0), (180, 255, 30))
    
    if mask_red.sum() > mask_black.sum():
#         img_hsv = cv.cvtColor(third_card, cv.COLOR_RGB2HSV)
#         ## Gen lower mask (0-5) and upper mask (175-180) of RED
#         mask1 = cv.inRange(img_hsv, (0,50,20), (5,255,255))
#         mask2 = cv.inRange(img_hsv, (125,50,20), (180,255,255))

#         ## Merge the mask and crop the red regions
#         mask_red = cv.bitwise_or(mask1, mask2)
        return mask_red, 'red'
    
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)    
    ## Gen lower mask (0-5) and upper mask (175-180) of BLACK
    mask_black = cv.inRange(img_hsv, (0, 0, 0), (180, 255, 80))
    
    return mask_black, 'black'



#Plot overlay game segmentation and Extract cards number and suite :
def extract_cards_suites_and_plot_overlay_rounds(game,show=False):
    ## to store result: 
    data_df = pd.DataFrame(columns=['game', 'P1_suite','P1_number','P1_color','P2_suite','P2_number','P2_color',
                           'P3_suite','P3_number','P3_color','P4_suite','P4_number','P4_color','D'], dtype=object)
    g_round_im , g_round_num = load(game)
    print(game)
    for img in g_round_im:
        
        # segmentation
        cropped_img, contours, boxes = seg_pipe_final(img)
        cards = contours[:4]
        dealer = contours[4]
        
        # plot overlay
        bounded = plot_bounding_boxes(cropped_img,contours, boxes)
        ordered,dealer_player, dealer_rect = assign_cards_dealer(cards, dealer)
        result = put_text(bounded,ordered,dealer_player,dealer_rect)
        if show:
            plot_image(result)
            plt.show()
        
        #extract cards and suites
        ordered_contours = [entry[1] for entry in ordered]
        ordered_boxes = get_contours_box(ordered_contours)
        first_card = cropped_img[ordered_boxes[0][2]:ordered_boxes[0][3],
                                 ordered_boxes[0][0]:ordered_boxes[0][1], :]
        second_card = cv.rotate(cropped_img[ordered_boxes[1][2]:ordered_boxes[1][3],
                                  ordered_boxes[1][0]:ordered_boxes[1][1], :],
                                cv.ROTATE_90_CLOCKWISE)
        third_card = cv.rotate(cropped_img[ordered_boxes[2][2]:ordered_boxes[2][3],
                                 ordered_boxes[2][0]:ordered_boxes[2][1], :],
                               cv.ROTATE_180)
        fourth_card = cv.rotate(cropped_img[ordered_boxes[3][2]:ordered_boxes[3][3],
                                  ordered_boxes[3][0]:ordered_boxes[3][1], :],
                                cv.ROTATE_90_COUNTERCLOCKWISE)
        

        first_mask, first_color = detect_red_or_black(first_card)
        second_mask, second_color = detect_red_or_black(second_card)
        third_mask, third_color = detect_red_or_black(third_card)
        fourth_mask, fourth_color = detect_red_or_black(fourth_card)
        
        first_suite,first_number = first_mask.copy()[:250,:300],first_mask.copy()[200:570,120:420]
        second_suite,second_number = second_mask.copy()[:250,:300],second_mask.copy()[200:570,120:420]
        third_suite,third_number = third_mask.copy()[:250,:300],third_mask.copy()[200:570,120:420]
        fourth_suite,fourth_number = fourth_mask.copy()[:250,:300],fourth_mask.copy()[200:570,120:420]
        
        row = {'game':game, 'P1_suite':first_suite,'P1_number':first_number,'P1_color':first_color,
               'P2_suite':second_suite,'P2_number':second_number,'P2_color':second_color,
               'P3_suite':third_suite,'P3_number':third_number,'P3_color':third_color,
               'P4_suite':fourth_suite,'P4_number':fourth_number,'P4_color':fourth_color,'D':dealer_player}
        data_df=data_df.append(row,ignore_index=True)
    return data_df


# Classifying suits
def histogram_equalization(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
    
    return img_output

def preprocess_for_segmentation(img, kernel_size=(55,55)):
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray,kernel_size,10)
    return blur

def perform_segmentation(preprocessed_img):
    flag, thresh = cv.threshold(preprocessed_img, 120, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea,reverse=True)
    return contours, hierarchy

def classify_suite(color,suite):
    img_eq = histogram_equalization(suite)
    processed_im = preprocess_for_segmentation(img_eq, kernel_size=(1,1))
    contours, _  = perform_segmentation(processed_im)
    area = cv.contourArea(contours[0])
    if color=='black':
        if area>=4900 : suite='S'
        else: suite='C'
    else:
        if area>=4065: suite='H'
        else: suite='D'
    return suite

def output_suites(dataframe):
    df=dataframe.copy()
    df["suite_p1"] = df.apply(lambda row: classify_suite(row.P1_color,row.P1_suite) ,axis=1)
    df["suite_p2"] = df.apply(lambda row: classify_suite(row.P2_color,row.P2_suite) ,axis=1)
    df["suite_p3"] = df.apply(lambda row: classify_suite(row.P3_color,row.P3_suite) ,axis=1)
    df["suite_p4"] = df.apply(lambda row: classify_suite(row.P4_color,row.P4_suite) ,axis=1)
    return df

## loading ground truth
def load_game_csv(game):
    path = f'train_games\\{game}\\{game}.csv'
    cgt = pd.read_csv(path, index_col = 0)
    cgt_rank = cgt[['P1', 'P2', 'P3', 'P4']].values
    dealer = cgt[['D']].values
    return cgt

def create_needed_df(game):
    grt=load_game_csv(game)
    df = pd.DataFrame({'P1_suite':[elem[1] for elem in grt.P1.values],
                   'P1_number':[elem[0] for elem in grt.P1.values],
                   'P2_suite':[elem[1] for elem in grt.P2.values],
                   'P2_number':[elem[0] for elem in grt.P2.values],
                   'P3_suite':[elem[1] for elem in grt.P3.values],
                   'P3_number':[elem[0] for elem in grt.P3.values],
                   'P4_suite':[elem[1] for elem in grt.P4.values],
                   'P4_number':[elem[0] for elem in grt.P4.values],
                   'D':grt.D.values})
    df['game']=game
    return df


# freeman code
def calculate_area(chain):
    area = 0
    x = 0
    y = 0
    x_coord = [None]*len(chain)
    y_coord = [None]*len(chain)
    
    for i in range(len(chain)): # Transforming the chain code into x and y coordinates to compute the area
        if chain[i] == 0:
            x -= 1
            y += 1
        elif chain[i] == 1:
            y += 1
        elif chain[i] == 2:
            x += 1
            y += 1
        elif chain[i] == 3:
            x += 1
        elif chain[i] == 4:
            x += 1
            y -= 1
        elif chain[i] == 5:
            y -= 1
        elif chain[i] == 6:
            x -= 1
            y -= 1
        elif chain[i] == 7:
            x -= 1
        else:
            x = 1000 # Big value to clearly indicate if an error occurs
            y = 1000
        x_coord[i] = x
        y_coord[i] = y
        
    a = 0    
    for j in range(len(chain)-1): # Calculation of area contributed by consecutive pixels.
        a += x_coord[j]*y_coord[j+1] - x_coord[j+1]*y_coord[j]
    a += x_coord[len(chain)-1]*y_coord[0] - x_coord[0]*y_coord[len(chain)-1]
    
    area = abs(a)/2
    
    return area


def get_chain_code(image):
    feat_vect1 = []
    feat_vect2 = []
    feat_vect3 = []
    feat_vect4 = []
    chain_area = 0
    # Erosion to get rid of small smudges off to the sides.
    erosion_number = erosion(image, selem=square(1), out=None)>0
    # Clean the images by removing image regions that are smaller than 30 pixels
    cleaned_number = remove_small_objects(erosion_number, min_size=30, connectivity=1, in_place=False)
    # Dilate the image around a square with side 1
    dilation_number = dilation(cleaned_number, selem=square(1), out=None)

    img = 255*dilation_number # Dilation step returns True/False values: have to change it back to 0 to 255
    
    start_point = (0,0)
    ## Discover the first point 
    for i, row in enumerate(img):
        for j, value in enumerate(row):
            if value == 255:
                start_point = (i, j)
                break
        else:
            continue
        break

    directions = [ 0,  1,  2,
                   7,      3,
                   6,  5,  4]
    dir2idx = dict(zip(directions, range(len(directions))))

    change_j =   [-1,  0,  1, # x or columns
                  -1,      1,
                  -1,  0,  1]

    change_i =   [-1, -1, -1, # y or rows
                   0,      0,
                   1,  1,  1]

    border = []
    chain = []
    curr_point = start_point
    for direction in directions:
        idx = dir2idx[direction]
        new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
        if img[new_point] != 0:
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break

    count = 0
    while curr_point != start_point:
        #figure direction to start search
        b_direction = (direction + 5) % 8 
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0]+change_i[idx], curr_point[1]+change_j[idx])
            if img[new_point] != 0: # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        if count == 15000: break
        count += 1

    chain_area = calculate_area(chain)
    
    d0=0
    d1=0
    d2=0
    d3=0
    d4=0
    d5=0
    d6=0
    d7=0
    
    for m in range(len(chain)): # Determines quantity of each direction
        if chain[m] == 0:
            d0 += 1
        elif chain[m] == 1:
            d1 += 1
        elif chain[m] == 2:
            d2 += 1
        elif chain[m] == 3:
            d3 += 1
        elif chain[m] == 4:
            d4 += 1
        elif chain[m] == 5:
            d5 += 1
        elif chain[m] == 6:
            d6 += 1
        elif chain[m] == 7:
            d7 += 1
        else:
            break
            
    directions_count = [d0,d1,d2,d3,d4,d5,d6,d7]
    
    # The deviation gives a quantity to how much the contours makes turns, i.e. a contour with a lot of curves
    # will have a big number, and contours like the number 1, will have smaller numbers.
    deviation = 0
    for i in range(len(chain)-1):
        deviation = deviation + abs(chain[i]-chain[i+1])
      
    return count, border, directions_count, chain_area, deviation


# Fourier Descriptor
def fourierDescriptor(contour_array):
    contour_complex = np.empty(contour_array.shape[0],dtype=complex)
    contour_complex.real = contour_array[:,0,0]
    contour_complex.imag = contour_array[:,0,1]
    fourier_result= np.fft.fft(contour_complex)
    magnitude = np.array([np.abs(fft) for fft in fourier_result][1:11])
    r = magnitude[0]
    magnitude = [m / r for m in magnitude]
    return magnitude

# train model
def train_model(cards_data,ground_data,model1,classes_map):
    
    ## prepare input MNIST
    X = []
    y = []
    
    with torch.no_grad():
        for i in tqdm(range(91)):

            data_row = cards_data.iloc[i]
            truth_row = ground_data.iloc[i]
            cards = ['P1_number', 'P2_number', 'P3_number', 'P4_number']

            for idx in cards:
                vector = cv.resize(data_row[idx], (28, 28))
                card = torch.tensor(vector, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                truth = truth_row[idx]    
                out = model1(card)
                X.append(out.numpy().reshape(-1))
                y.append(classes_map[truth])

# UNCOMMENT TO USE FREEMAN CODING
    ## prepare input (freeman)
    X_chain = []
    y_chain = []
    for i in range(91):

        data_row = cards_data.iloc[i]
        truth_row = ground_data.iloc[i]
        cards = ['P1_number', 'P2_number', 'P3_number', 'P4_number']

        for index in cards:
            img = data_row[index]
            img = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=((0, 0), (0, 0)))
            truth = truth_row[index]
            _, _, directions_count, area, indiv_deviation = get_chain_code(img)
            code_features = directions_count + [area] + [indiv_deviation]
            X_chain.append(code_features)
            y_chain.append(classes_map[truth])
            
    X_chain = np.array(X_chain)
    y_chain = np.array(y_chain)

    ## Fourier descriptors
    X_fft = []
    for i in range(91):

        data_row = cards_data.iloc[i]
        cards = ['P1_number', 'P2_number', 'P3_number', 'P4_number']

        for index in cards:
            img = data_row[index]
            contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda contour: cv.contourArea(contour), reverse= True)[0]
            fft_features = fourierDescriptor(contours)
            X_fft.append(fft_features)
            
        
    X_fft = np.array(X_fft)
    
    ## concatenate inputs and train XGBClassifier
    ### UNCOMMENT TO USE FREEMAN CODE
    #X_both = np.concatenate((X, X_chain), axis=1)
    
    ### Comment if you use FREEMAN CODE
    X_both = np.concatenate((X_fft, X), axis=1)
    model = xgboost.XGBClassifier()
    model = model.fit(X_both, y_chain)
    
    return model


def process_data_for_number_classification(game_df,model1):
    X_test = []
    X_test_chain = []
    
    with torch.no_grad():
        for i in tqdm(range(len(game_df))):

            data_row = game_df.iloc[i]
            cards = ['P1_number', 'P2_number', 'P3_number', 'P4_number']

            for idx in cards:
                vector = cv.resize(data_row[idx], (28, 28))
                card = torch.tensor(vector, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                out = model1(card)
                X_test.append(out.numpy().reshape(-1))
    
    ## Freeman code
    for i in range(len(game_df)):

        data_row = game_df.iloc[i]
        cards = ['P1_number', 'P2_number', 'P3_number', 'P4_number']

        for index in cards:
            img = data_row[index]
            img = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=((0, 0), (0, 0)))
            _, _, directions_count, area, indiv_deviation = get_chain_code(img)
            code_features = directions_count + [area] + [indiv_deviation]
            X_test_chain.append(code_features)

    X_test_fft = []
    for i in range(len(game_df)):

        data_row = game_df.iloc[i]
        cards = ['P1_number', 'P2_number', 'P3_number', 'P4_number']

        for index in cards:
            img = data_row[index]
            contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda contour: cv.contourArea(contour), reverse= True)[0]
            fft_features = fourierDescriptor(contours)
            X_test_fft.append(fft_features)
            
        
    X_test_fft = np.array(X_test_fft)
            
    X_both = np.concatenate((X_test_fft, X_test), axis=1)
#     # Uncomment to use Freeman Code
#     X_test_chain = np.array(X_test_chain)
#     X_both = np.concatenate((X_test, X_test_chain), axis=1)
    return X_both

def assign_predicted_to_players(pred,dataframe):
    p1=[]
    p2=[]
    p3=[]
    p4=[]
    for i in range(int(len(pred)/4)):
        p1.append(pred[4*i])
        p2.append(pred[4*i+1])
        p3.append(pred[4*i+2])
        p4.append(pred[4*i+3])
    
    df=dataframe.copy()
    df["number_p1"] = p1
    df["number_p2"] = p2
    df["number_p3"] = p3
    df["number_p4"] = p4
    return df

def output_result(dataframe):
    df = dataframe.copy()
    df["P1"] = df.apply(lambda row: row.number_p1 + row.suite_p1 ,axis=1)
    df["P2"] = df.apply(lambda row: row.number_p2 + row.suite_p2 ,axis=1)
    df["P3"] = df.apply(lambda row: row.number_p3 + row.suite_p3 ,axis=1)
    df["P4"] = df.apply(lambda row: row.number_p4 + row.suite_p4 ,axis=1)
    return df[["P1","P2","P3","P4","D"]].copy()

def compute_score_standard(P1,P2,P3,P4):
    p1=int(P1[0].replace('J','10').replace('Q','11').replace('K','12'))
    p2=int(P2[0].replace('J','10').replace('Q','11').replace('K','12'))
    p3=int(P3[0].replace('J','10').replace('Q','11').replace('K','12'))
    p4=int(P4[0].replace('J','10').replace('Q','11').replace('K','12'))
    numbers = [p1,p2,p3,p4]
    round_max = np.max([p1,p2,p3,p4])
    round_pts = np.zeros(4,dtype=int)
    for i in range(len(numbers)):
        if numbers[i]==round_max:
            round_pts[i]=1
    return round_pts

def compute_score_advanced(P1,P2,P3,P4,D):  
    p1_number=int(P1[0].replace('J','10').replace('Q','11').replace('K','12'))
    p2_number=int(P2[0].replace('J','10').replace('Q','11').replace('K','12'))
    p3_number=int(P3[0].replace('J','10').replace('Q','11').replace('K','12'))
    p4_number=int(P4[0].replace('J','10').replace('Q','11').replace('K','12'))
    player_numbers=[p1_number,p2_number,p3_number,p4_number]
    
    p1_suite=P1[1]
    p2_suite=P2[1]
    p3_suite=P3[1]
    p4_suite=P4[1]
    
    suites =[p1_suite,p2_suite,p3_suite,p4_suite]

    dealer_indx = int(D)-1
    numbers = []
    for i in range(len(suites)):
        if suites[i]!=suites[dealer_indx]:
            numbers.append(0)
        else:
            numbers.append(player_numbers[i])
    
    
    round_winner = np.argmax(numbers)
    round_pts = np.zeros(4,dtype=int)
    round_pts[round_winner]=1
    return round_pts

def predict_game_points(result):
    scores = pd.DataFrame()
    scores['standard'] = result.copy().apply(lambda row: compute_score_standard(row.P1,row.P2,row.P3,row.P4),axis=1)
    scores['advanced'] = result.copy().apply(lambda row: compute_score_advanced(row.P1,row.P2,row.P3,row.P4,row.D),axis=1)
    pts_standard, pts_advanced = np.sum(scores.standard),np.sum(scores.advanced)
    return pts_standard, pts_advanced
