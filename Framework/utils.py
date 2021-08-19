from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import numpy as np
import cv2
import math
import os
import re

def get_hue(r, g, b):
    """
    get the HUE value from RGB values

    :param r: red
    :param g: green
    :param b: blue
    :return: the hue value based on the RGB value
    """

    minimum = min(r, g, b)
    maximum = max(r, g, b)

    if min == max:
        return 0

    hue = 0.0
    if max == r:
        hue = (g - b) / (maximum - minimum)
    if max == g:
        hue = 2.0 + ((b - r) / (maximum - minimum))
    if max == b:
        hue = 4.0 + ((r - g) / (maximum - minimum))

    hue = hue * 60
    if hue < 0.0:
        hue = hue + 360

    return round(hue)


def get_color_name_RGB(r, g, b):
    """
    get color name based on RGB distance
    RGB values of colors are acquired from:
    https://among-us.fandom.com/wiki/Category:Colors

    :param r: red
    :param g: green
    :param b: blue
    :return: The color closest to the given RGB values based on RGB distance
    """
    red = ("RED", 197, 17, 17)
    lime = ("LIME", 80, 239, 58)
    black = ("BLACK", 63, 71, 78)
    purple = ("PURPLE", 108, 46, 188)
    orange = ("ORANGE", 239, 124, 12)
    cyan = ("CYAN", 57, 255, 221)
    green = ("GREEN", 18, 127, 45)
    pink = ("PINK", 240, 84, 189)
    yellow = ("YELLOW", 244, 245, 84)
    blue = ("BLUE", 18, 44, 212)
    white = ("WHITE", 214, 222, 241)
    brown = ("BROWN", 113, 73, 30)

    color_list = [red] + [lime] + [black] + [purple] + [orange] + [cyan] + [green] + [pink] + [yellow] + [blue] + \
                 [white] + [brown]

    # print(color_list)

    best_match_color = "NONE"
    closest_dist = 99999
    for color in color_list:
        RGB_distance = abs(r - color[1]) + abs(g - color[2]) + abs(b - color[3])
        if RGB_distance < closest_dist:
            best_match_color = color[0]
            closest_dist = RGB_distance

    return best_match_color



def get_color_name(r, g, b):
    """
    get color name based on cie2000 distance
    RGB values of colors are acquired from:
    https://among-us.fandom.com/wiki/Category:Colors

    :param r: red
    :param g: green
    :param b: blue
    :return: The color closest to the given RGB values based on cie2000 distance
    """

    red = ("RED", 197, 17, 17)
    lime = ("LIME", 80, 239, 58)
    black = ("BLACK", 63, 71, 78)
    purple = ("PURPLE", 108, 46, 188)
    orange = ("ORANGE", 239, 124, 12)
    cyan = ("CYAN", 57, 255, 221)
    green = ("GREEN", 18, 127, 45)
    pink = ("PINK", 240, 84, 189)
    yellow = ("YELLOW", 244, 245, 84)
    blue = ("BLUE", 18, 44, 212)
    white = ("WHITE", 214, 222, 241)
    brown = ("BROWN", 113, 73, 30)

    color_list = [red] + [lime] + [black] + [purple] + [orange] + [cyan] + [green] + [pink] + [yellow] + [blue] + [
        white] + [brown]

    # print(color_list)

    best_match_color = "NONE"
    closest_dist = 99999

    for color in color_list:
        color1_rgb = sRGBColor(r, g, b, True)
        color2_rgb = sRGBColor(color[1], color[2], color[3], True)
        color1_lab = convert_color(color1_rgb, LabColor)
        color2_lab = convert_color(color2_rgb, LabColor)
        delta_e = delta_e_cie2000(color1_lab, color2_lab)

        if delta_e < closest_dist:
            best_match_color = color[0]
            closest_dist = delta_e

    return best_match_color



def get_ghost_color_name(r, g, b):
    """
    ghosts have a different RGB color encoding
    currently still using RGB distance since the defeat screen adds a red hue
    which could interfere with the deltaE of cie2000

    blue and purple ghosts look quite alike though.
    Even manually distinguishing between them is difficult
    still need to look for a fix for that

    :param r: red
    :param g: green
    :param b: blue
    :return: The ghost color name closest to the given RGB values based on RGB distance
    """
    # extracted ghost colors

    red = ("RED", 127, 15, 2)
    lime = ("LIME", 74, 76, 15)
    black = ("BLACK", 80, 28, 28)
    purple = ("PURPLE", 66, 17, 87)
    orange = ("ORANGE", 146, 49, 6)
    cyan = ("CYAN", 80, 104, 92)
    green = ("GREEN", 80, 80, 20)
    pink = ("PINK", 171, 49, 77)
    yellow = ("YELLOW", 101, 61, 20)
    blue = ("BLUE", 64, 21, 97)
    white = ("WHITE", 155, 104, 111)
    brown = ("BROWN", 81, 36, 37)

    # end of extracted ghost colors

    colorList = [red] + [lime] + [black] + [purple] + [orange] + [cyan] + [green] + [pink] + [yellow] + [blue] + [
        white] + [brown]

    # print(colorList)

    best_match_color = "NONE"
    closest_dist = 99999

    for color in colorList:
        # print (color)
        # RNGdistance = 0
        RGB_distance = abs(r - color[1]) + abs(g - color[2]) + abs(b - color[3])
        if RGB_distance < closest_dist:
            best_match_color = color[0]
            closest_dist = RGB_distance

    return best_match_color


############ Utility functions for text detection ############

def four_points_transform(frame, vertices):
    """
    four points transform.
    Used for getting a cropped image using perspective transform

    :param frame: the frame being transformed
    :param vertices: 4 corner of the rotation rectangle
    :return: transformed (cropped image)
    """
    vertices = np.asarray(vertices)
    output_size = (100, 32)
    target_vertices = np.array([
        [0, output_size[1] - 1],
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1]], dtype="float32")

    rotation_matrix = cv2.getPerspectiveTransform(vertices, target_vertices)
    result = cv2.warpPerspective(frame, rotation_matrix, output_size)
    return result



def decode_bounding_boxes(scores, geometry, score_thresh):
    """
    give detections and confidences based on the parameters

    :param scores: Score given by the NN
    :param geometry: geometry of the bounding boxes given by the NN
    :param score_thresh: confidence threshold
    :return: detections and confidences
    """
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if score < score_thresh:
                continue

            # Calculate offset
            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = (
                [offset_x + cos_a * x1_data[x] + sin_a * x2_data[x], offset_y - sin_a * x1_data[x] + cos_a * x2_data[x]])

            # Find points for rectangle
            p1 = (-sin_a * h + offset[0], -cos_a * h + offset[1])
            p3 = (-cos_a * w + offset[0], sin_a * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

## end of utility functions for text detection ######

def pictures_to_video(dir_path):
    """
    Takes all png files in dir_path and makes a video out ot it

    :param dir_path: path of the directory of all the png files
    :return: video make out of all the image files
    """
    # Arguments
    dir_path = 'D:/PythonCVtest/DataSets/Youtube/boxes/'
    ext = ".png"
    output = dir_path + "output.mp4"

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)

    images.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
        out.write(frame)  # Write out frame to video
        cv2.imshow('video', frame)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

def grab_colors(img, output_number = 0):
    """
    Determines which impostors there are on the screen and
    draws a square around them with their respective color

    :param img: the image which is analysed
    :param output_number: debug number
    :return:
    """

    dim = (200, 150)
    # resize image
    lowrez_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    red = list(("RED", 197, 17, 17, 0, 0, 0))
    lime = list(("LIME", 80, 239, 58, 0, 0, 0))
    black = list(("BLACK", 63, 71, 78, 0, 0, 0))
    purple = list(("PURPLE", 108, 46, 188, 0, 0, 0))
    orange = list(("ORANGE", 239, 124, 12, 0, 0, 0))
    cyan = list(("CYAN", 57, 255, 221, 0, 0, 0))
    green = list(("GREEN", 18, 127, 45, 0, 0, 0))
    pink = list(("PINK", 240, 84, 189, 0, 0, 0))
    yellow = list(("YELLOW", 244, 245, 84, 0, 0, 0))
    blue = list(("BLUE", 18, 44, 212, 0, 0, 0))
    white = list(("WHITE", 214, 222, 241, 0, 0, 0))
    brown = list(("BROWN", 113, 73, 30, 0, 0, 0))

    color_list = [red] + [lime] + [black] + [purple] + [orange] + [cyan] + \
                 [green] + [pink] + [yellow] + [blue] + [white] + [brown]

    height, width, channel = img.shape
    print('width:  ', width)
    print('height: ', height)
    print('channel:', channel)

    height_lowrez, width_lowrez, channel_lowrez = lowrez_img.shape
    print('width lowrez:  ', width_lowrez)
    print('height lowrez: ', height_lowrez)
    print('channel lowrez:', channel_lowrez)

    for y in range(0, height_lowrez):
        for x in range(0, width_lowrez):
            for current_color in color_list:
                b, g, r = lowrez_img[y, x]
                matches_color = False
                color1_rgb = sRGBColor(r, g, b, True)
                color2_rgb = sRGBColor(current_color[1], current_color[2], current_color[3], True)
                color1_lab = convert_color(color1_rgb, LabColor)
                color2_lab = convert_color(color2_rgb, LabColor)
                delta_e = delta_e_cie2000(color1_lab, color2_lab)

                if delta_e < 6.0:
                    matches_color = True

                if matches_color:
                    current_color[4] = current_color[4] + x
                    current_color[5] = current_color[5] + y
                    current_color[6] = current_color[6] + 1
                    break

    for current_color in color_list:
        total_positions = current_color[6]
        print(current_color[0])
        print(total_positions)
        total_x = current_color[4]
        total_y = current_color[5]

        if total_positions > 3:
            average_x = int(total_x / total_positions)
            average_y = int(total_y / total_positions)

            x_percent = average_x / dim[0]
            y_percent = average_y / dim[1]

            print(x_percent)
            print(y_percent)

            # project the box location to the regular image
            x_location = int(width * x_percent)
            y_location = int(height * y_percent)

            print(x_location)
            print(y_location)

            rec_size = 50
            cv2.rectangle(img, (x_location - rec_size, y_location - rec_size),
                          (x_location + rec_size, y_location + rec_size),
                          (current_color[3], current_color[2], current_color[1]), 5)

    # output debug for image
    img_output_loc = "D:/PythonCVtest/DataSets/Youtube/crew/" + str(output_number) + ".png"
    cv2.imwrite(img_output_loc, img)

    # show image
    cv2.imshow('image', img)
    cv2.waitKey(1)
    print("done")
