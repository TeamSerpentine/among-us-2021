# Computer vision framework for Among us
# Current author(s): Martin Rooijackers
#
#
# Text detection taken from:
# https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
# ocr done with Tesseract

from dataclasses import dataclass
import os

# Python code to reading an image using OpenCV
import numpy as np
import cv2

from Levenshtein import distance as lev

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

#utility functions for the framework.
import utils

# ocr helper functions
import ocr

# settings used by the framework
import settings

# vision utilities
import vision

@dataclass
class ChatLog:
    """
    Dataclass for storing chat messages

    id: The id of the message
    r: the r value of the player's color
    g: the g value of the player's color
    b: the b value of the player's color

    colorName: The name of the player's color
    Check https://among-us.fandom.com/wiki/Category:Colors for all names

    frameCount: the frame of the video where this chat message appears
    name: The username of the player
    message: the chat message send by the player

    """
    id: int
    r: int
    g: int
    b: int
    color_name: str
    frame_count: int
    name: str
    message: str



video_state_looking_for_start = 1
video_state_looking_for_end = 2

check_voting_state_looking_for_vote = 1  # look to see if there is a voting going on
check_voting_state_looking_for_confirm = 2  # a voting has been done, confirm if it is an impostor


############ color functions ############

def text_detection(frame):
    """
    The function which performs the text detection on a frame of a video
    This function will draw boxes around all detected text fields

    :param frame: The frame of the video
    :return:
    """
    # Read and store arguments
    conf_threshold = 0.3  # Confidence threshold.
    nms_threshold = 0.2  # Non-maximum suppression threshold.
    inp_width = 1920 - 320  # resize  width (mutliple of 32)
    inp_height = 1056 + 32  # resize  height (mutliple of 32)

    # Create a new named window
    k_win_name = "EAST: An Efficient and Accurate Scene Text Detector"
    cv2.namedWindow(k_win_name, cv2.WINDOW_NORMAL)
    out_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Get frame height and width
    height_ = frame.shape[0]
    width_ = frame.shape[1]
    r_w = width_ / float(inp_width)
    r_h = height_ / float(inp_height)

    # Create a 4D blob from frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (inp_width, inp_height),
                                 (123.68, 116.78, 103.94), True, False)

    # Run the detection model
    settings.detector.setInput(blob)

    # tickmeter.start()
    outs = settings.detector.forward(out_names)
    # tickmeter.stop()

    # Get scores and geometry
    scores = outs[0]
    geometry = outs[1]
    [boxes, confidences] = utils.decode_bounding_boxes(scores, geometry, conf_threshold)

    # Apply NMS
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] *= r_w
            vertices[j][1] *= r_h

        # get cropped image using perspective transform
        if True:
            cropped = utils.four_points_transform(frame, vertices)
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # Create a 4D blob from cropped image
            blob = cv2.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
            # recognizer.setInput(blob)

        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            cv2.line(frame, p1, p2, (0, 255, 0), 2)


    # Display the frame
    cv2.imshow(k_win_name, frame)




def grab_settings(frame):
    """
    Function which grabs the settings that are used to play the game
    These settings appear in the lobby

    :param frame: A frame of the video when the game is still in the lobby
    :return: a string with the ocr result which gives the settings used
    """
    crop_img = frame[0:1000, 0:500]  # where the setting text is in 1920 x 1080
    crop_img = cv2.bitwise_not(crop_img)  # invert the image for Tesseract
    ocr_results = ocr.get_text_from_image(crop_img)
    return ocr_results

def extract_text_chat_screen(frame, frame_count=0):
    """
    function that performs ocr on the chat screen
    The chat messages are associated with the color of the person who wrote this
    the information is logged in ChatLogArray , duplicates are not stored

    :param frame_count: the frame count of the frame being analyzed
    :param frame: the frame of a video being analyzed
    :return:
    """

    threshold = 245
    maximum_value = 255
    end_of_chat_y = 750

    #global CHATLOG_ARRAY
    settings.CHATLOG_ARRAY
    img = frame.copy()

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    ret, bin = cv2.threshold(gray, threshold, maximum_value, cv2.THRESH_BINARY)

    # closing
    kernel = np.ones((3, 3), np.uint8) # 3x3 kernel filled with 1
    closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

    # a boolean which indicated if the messages from the
    # person playing the game should also be included
    include_player_messages = True

    # because the iterator might need to change, don't use for loop
    y = 0  # start at the top
    while y < end_of_chat_y:  # stop when the chat input box is reached
        y = y + 1
        if y > end_of_chat_y:  # reached the end of chat. So stop here
            break
        x = 288  # in the case of 1920 x 1080

        is_message_box = False  # Check if there is a message box at this y
        if closing[y, x] == 255 and closing[y + 100, x] == 255:
            is_message_box = True

        if is_message_box:
            end_y = y + 10
            while closing[end_y, x] == 255:
                end_y = end_y + 1

            crop_img_text = closing[y + 50:end_y - 2, 420:1253]
            if crop_img_text.size == 0:
                print("failure in grabbing text from chat")
                y = end_y + 10
                continue

            ocr_result = ocr.get_text_from_image(crop_img_text)

            # how many pixels to move downwards to find the color of the person
            color_person_y = 50
            # get the color values of the player saying the word
            b, g, r = frame[y + color_person_y, x + 100]
            #  get the color name based on the RGB values
            color_name = utils.get_color_name(r, g, b)

            #  variable which indicates if a chat message is already logged
            #  This is done to prevent duplicates
            already_logged = False
            for chat in settings.CHATLOG_ARRAY:

                if color_name == chat.color_name:
                    if chat.message == ocr_result:
                        already_logged = True
                    # Also check if the levenshtein distance == 1 in case of ocr errors
                    # if the distance is only 1, then it very likely saw the same message,
                    # which is usually caused by an ocr error reading a character the wrong way
                    levenshtein_distance = lev(chat.message, ocr_result)
                    if levenshtein_distance == 1:
                        already_logged = True

            # if the chat entry doesn't exist yet, create it
            if not already_logged:
                # now get the name through ocr
                crop_img = closing[y:y + 50, 420:1253]

                ocr_result_name = ocr.get_text_from_image(crop_img)

                # now put it all in the chat logger
                new_log = ChatLog(len(settings.CHATLOG_ARRAY), r, g, b, color_name, frame_count, ocr_result_name, ocr_result)
                settings.CHATLOG_ARRAY.append(new_log)
            y = end_y + 10

    # now do the same for the player's chat messages.
    y = 0
    while y < end_of_chat_y:
        y = y + 1
        if y > end_of_chat_y:  # reached the end of chat. So stop here
            break
        x_player = 1375  # player messages in the case of 1920 x 1080

        is_message_box_player = False
        if closing[y, x_player] == 255 and closing[y + 100, x_player] == 255 and include_player_messages:
            is_message_box_player = True

        if is_message_box_player:
            end_y = y + 10
            while closing[end_y, x_player] == 255:
                end_y = end_y + 1

            crop_img_text = closing[y + 50:end_y - 2, 420:1253]
            if crop_img_text.size == 0:
                print("failure in grabbing text from chat")
                y = end_y + 10
                continue

            ocr_result = ocr.get_text_from_image(crop_img_text)

            # how many pixels to move downwards to find the color of the person
            color_person_y = 50

            # get the color values of the player saying the word
            b, g, r = frame[y + color_person_y, x_player - 40]

            color_name = utils.get_color_name(r, g, b)

            already_logged = False
            for chat in settings.CHATLOG_ARRAY:
                # and chat.r == r and chat.g == g and chat.b == b
                # check if this crewmate already said this (don't log double)

                if color_name == chat.color_name:
                    if chat.message == ocr_result:
                        already_logged = True
                    # Also check if the levenshtein distance == 1 in case of ocr errors
                    # if the distance is only 1, then it very likely saw the same message,
                    # which is usually caused by an ocr error reading a character the wrong way
                    levenshtein_distance = lev(chat.message, ocr_result)
                    if levenshtein_distance == 1:
                        already_logged = True

                # if chat.r == r and chat.g == g and chat.b == b:
                # already_logged = True

            # if the chat entry doesn't exist yet. Create it
            if not already_logged:
                # now get the name through ocr
                # crop_img = closing[ y+50:y+100, 420:1253]
                crop_img = closing[y:y + 50, 420:1253]

                ocr_result_name = ocr.get_text_from_image(crop_img)

                # now put it all in the chat logger
                new_log = ChatLog(len(settings.CHATLOG_ARRAY), r, g, b, color_name, frame_count, ocr_result_name, ocr_result)
                settings.CHATLOG_ARRAY.append(new_log)
                # print(ChatLogArray)
            # cv2.floodFill()
            # cv2.imshow("img_outline", crop_img)
            y = end_y + 10
            # print("new y: " ,y)
            # cv2.waitKey()

    # cv2.waitKey()


def check_if_chat_screen(frame, frame_count=0):
    """
    This function checks if the given frame is a chat screen.
    Currently done by looking at the x/100 in the chat screen
    if so, it does ocr

    :param frame: The frame of a video
    :param frame_count: variable which indicates which frame of the video this is.
    :return:
    """

    width = 1920
    height = 1080

    img = cv2.resize(frame, (width, height))  # resize the image to (1920, 1080)
    crop_img = img[822:850, 1338:1410]  # location where the x/100 should be in 1920x1080
    ocr_result = ocr.get_text_from_image(crop_img)

    if "100" in ocr_result:
        extract_text_chat_screen(img, frame_count)
        #text_detection(frame)
        cv2.waitKey(1)



def grab_colors_defeat_screen(frame, total_impostors):
    """
    function which tries to extract the colors of the impostors in a defeat screen
    Some cosmetics will interfere with how this process currently works
    Should eventually be changed into something that can also handle cosmetics

    :param frame: an image containing a defeat screen, and the total number of impostors
    :param total_impostors: The total number of impostors that are in the game
    :return: a list of the colors of the impostors that appear in the image
    """

    # Make sure to check if they are ghosts
    # if they are, then there should be a black pixel at::
    # ghost 1,   x = 874 , y = 734
    # ghost 2  ,  x = 1121 , y = 772=0

    width = 1920
    height = 1080

    img = cv2.resize(frame, (width, height))  # resize the image to (1920, 1080)
    colors = []
    impostor1_ghost = False
    impostor2_ghost = False
    # Haven't seen loc of ghost 3 yet, once I do I will add it

    # check if impostor 1 is a ghost
    b, g, r = img[734, 874]
    if (b < 30) and (g < 30) and (r < 30):
        impostor1_ghost = True

    # check if impostor 2 is a ghost
    b, g, r = img[770, 1121]
    if (b < 30) and (g < 30) and (r < 30):
        impostor2_ghost = True

    # 1st (and only) impostor: x = 1000 , y = 725
    if total_impostors == 1:
        b, g, r = img[725, 1000]
        color_name = utils.get_color_name(r, g, b)
        if impostor1_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]

    # 1st impostor , x = 1000 , y = 725
    # 2nd impostor, x = 1110 , y = 700
    if total_impostors == 2:
        # 1st impostor
        b, g, r = img[725, 1000]
        color_name = utils.get_color_name(r, g, b)
        if impostor1_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]
        # 2nd impostor
        b, g, r = img[700, 1110]
        color_name = utils.get_color_name(r, g, b)
        if impostor2_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]

    # 1st impostor =   x= 1010, y = 723
    # 2nd impostor ,  x=  1157, y = 700
    # 3rd impostor ,   x = 832 , y = 692
    if total_impostors == 3:
        # 1st impostor
        b, g, r = img[723, 1010]
        color_name = utils.get_color_name(r, g, b)
        if impostor1_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]
        # 2nd impostor
        b, g, r = img[700, 1157]
        color_name = utils.get_color_name(r, g, b)
        if impostor2_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]
        # 3th impostor
        b, g, r = img[692, 832]
        color_name = utils.get_color_name(r, g, b)
        colors = colors + [color_name]

    return colors


def output_impostor_data(output_loc, impostor_colors, number_of_impostors):
    """
    function which generates the .txt file with the impostor info
    one integer with the number of impostors followed by the
    list of color from the input separated by newlines

    :param output_loc: the location of where to put the .txt
    :param impostor_colors: the list of impostor colors
    :param number_of_impostors: the number of impostors
    :return:
    """

    # in case the function gets the .json location with this file is related to
    output_loc = output_loc.replace(".json", ".txt")

    f = open(output_loc, "w")
    number_write = str(number_of_impostors)
    f.write(number_write)
    f.write("\n")
    for color in impostor_colors:
        f.write(color)
        f.write("\n")


def output_ocr_data(outputlocation):
    """
    function which outputs the ChatLogArray into a .json file

    :param outputlocation: the location of where to put the .json file
    :return:
    """

    #global CHATLOG_ARRAY

    f = open(outputlocation, "w")
    total_iterations = 0
    loop_frame = 0
    if len(settings.CHATLOG_ARRAY) > 0:
        loop_frame = settings.CHATLOG_ARRAY[0].frame_count

    meeting_id = 1
    f.write("{\n \"ChatMessages\": [")
    for chat in settings.CHATLOG_ARRAY:
        total_iterations = total_iterations + 1
        f.write("  {\n")
        f.write("    \"ID\": ")
        f.write(str(chat.id))
        f.write(",\n")

        f.write("    \"RED\": ")
        f.write(str(chat.r))
        f.write(",\n")

        f.write("    \"GREEN\": ")
        f.write(str(chat.g))
        f.write(",\n")

        f.write("    \"BLUE\": ")
        f.write(str(chat.b))
        f.write(",\n")

        f.write("    \"FRAMECOUNT\": ")
        f.write(str(chat.frame_count))
        f.write(",\n")

        # current method of determining meetingID
        if chat.frame_count - loop_frame > 2000:
            meeting_id = meeting_id + 1

        loop_frame = chat.frame_count

        f.write("    \"meeting_id\": ")
        f.write(str(meeting_id))
        f.write(",\n")

        f.write("    \"COLORNAME\": \"")
        f.write(str(chat.color_name))
        f.write("\",\n")

        name_write = chat.name.replace("\n\x0c", "").replace("\n", "").replace("\"", "\\\"")
        name_write = "    \"NAME\": \"" + name_write + "\",\n"
        f.write(name_write)
        text_write = chat.message.replace("\n\x0c", "").replace("\n", "").replace("\"", "\\\"")
        text_write = "    \"MESSAGE\": \"" + text_write + "\"\n"
        f.write(text_write)
        if total_iterations == len(settings.CHATLOG_ARRAY):
            f.write("  }\n")
        else:
            f.write("  },\n")

    f.write("  ]\n}\n")

    f.close()


def tally_votes(frame):
    """
    determine who is getting voted out, or if the vote will be a tie/skipped

    :param frame: an image with the voting screen with the votes being displayed
    :return:
    """
    frame = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)

    result = "Nothing"

    step_vote_icon = 459 - 404  # step size in the x direction between each vote
    max_votes = 0
    current_voted = "Nothing"

    first_x = 322  # location of the player colors in the left side of the vote screen
    second_x = 980  # location of the player colors in the right side of the vote screen
    # x,y for voting screen,  x+y for color , then x,y for start of voting icons
    all_vote_positions = [(284, 222, first_x, 282, 415, 291), (277, 359, first_x, 414, 415, 428),
                          (282, 494, first_x, 562, 415, 565), (282, 632, first_x, 687, 415, 705)]
    all_vote_positions = all_vote_positions + [(275, 766, first_x, 834, 415, 841), (935, 219, second_x, 282, 1070, 293),
                                               (935, 359, 1006, 436, 1070, 428)]
    all_vote_positions = all_vote_positions + [(936, 495, second_x, 543, 1070, 565),
                                               (933, 620, second_x, 693, 1070, 705),
                                               (941, 770, second_x, 830, 1070, 841)]

    # original 2nd right side (935,359,second_x,411,1070,428)
    # changed because the broken screen as ghost can interfere

    for pos in all_vote_positions:

        x = pos[2]
        y = pos[3]
        b, g, r = frame[y, x]
        vote_color = utils.get_color_name(r, g, b)
        x = pos[0]
        y = pos[1]
        b, g, r = frame[y, x]
        # check if the player is actually in the game
        if (b > 200) and (g > 200) and (r > 200):
            print("Still in Game")
            # now count the total amount of people who voted for this player
            total_votes = 0
            for i in range(7):
                x = pos[4] + (step_vote_icon * i)
                y = pos[5]
                b, g, r = frame[y][x]
                if (b > 220) and (r > 200) and (g > 200):  # no more votes for this person
                    break
                total_votes = total_votes + 1
                print(total_votes)
                if total_votes == max_votes:  # Tie
                    current_voted = "Nothing"
                if total_votes > max_votes:  # current most voted
                    max_votes = total_votes
                    current_voted = vote_color

    result = current_voted

    # also check the amount of votes for skipping
    # vote icons for that start at x = 430, y = 960  inn 1920x1080
    skip_step_size = 52  # x distance between each vote icon for skipping vote
    x = 430
    y = 960

    total_vote_skip = 0
    for i in range(7):
        x = 430 + (i * skip_step_size)
        b, g, r = frame[y][x]
        if r > 150:
            break
        total_vote_skip = total_vote_skip + 1
    if total_vote_skip >= max_votes:
        result = "Nothing"

    return result


def find_votes(frame):
    """
    function which checks if a voting has occurred

    :param frame: a frame of a video being analyzed
    :return: the string "Nothing" if no voting has occured
    Otherwise returns the result of the tally_votes function
    """

    width = 1920
    height = 1080

    begin_y = 67
    end_y = 1010

    threshold = 20
    maximum_value = 255

    img = cv2.resize(frame, (width, height))  # resize the image to (1920, 1080)
    # check if this is a voting screen
    total_grey_dots = 0
    x = 1602
    for y in range(begin_y, end_y):
        b, g, r = img[y, x]
        if (b > 140) and (b < 170) and (r > 130) and (r < 170) and (g < 170) and (g > 130):
            total_grey_dots = total_grey_dots + 1

    if total_grey_dots < 500:
        return "Nothing"

    crop_img = img[910:967, 1230:1555]  # currently testing for 1920 x 1080
    height, width, channel = crop_img.shape

    for x in range(0, width):
        for y in range(0, height):
            b, g, r = crop_img[y, x]
            if (b > 200) and (g > 200) and (r > 200):  # white
                crop_img[y, x] = (0, 0, 0)
            elif (b < 50) and (g < 50) and (r > 190):  # red
                crop_img[y, x] = (0, 0, 0)
            else:
                crop_img[y, x] = (255, 255, 255)

    inv = cv2.bitwise_not(crop_img)
    gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    #     ret, bin = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    # threshold
    ret, bin = cv2.threshold(gray, threshold, maximum_value, cv2.THRESH_BINARY)
    # closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
    # invert black/white
    inv = cv2.bitwise_not(closing)

    ocr_result = ocr.get_text_from_image(inv)

    # actually "Proceeding"  , but the screen has a broken look if you are a ghost
    if "eeding" in ocr_result:
        # votes will all appear at 2seconds , so any time after will allow a tally of them
        if ("2" in ocr_result) or ("1" in ocr_result) or ("0" in ocr_result):
            # count votes
            results_tally = tally_votes(img)
            print(results_tally)
            # cv2.imshow("vote screen original", frame)
            # cv2.waitKey(0)
            return results_tally

    return "Nothing"


def check_impostor_eject(frame):
    """
    A function which checks if the person being ejected is an impostor
    this function will only work correctly if the "confirm ejects"
    option is turned on in the game being analyzed

    :param frame: A frame of the video being analyzed
    :return: true if the player being ejected is not an impostor (given the current frame).
    Otherwise it returns False
    """

    width = 1920
    height = 1080

    img = cv2.resize(frame, (width, height))  # resize the image to (1920, 1080)
    crop_img = img[535:603, 490:1456]  # currently testing for 1920 x 1080

    height, width, channel = crop_img.shape

    # invert the colors and turn the image into a binary image for better ocr
    for x in range(0, width):
        for y in range(0, height):
            b, g, r = crop_img[y, x]
            if (b > 200) and (g > 200) and (r > 200):  # white
                crop_img[y, x] = (0, 0, 0)
            else:
                crop_img[y, x] = (255, 255, 255)

    ocr_result = ocr.get_text_from_image(crop_img)

    # for when there are multiple impostors
    if "not An Impostor" in ocr_result:
        return True

    # for when there is 1 impostor
    if "not The Impostor" in ocr_result:
        return True

    return False




def process_video(location, output_location):
    """
    the main function for processing a single video
    this function calls the other functions to analyze
    the video to extract the information needed for classification

    :param location: location of the video that is being analyzed
    :param output_location: Where to put the .json and .txt files with the ocr and impostor data
    :return:
    """

    # print(location)
    # clear previous data
    #global CHATLOG_ARRAY
    settings.CHATLOG_ARRAY = []

    width = 1920
    height = 1080

    cap = cv2.VideoCapture(location)

    # make a special folder for this video for putting all the data in
    if not os.path.isdir(output_location):
        os.mkdir(output_location)

    total_games_processed = 1

    current_state = video_state_looking_for_start
    total_impostors = 0

    current_voting_check_state = check_voting_state_looking_for_vote
    current_color_vote_check = "Nothing"  # the color to check if it actually is an impostor (when confirm eject is on)
    frame_star_checking = 0  # the frame where the algorithm starts checking for eject confirmation
    impostor_list_voting = []

    check_grey_to_black = False

    current_frame = 0

    while cap.isOpened():

        # Read frame
        has_frame, frame = cap.read()
        current_frame = current_frame + 1
        if not has_frame:
            print("Done processing this video")
            break

        if current_voting_check_state == check_voting_state_looking_for_confirm:

            max_frames_to_check = frame_star_checking + (30 * 20)  # check for a max of 600 frames
            if current_frame > max_frames_to_check:
                current_voting_check_state = check_voting_state_looking_for_vote
                # after all this time, still no message found that the player was not an impostor
                # so add the player to the impostor list
                impostor_list_voting = impostor_list_voting + [current_color_vote_check]
                current_color_vote_check = "Nothing"

            player_not_impostor = check_impostor_eject(frame)
            if player_not_impostor:
                current_color_vote_check = "Nothing"
                current_voting_check_state = check_voting_state_looking_for_vote

        if current_state == video_state_looking_for_start:
            total_impostors = vision.check_if_start_screen(frame)

        if total_impostors > 0 and current_state == video_state_looking_for_start:
            current_state = video_state_looking_for_end
            start_screen_debug_loc = output_location + "/" + str(total_games_processed) + ".Start.png"
            cv2.imwrite(start_screen_debug_loc, frame)

        # only check for votes when a new game is being analyzed
        # otherwise you will be spending time checking for votes in Impostor games as well, which we skip
        if (current_voting_check_state == check_voting_state_looking_for_vote) and (
                current_state == video_state_looking_for_end) and (not check_grey_to_black):
            check_grey_to_black = vision.check_if_vote_screen(frame)

        # begin rollback in video to check votes
        if check_grey_to_black:
            # resized_image =
            is_fading = vision.check_if_vote_screen_fading(frame)
            if is_fading:
                # if it is fading, that must mean that the count has happened
                # so roll back to there to start the count
                get_current_frame = current_frame  # useful for rolling back
                rollback_frame = current_frame
                resized_image = cv2.resize(frame, (width, height))
                b, g, r = resized_image[100, 1602]
                # now keep going back till there is a good overview of the votes
                best_score, best_frame = vision.grab_voting_screen(cap, rollback_frame)
                if best_frame is None:
                    best_frame = current_frame

                cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
                rollback_has_frame, frame_roll_back = cap.read()
                # print("tally voted with rollback frame")
                # cv2.imshow("tally vote", frame_roll_back)
                # cv2.waitKey(0)

                result_tally = tally_votes(frame_roll_back)
                cap.set(cv2.CAP_PROP_POS_FRAMES, get_current_frame)

                if not ("Nothing" in result_tally):  # a player is being voted out
                    current_voting_check_state = check_voting_state_looking_for_confirm
                    current_color_vote_check = result_tally
                    frame_star_checking = current_frame
                    # debugInfo
                    vote_debug_loc = output_location + "/" + str(
                        total_games_processed) + current_color_vote_check + ".Vote.png"
                    resized_frame = cv2.resize(frame_roll_back, (width, height))
                    cv2.imwrite(vote_debug_loc, resized_frame)
                else:  # otherwise, skip forward
                    cap.set(cv2.CAP_PROP_POS_FRAMES, get_current_frame + (30 * 5))
                    current_frame = get_current_frame + (30 * 5)
                check_grey_to_black = False

        # end rollback check votes

        # as long as a game is ongoing, do ocr
        if current_state == video_state_looking_for_end:
            check_if_chat_screen(frame, current_frame)

        if current_state == video_state_looking_for_end:
            result = vision.check_if_end_screen(frame)
            # once the game has ended, output the ocr and start looking for the next game
            if not ("neither" in result):

                # if the game ended, and a color is still under consideration for impostor status
                # then that color must be the impostor
                if not ("Nothing" in current_color_vote_check):
                    impostor_list_voting = impostor_list_voting + [current_color_vote_check]

                end_screen_debug_loc = output_location + "/" + str(total_games_processed) + "End.png"
                resized_frame = cv2.resize(frame, (1920, 1080))
                cv2.imwrite(end_screen_debug_loc, resized_frame)
                # cv2.imwrite(end_screen_debug_loc, frame)

                # total_games_processed = total_games_processed + 1
                output_ocr_loc = output_location + "/" + str(total_games_processed) + ".json"
                current_state = video_state_looking_for_start
                output_ocr_data(output_ocr_loc)
                CHATLOG_ARRAY = []
                # if there is a defeat screen, grab the impostor data from there
                output_impostor_loc = output_location + "/" + str(total_games_processed) + ".txt"
                if "defeat" in result:
                    colors_impostors = grab_colors_defeat_screen(frame, total_impostors)
                    output_impostor_data(output_impostor_loc, colors_impostors, total_impostors)
                    # outputImpostorData(output_impostor_loc, impostor_list_voting, total_impostors)

                    print(colors_impostors)
                    # cv2.imshow("crew screen", frame)
                    # cv2.waitKey(0)
                else:  # otherwise, rely on voting data

                    output_impostor_data(output_impostor_loc, impostor_list_voting, total_impostors)
                total_games_processed = total_games_processed + 1
                # reset some more variables
                current_voting_check_state = check_voting_state_looking_for_vote
                current_color_vote_check = "Nothing"  # the color to check if it actually is an impostor
                impostor_list_voting = []
                check_grey_to_black = False

        # skip 15 frames
        # TO maybe DO: use grab() instead so OpenCV doesn't have to decode
        frames_to_skip = 15

        while frames_to_skip > 0:
            frames_to_skip = frames_to_skip - 1
            cap.read()
            current_frame = current_frame + 1

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

    # return


#  The main function. Here the text detection NN is loaded, and the list of videos is extracted from the location
#  The actual video analysis happens in the ProcessVideo function
def main():
    """
    The main function. Here the text detection NN is loaded,
    and the list of videos is extracted from the location

    The actual video analysis happens in the ProcessVideo function

    :return:
    """

    # Load network
    #global detector
    settings.detector = cv2.dnn.readNet(settings.MODEL_DETECTOR)

    videos = os.listdir(settings.VIDEO_LOCATION)
    for video in videos:
        video_loc = settings.VIDEO_LOCATION + video
        remove_extension = video.split(".")[0] # removes the video file extension
        process_video(video_loc, settings.VIDEO_LOCATION + remove_extension)
        print(video)
    exit(0)


if __name__ == "__main__":
    main()
