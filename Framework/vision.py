
import numpy as np
import cv2

import ocr

def check_if_start_screen(frame):
    """
    Function which checks if this screen is the start of the round screen
    For an example of such screen, check the Among Us wiki:
    https://among-us.fandom.com/wiki/Crewmate

    Should also be able to check if you are starting as a crewmate or impostor


    :param frame: frame of a video being analyzed
    :return: 0 if it is not a start screen, otherwise returns how many impostors there are
    """

    width = 1920
    height = 1080

    begin_x = 370
    end_x = 1550

    img = cv2.resize(frame, (width, height))  # resize the image to (1920, 1080)
    total_black_dots = 0
    total_cyan_dots = 0
    for x in range(begin_x, end_x):
        y = 300
        b, g, r = img[y, x]
        if b < 10 and g < 10 and r < 10:
            total_black_dots = total_black_dots + 1
        if (120 < r < 160) and (g > 230) and (b > 230):
            total_cyan_dots = total_cyan_dots + 1

    # total 1180 dots/pixels
    is_crewmate = False  # check if this screen indicates that the game is starting with the player as crewmate
    if total_black_dots > 100 and total_cyan_dots > 10:
        is_crewmate = True

    if is_crewmate:
        print("found crew frame")

        crop_img = img[400:475, 417:1508]  # currently testing for 1920 x 1080
        crop_img = ocr.preprocess_for_ocr(crop_img)
        ocr_result = ocr.get_text_from_image(crop_img)

        if "1" in ocr_result:
            return 1
        if "2" in ocr_result:
            return 2
        if "3" in ocr_result:
            return 3
        # in case the ocr has trouble recognizing numbers
        # if there is an "is" , then there is only 1 impostor
        if "is" in ocr_result:
            return 1

        # two common ocr errors, should be 2
        if " e " in ocr_result:
            return 2
        if " ec " in ocr_result:
            return 2

        # are could mean 2 or 3
        if "are" in ocr_result:
            # might also be 3, but ocr errors seems to mostly occur with 2
            # Can be skipped if need be
            return 2

    return 0




def check_if_end_screen(frame):
    """
    Function which checks if this screen is the end of the round screen


    :param frame: frame of a video being analyzed
    :return: ""Neither"" if it is not an end screen,
    otherwise returns "defeat" or "victory" to indicate what kind of screen it is
    """

    # check y = 225
    # the higher up the y (check), the more likely it is that the end screen is still fading in

    width = 1920
    height = 1080

    begin_x = 230
    end_x = 1676

    img = cv2.resize(frame, (width, height))  # resize the image to (1920, 1080)
    output = "neither"

    total_black_dots = 0
    total_red_dots = 0
    total_blue_dots = 0
    for x in range(begin_x, end_x):
        y = 225
        b, g, r = img[y, x]
        if b < 10 and g < 10 and r < 10:
            total_black_dots = total_black_dots + 1
        if r < 10 and (115 < g < 150) and b > 215:
            total_blue_dots = total_blue_dots + 1
        if (r > 230) and (g < 10) and (b < 10):
            total_red_dots = total_red_dots + 1

    if total_black_dots > 200 and total_red_dots > 15:
        output = "defeat"

    if total_black_dots > 200 and total_blue_dots > 10:
        output = "victory"

    return output



def check_if_vote_screen(frame):
    """
    check if the frame/image represents a vote screen

    :param frame: the frame of a video being analyzed
    :return: True if the frame represents a vote screen
    False otherwise
    """
    # grey color in voting screen:  r = 146, g = 156 , b = 170

    width = 1920
    height = 1080

    img = cv2.resize(frame, (width, height))  # resize the image to (1920, 1080)

    # check if this is a voting screen
    total_grey_dots = 0
    x = 1602
    for y in range(60, 400):
        b, g, r = img[y, x]
        if (180 > b > 150) and (r > 130) and (r < 170) and (130 < g < 170):
            total_grey_dots = total_grey_dots + 1

    if total_grey_dots < 300:
        return False

    return True


def grab_voting_screen(cap, rollback_frame):
    """
    This function tries to grab the frame which shows how everyone voted.
    This is currently done by going backwards from the rollback frame

    :param cap: the video object
    :param rollback_frame: the frame to go back to
    :return: the best scoring frame, followed by the frame itself
    """
    width = 1920
    height = 1080

    best_score = 0
    best_frame = None

    # while getting_brighter:
    for countF in range(1, 30):
        # getting_brighter = False
        # print("Getting more brighter")
        rollback_frame = rollback_frame - countF
        cap.set(cv2.CAP_PROP_POS_FRAMES, rollback_frame)
        rollback_has_frame, frame_roll_back = cap.read()
        if not rollback_has_frame:
            break
        new_resized_image = cv2.resize(frame_roll_back, (width, height))
        new_b, new_g, new_r = new_resized_image[100, 1602]

        # check if the chat screen is still visible
        chat_b, chat_g, chat_r = new_resized_image[105, 1549]
        # if so, skip this frame if the picture is bright enough

        # if any of the color channels get lower, or all are the same
        # then you can stop searching

        get_score = int(new_b) + int(new_g) + int(new_r)

        expected_value = 140 + 151 + 164
        # print((expected_value - get_score))
        if (chat_b > 220) and (chat_g > 220) and (chat_r > 220) and (expected_value - get_score) < 20:
            continue

        if get_score > best_score:
            # if get_score > best_score and (( get_score - best_score ) > 10):
            best_score = get_score
            best_frame = rollback_frame

    return (best_score, best_frame)



def check_if_vote_screen_fading(frame):
    """
    checks if a voting screen is fading into the result of the vote
    used to more easily detect a voting screen where people have voted

    :param frame: frame of a video being analyzed
    :return: True if the screen is fading
    False otherwise
    """
    # check the pixel at   x = 1602 , y = 100

    width = 1920
    height = 1080

    img = cv2.resize(frame, (width, height))  # resize the image to (1920, 1080)

    x = 1602
    y = 100

    b, g, r = img[y, x]

    # regular grey:  r = 146, g = 156 , b = 170
    if (b < 120) and (r < 120) and (g < 120):
        return True

    return False

