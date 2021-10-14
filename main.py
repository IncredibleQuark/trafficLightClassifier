import cv2 as cv2
import helpers
import numpy as np



def one_hot_encode(label):
    # Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = [1, 0, 0]

    if label == "yellow":
        one_hot_encoded = [0, 1, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1]

    return one_hot_encoded


def standardize_input(image):
    # Resize image and pre-process so that all "standard" images are the same size
    standard_im = np.copy(image)
    width = 32
    height = 32
    standard_im = cv2.resize(standard_im, (width, height))
    return standard_im


def standardize(image_list):
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list


def create_feature(rgb_image):
    converted_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    green_result = np.copy(rgb_image)
    green_low = np.array([85, 100, 100])
    green_high = np.array([164, 255, 255])
    green_mask = cv2.inRange(converted_image, green_low, green_high)

    green_result[np.where(green_mask == 0)] = 0
    green_bright_sum = np.sum(green_result[:, :, 2])

    yellow_result = np.copy(rgb_image)
    yellow_low = np.array([15, 25, 25])
    yellow_high = np.array([83, 255, 255])
    yellow_mask = cv2.inRange(converted_image, yellow_low, yellow_high)

    yellow_result[np.where(yellow_mask == 0)] = 0
    yellow_bright_sum = np.sum(yellow_result[:, :, 2])

    red_result = np.copy(rgb_image)
    red_low_1 = np.array([150, 25, 25])
    red_high_1 = np.array([180, 255, 255])
    red_mask_1 = cv2.inRange(converted_image, red_low_1, red_high_1)
    # red_low_2 = np.array([2, 25, 25])
    # red_high_2 = np.array([7, 255, 255])
    # red_mask_2 = cv2.inRange(converted_image, red_low_2, red_high_2)

    red_result[np.where(red_mask_1 == 0)] = 0

    red_bright_sum = np.sum(red_result[:, :, 2])

    return [green_bright_sum, yellow_bright_sum, red_bright_sum]


def estimate_label(rgb_image):
    # Extract feature(s) from the RGB image and use those features to
    # classify the image and output a one-hot encoded label
    feature = create_feature(rgb_image)
    max_value = max(feature)

    max_index = feature.index(max_value)

    if feature[0] == feature[1] == feature[2]:
        return [1, 0, 0]

    if max_index == 0:
        return [0, 0, 1]

    if max_index == 1:
        return [0, 1, 0]

    return [1, 0, 0]


def main():
    IMAGE_DIR_TRAINING = "./traffic_light_images/training/"
    IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
    STANDARDIZED_LIST = standardize(IMAGE_LIST)

    IMAGE_DIR_TEST = "./traffic_light_images/test/"
    TEST_LIST = helpers.load_dataset(IMAGE_DIR_TEST)
    STANDARDIZED_TEST_LIST = standardize(TEST_LIST)

    for item in STANDARDIZED_LIST:
        label = estimate_label(item[0])
    # for item in STANDARDIZED_TEST_LIST:
    #     label = estimate_label(item[0])

        if label == [1, 0, 0]:
            print("Red light")
        elif label == [0, 1, 0]:
            print("Yellow light")
        else:
            print("Green light")


if __name__ == '__main__':
    main()
