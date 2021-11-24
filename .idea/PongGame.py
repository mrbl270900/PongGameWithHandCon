import pygame

import tensorflow.keras
import numpy as np
import cv2
import logging

def gen_labels():
    labels = {}
    with open("labels.txt", "r") as label:
        text = label.read()
        lines = text.split("\n")
        print(lines)
        for line in lines[0:-1]:
            hold = line.split(" ", 1)
            labels[hold[0]] = hold[1]
    return labels

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
image = cv2.VideoCapture(0)
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

"""
Create the array of the right shape to feed into the keras model
The 'length' or number of images you can put into the array is
determined by the first position in the shape tuple, in this case 1."""
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# A dict that stores the labels
labels = gen_labels()
# Load the model

pygame.init() #her intitere vi vores pygame samt vores fonter
pygame.font.init()
my_font = pygame.font.SysFont('Helvetica', 20)
hight = 300
width = 500
size_paddle_y = 50
size_paddle_x = 5
size_ball = 15
screen = pygame.display.set_mode([width, hight])
running = True
color = (255,255,255)
xy_c = [250,150]
speed_x = 5
speed_y = 5
count = 10
y1 = 0
bounce = 0
loged = False
logging.basicConfig(filename='data.log', level=logging.DEBUG)

while running: #her startes vores whille loop
    # Choose a suitable font
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame = image.read()
    frame = cv2.flip(frame, 1)
    # In case the image is not read properly
    if not ret:
        continue
    # Draw another rectangle in which the image to labelled is to be shown.
    frame2 = frame[80:360, 220:530]
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    frame2 = cv2.resize(frame2, (224, 224))
    # turn the image into a numpy array
    image_array = np.asarray(frame2)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    pred = model.predict(data)
    result = np.argmax(pred[0])

    # Print the predicted label into the screen.
    cv2.putText(frame,  "Label : " +
                labels[str(result)], (280, 400), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Frame', frame)


    screen.fill((0,0,0))
    if result == 1:
        y1 = y1 + 10
    elif result == 2:
        y1 = y1 - 10

    if y1 > hight-size_paddle_y:
        y1 = hight-size_paddle_y
    elif y1 < 1:
        y1 = 1

    pygame.draw.rect(screen, color, pygame.Rect(1,y1,size_paddle_x,size_paddle_y))
    pygame.draw.rect(screen, color, pygame.Rect(width-size_paddle_x,xy_c[1]-(size_paddle_y/2),size_paddle_x,size_paddle_y))
    bounce_text = my_font.render(str(bounce) + " Bounces", False, color)
    screen.blit(bounce_text,(width/2 - 30,25))

    xy_c[0] = xy_c[0] + speed_x
    xy_c[1] = xy_c[1] + speed_y

    if xy_c[1] > hight-size_ball or xy_c[1] < 0+size_ball:
        speed_y=speed_y*(-1)

    pygame.draw.circle(screen, color, xy_c,size_ball)

    if xy_c[0]+size_ball > width-size_paddle_x and count < 0:
        if xy_c[1]-size_ball < xy_c[1]+size_paddle_y and xy_c[1]+size_ball > xy_c[1]:
            speed_x = speed_x * (-1)
            count = 10

    if xy_c[0]-size_ball < 1+size_paddle_x and count < 0:
        if xy_c[1]-size_ball < y1+size_paddle_y and xy_c[1]+size_ball > y1:
            speed_x = speed_x * (-1)
            count = 10
            bounce = bounce + 1

    if xy_c[0]+size_ball > width+1 or xy_c[0]-size_ball < 0-1:
        if loged == False:
            logging.info("You got " + str(bounce) + " Bounces")
            loged = True
        xy_c[0]=1000
        lost_text_field = my_font.render("Game Over", False, color)
        play_again_text = my_font.render("Play Again by clicking your mouse", False, color)
        screen.blit(lost_text_field,(width/2 - 30,hight/2))
        screen.blit(play_again_text, (width/2 - 30,hight/2 + 25))
        for ev in pygame.event.get():
            if ev.type == pygame.MOUSEBUTTONDOWN:
                xy_c[0] = 250
                xy_c[1] = 150
                speed_x = 5
                speed_y = 5
                count = 10
                loged = False
                bounce = 0
    count = count - 1

    pygame.display.flip()
    for event in pygame.event.get(): # dette bruges til at stoppe pogramet når man trykker på krydset i pygame vinduet
        if event.type == pygame.QUIT:
            running = False
image.release()
cv2.destroyAllWindows()