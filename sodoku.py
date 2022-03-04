from cvzone.HandTrackingModule import HandDetector
import cv2
import time
import cvzone
import random
import numpy as np
from random import sample
import copy

def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha*img + beta, dtype=np.uint8)   # cast pixel values to int
    img_new[img_new > 255] = 255
    img_new[img_new < 0] = 0
    return img_new

# do mau
def FocalLength(pixel_length = 180, meansured_distance = 19, real_length = 6.5):
    focal_length = int((pixel_length*meansured_distance)/real_length)
    return focal_length

def Caculate_real(pixel_legth, real_length = 6.5):
    focal_legth = FocalLength()
    real_distance = int((focal_legth/pixel_legth)*real_length)
    return real_distance

class Board:
    def __init__(self, hide_number, base=3):
        self.hide_number = hide_number
        self.base = base
        self.side = base*base
        self.rBase = range(base)
        self.rows = [g * base + r for g in self.shuffle(self.rBase) for r in self.shuffle(self.rBase)]
        self.cols = [g * base + c for g in self.shuffle(self.rBase) for c in self.shuffle(self.rBase)]
        nums = self.shuffle(range(1, base * base + 1))

        self.result_board = [[nums[self.pattern(r, c)] for c in self.cols] for r in self.rows]
        self.board_play, self.defaut_0 = self.create_board()

    def shuffle(self, s):
        return sample(s, len(s))

    def pattern(self, r, c):
        return (self.base*(r%self.base)+r//self.base+c)%self.side

    def create_board(self):
        board = copy.deepcopy(self.result_board)
        hide_num = 0
        ij = []
        while hide_num < self.hide_number:
            i = random.randint(0,8)
            j = random.randint(0,8)
            if [i, j] in ij:
                continue

            if [i, j] not in ij:
                ij.append([i, j])
                board[i][j] = 0
                hide_num += 1
        return board, ij

    def is_win(self):
        if self.board_play == self.result_board:
            return True
        return False

    def update(self, x, y, value):
        x_new = None
        y_new = None
        for i in range(9):
            for j in range(9):
                # if self.board_play[i][j] == 0:
                if [i,j] in self.defaut_0:
                    if 10 + i*70 <= x < 10 + (i+1)*70:
                        x_new = i
                    if 10 + j*70 <= y < 10 + (j+1)*70:
                        y_new = j
        if x_new != None and y_new != None:
            self.board_play[x_new][y_new] = value

    def draw(self):
        space = 70
        for i in range(0, 9):
            for j in range(0, 9):
                if [i,j] in self.defaut_0:
                    cv2.rectangle(final_img, (10 + i * space + 5, 10 + j * space + 5),
                                  (10 + (i + 1) * space - 5, 10 + (j + 1) * space - 5), PURPLE_COLOR, 2)
                cv2.rectangle(final_img, (10 + i*space, 10 + j*space), (10 + (i+1)*space, 10+(j+1)*space), GRAY_COLOR, 2)
                if self.board_play[i][j] != 0:
                    # cv2.rectangle(final_img, (30+j*60, 20+i*60), (30+60+j*60, 20+60+i*60), (0,255,0), -1)
                    cv2.putText(final_img, str(self.board_play[i][j]), (40 + i * space, 50 +j*space), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)

class Number():
    def __init__(self, num=9):
        self.num = num
        self.lis = []
        for i in range(num):
            self.lis.append(i+1)
        self.rec = []

    def number_chose(self, x, y):
        value = 0
        if 750 <= x <= 820:
            for i in range(9):
                if 10 + i*70 <= y < 10 + (i+1)*70:
                    value = self.lis[i]
        return value

    def draw(self):
        space = 70
        for i in range(9):
            cv2.rectangle(final_img, (750, 10 + i*space), (820, 10 + (i+1)*space), GRAY_COLOR, 2)
            self.rec.append(((750, 10 + i*space), (820, 10 + (i+1)*space)))
        for i in range(self.num):
            cv2.putText(final_img, str(self.lis[i]), (750 + 30, 50 + i * space), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
pTime = 0
w_screen = 1000
h_screen = 800

PURPLE_COLOR = (255,0,255)
RED_COLOR = (0,0,255)
WHITE_COLOR = (255,255,255)
GREEN_COLOR = (0,255,0)
GRAY_COLOR = (127,127,127)

#bien tro choi
board = Board(hide_number=3)
number_chose = Number()

ls_enemy = []
push = False
push_reset = False
check = False
distance_detect = 30
range_finger = 40
is_chose_number = False

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.resize(img, dsize=(w_screen, h_screen))
    img = cv2.flip(img, 1)
    img_new = change_brightness(img, 0.2, 5)
    img_for_blur = copy.deepcopy(img_new)
    final_img = img_new

    # Find the hand and its landmarks
    # hands, img = detector.findHands(img)  # with draw
    hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList = hand1["lmList"]  # List of 21 Landmark points
        bbox = hand1["bbox"]  # Bounding box info x,y,w,h
        center_bbox = hand1['center']
        # handType1 = hand1["type"]  # Handtype Left or Right
        pixel_legth, info = detector.findDistance(lmList[5][:2], lmList[17][:2])
        x, y = lmList[8][0], lmList[8][1]

        roi = np.zeros(shape=img.shape[:2], dtype=np.uint8)
        roi = cv2.circle(roi, (x, y), range_finger, 255, -1)

        mask_roi = cv2.bitwise_and(img, img, mask=roi)

        blur_bg = cv2.blur(img_for_blur, (21,21))
        bg_roi = cv2.bitwise_and(blur_bg, blur_bg, mask=~roi)

        final_img = cv2.add(mask_roi, bg_roi)

        real_distance = Caculate_real(pixel_legth)

        if real_distance < distance_detect and push == False:
            if is_chose_number == False:
                number_value = number_chose.number_chose(x,y)
                if number_value == 0:
                    push = True
                else:
                    is_chose_number = True
            else:
                d_2_ngontay,_ = detector.findDistance(lmList[8][:2], lmList[12][:2])
                if d_2_ngontay < 40:
                    # kep ngon tay khong nam trong bang
                    board.update(x, y, number_value)
                    push = True
                else:
                    cv2.putText(final_img, str(number_value), (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, PURPLE_COLOR, 2)

        if real_distance < distance_detect and push_reset == False:
            push_reset = True
            if 400 < x < 600 and 680 < y < 750:
                board = Board(hide_number=3)

        if real_distance > distance_detect:
            push = False
            push_reset = False
            is_chose_number = False

        cvzone.putTextRect(final_img, f'{real_distance} cm', (x, y-30), scale=1, thickness=1)

        if 400 < x < 600 and 680 < y < 750:
            cv2.rectangle(final_img, (400 + 10, 680 + 10), (600-10,750-10), GREEN_COLOR, 2)

    if board.is_win():
        cv2.putText(final_img, 'WIN', (w_screen - 150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, RED_COLOR, 2)
    else:
        cv2.putText(final_img, 'PLAY', (w_screen - 150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, RED_COLOR, 2)

    cv2.putText(final_img, 'STATUS:', (w_screen - 150, 50), cv2.FONT_HERSHEY_COMPLEX, 1, GREEN_COLOR, 1)
    cv2.rectangle(final_img, (400,680), (600,750), GRAY_COLOR, 2)
    cv2.putText(final_img, 'RESET', (450,725), cv2.FONT_HERSHEY_COMPLEX, 1, GREEN_COLOR, 2)
    board.draw()
    number_chose.draw()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(final_img, f'FPS: {int(fps)}', (20, h_screen-20), cv2.FONT_HERSHEY_COMPLEX,
                0.7, (255, 0, 0), 1)
    cv2.imshow('Game', final_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()