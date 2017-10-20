# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:41:09 2017

@author: trent
"""

import numpy as np
from PIL import Image
import time
import pyautogui #To make screenshot same size
import pytesseract
from grabscreen import grab_screen
from click import click
from directkeys import PressKey,ReleaseKey, W, A, S, D
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import cv2

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
GameOver = Image.open("GameOver.jpg")
GameOverArray = np.array(GameOver)
GameOverArray = cv2.cvtColor(cv2.resize(GameOverArray, (80,80)), cv2.COLOR_BGR2GRAY)


#Parameters
keys = [W,A,S,D]
action_size = len(keys)
memory = deque(maxlen=2000)
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001

#Model
model = Sequential()
model.add(Dense(80, input_dim=80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(action_size, activation='linear'))
model.compile(loss='mse',
              optimizer=Adam(lr=learning_rate))


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def check_score(img):
    img = Image.fromarray(img)
    reward = pytesseract.image_to_string(img, lang='eng', boxes=False, \
    config='--psm 10 --eom 3 -c tessedit_char_whitelist=0123456789')
    return reward

def get_init_score():
    screen = grab_screen(region=(30,130,940,1030))
    crop_screen = screen[30:80,80:150]
    return crop_screen, screen

def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    act_values = model.predict(state)
    return np.argmax(act_values[0])

def remember(state, action, reward, next_state, done):
    memory.append((state,action,reward,next_state, done))

def replay(batch_size):
    global epsilon
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma *
                      np.amax(model.predict(next_state)[0]))
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def screen_record():
    #Set Initial Values
    current_score, screen = get_init_score()
    reward = 0
    state = grab_screen(region=(30,130,940,1030))
    state = cv2.cvtColor(cv2.resize(state, (80,80)), cv2.COLOR_BGR2GRAY)
    batch_size = 32
    done = False
    episodes = 0
    
    if mse(state, GameOverArray) < 500:
        print("Getting Ready to Start")
        click(400,800) #Click to Restart
        time.sleep(2)
    else:
        print('Please Start on Play again Screen')
        quit()
    
    while True:

        done = False
        
        #Get Current State
        state = grab_screen(region=(30,130,940,1030))
        score = state[30:80,80:150] #Grab Score from Image
        state = cv2.cvtColor(cv2.resize(state, (80,80)), cv2.COLOR_BGR2GRAY)
        
        action = act(state) #Perform Action
        PressKey(keys[action])
        next_state = grab_screen(region=(30,130,940,1030))

        if mse(current_score, score) > 500: #Different Score Detected
            reward = int(check_score(score))
            current_score = score
        if mse(state, GameOverArray) < 500: #We're most likely dead
#                reward = -10
            click(400,800) #Click to Restart
            done = True
            episodes += 1

        next_state = cv2.cvtColor(cv2.resize(next_state, (80,80)), cv2.COLOR_BGR2GRAY)
        remember(state, action, reward, next_state, done)
        
        if done:
            print('Reward: {}, Epsilon: {:.2}, Episodes: {}'.format(reward,epsilon, episodes))
        
        if len(memory) > batch_size:
            replay(batch_size)
            



         
        
screen_record()